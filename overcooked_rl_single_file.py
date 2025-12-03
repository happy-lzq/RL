import numpy as np
import random
import time
import os
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # 设置后端为 Agg，避免在 Ubuntu 无显示器环境下报错
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import datetime

# ==========================================
# 配置参数
# ==========================================
GRID_SIZE = 5
MAX_STEPS = 100
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 100000  # 增加衰减步数，让智能体在训练前中期保持探索 (对应 10000 episodes)
TARGET_UPDATE = 50    # 稍微降低目标网络更新频率，提高稳定性
LR = 1e-4  # 降低学习率，防止震荡
MEMORY_SIZE = 50000  # 增大记忆库，保留更多成功经验
NUM_EPISODES = 5000  # 5000轮足够了，关键是存下最好的
MEMORY_SIZE = 10000
NUM_EPISODES = 3000 # 适当减少以便快速演示，实际训练建议 3000+

# 动作定义
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_INTERACT = 4
ACTION_STAY = 5
NUM_ACTIONS = 6

# 物品定义
ITEM_NONE = 0
ITEM_ONION = 1
ITEM_DISH = 2
ITEM_SOUP = 3

# 地图元素
TILE_EMPTY = 0
TILE_ONION_DISPENSER = 1
TILE_DISH_DISPENSER = 2
TILE_POT = 3
TILE_SERVING = 4
TILE_COUNTER = 5

# 锅的状态
POT_EMPTY = 0
POT_COOKING = 1
POT_READY = 2
COOK_TIME = 3

# ==========================================
# 1. 环境定义
# ==========================================
class SimpleOvercookedEnv:
    def __init__(self):
        self.grid_size = GRID_SIZE
        self.agents = []
        self.pot_status = POT_EMPTY
        self.pot_timer = 0
        self.pot_contents = 0
        
        # 初始化地图布局
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.grid[0, 0] = TILE_ONION_DISPENSER
        self.grid[0, GRID_SIZE-1] = TILE_POT
        self.grid[GRID_SIZE-1, 0] = TILE_DISH_DISPENSER
        self.grid[GRID_SIZE-1, GRID_SIZE-1] = TILE_SERVING
        
        self.pos_onion = (0, 0)
        self.pos_pot = (0, GRID_SIZE-1)
        self.pos_dish = (GRID_SIZE-1, 0)
        self.pos_serve = (GRID_SIZE-1, GRID_SIZE-1)

    def reset(self):
        self.agents = [
            {'x': 1, 'y': 1, 'holding': ITEM_NONE},
            {'x': 2, 'y': 2, 'holding': ITEM_NONE}
        ]
        self.pot_status = POT_EMPTY
        self.pot_timer = 0
        self.pot_contents = 0
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        obs = []
        for agent in self.agents:
            obs.extend([agent['x']/GRID_SIZE, agent['y']/GRID_SIZE, agent['holding']/3.0])
        obs.extend([self.pot_status/2.0, self.pot_timer/COOK_TIME, self.pot_contents])
        return np.array(obs, dtype=np.float32)

    def step(self, actions):
        self.steps += 1
        rewards = [0.0, 0.0]
        
        # 锅逻辑
        if self.pot_status == POT_COOKING:
            self.pot_timer += 1
            if self.pot_timer >= COOK_TIME:
                self.pot_status = POT_READY
                self.pot_timer = 0
        
        # 移动逻辑
        new_positions = []
        for i, agent in enumerate(self.agents):
            action = actions[i]
            nx, ny = agent['x'], agent['y']
            
            if action == ACTION_UP: nx = max(0, nx - 1)
            elif action == ACTION_DOWN: nx = min(GRID_SIZE - 1, nx + 1)
            elif action == ACTION_LEFT: ny = max(0, ny - 1)
            elif action == ACTION_RIGHT: ny = min(GRID_SIZE - 1, ny + 1)
            
            if self.grid[nx, ny] != TILE_EMPTY:
                nx, ny = agent['x'], agent['y']
            
            new_positions.append((nx, ny))

        if new_positions[0] == new_positions[1]:
            new_positions[0] = (self.agents[0]['x'], self.agents[0]['y'])
            new_positions[1] = (self.agents[1]['x'], self.agents[1]['y'])
        elif new_positions[0] == (self.agents[1]['x'], self.agents[1]['y']) and \
             new_positions[1] == (self.agents[0]['x'], self.agents[0]['y']):
            new_positions[0] = (self.agents[0]['x'], self.agents[0]['y'])
            new_positions[1] = (self.agents[1]['x'], self.agents[1]['y'])

        for i, pos in enumerate(new_positions):
            self.agents[i]['x'], self.agents[i]['y'] = pos

        # 交互逻辑
        shared_reward = 0
        task_completed = False
        for i, agent in enumerate(self.agents):
            if actions[i] == ACTION_INTERACT:
                r, completed = self._handle_interact(i)
                shared_reward += r
                if completed: task_completed = True
        
        shared_reward -= 0.01 # 时间惩罚
        
        rewards = [shared_reward, shared_reward]
        done = self.steps >= MAX_STEPS or task_completed
        
        return self._get_obs(), rewards, done, {'task_completed': task_completed}

    def _handle_interact(self, agent_idx):
        agent = self.agents[agent_idx]
        ax, ay = agent['x'], agent['y']
        
        target_pos = None
        target_type = TILE_EMPTY
        
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            tx, ty = ax+dx, ay+dy
            if 0 <= tx < GRID_SIZE and 0 <= ty < GRID_SIZE:
                if self.grid[tx, ty] != TILE_EMPTY:
                    target_pos = (tx, ty)
                    target_type = self.grid[tx, ty]
                    break
        
        if target_pos is None:
            return 0, False

        if target_type == TILE_ONION_DISPENSER:
            if agent['holding'] == ITEM_NONE:
                agent['holding'] = ITEM_ONION
                return 0.1, False
        
        elif target_type == TILE_DISH_DISPENSER:
            if agent['holding'] == ITEM_NONE:
                agent['holding'] = ITEM_DISH
                return 0.1, False

        elif target_type == TILE_POT:
            if agent['holding'] == ITEM_ONION:
                if self.pot_status == POT_EMPTY:
                    agent['holding'] = ITEM_NONE
                    self.pot_contents += 1
                    self.pot_status = POT_COOKING
                    return 1.0, False
            
            elif agent['holding'] == ITEM_DISH:
                if self.pot_status == POT_READY:
                    agent['holding'] = ITEM_SOUP
                    self.pot_status = POT_EMPTY
                    self.pot_contents = 0
                    return 1.0, False
        
        elif target_type == TILE_SERVING:
            if agent['holding'] == ITEM_SOUP:
                agent['holding'] = ITEM_NONE
                return 10.0, True # 任务完成
                
        return 0, False

    def render_text(self):
        display_grid = [['.' for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        display_grid[0][0] = 'O'
        display_grid[0][GRID_SIZE-1] = 'P'
        display_grid[GRID_SIZE-1][0] = 'D'
        display_grid[GRID_SIZE-1][GRID_SIZE-1] = 'S'
        
        if self.pot_status == POT_COOKING: display_grid[0][GRID_SIZE-1] = 'p'
        if self.pot_status == POT_READY: display_grid[0][GRID_SIZE-1] = '!'
        
        for i, agent in enumerate(self.agents):
            sym = str(i+1)
            if agent['holding'] == ITEM_ONION: sym = '🌰'
            elif agent['holding'] == ITEM_DISH: sym = '🥣'
            elif agent['holding'] == ITEM_SOUP: sym = '🍲'
            display_grid[agent['x']][agent['y']] = sym
            
        print("-" * (GRID_SIZE + 2))
        for row in display_grid:
            print("|" + "".join(row) + "|")
        print("-" * (GRID_SIZE + 2))

# ==========================================
# 2. 强化学习组件
# ==========================================
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128) # 增加网络容量
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, device):
        self.device = device
        self.action_size = action_size
        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.steps_done = 0

    def select_action(self, state, eval_mode=False):
        if eval_mode:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                return self.policy_net(state_t).max(1)[1].view(1, 1)

        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            np.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        
        if sample > eps_threshold:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                return self.policy_net(state_t).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_size)]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return None
        
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([torch.FloatTensor(s).unsqueeze(0) for s in batch.next_state if s is not None]).to(self.device)
        state_batch = torch.cat([torch.FloatTensor(s).unsqueeze(0) for s in batch.state]).to(self.device)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        return loss.item()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# ==========================================
# 3. 评估与可视化工具
# ==========================================
def plot_metrics(rewards, losses, success_rates, filename="training_metrics.png"):
    """绘制训练曲线"""
    plt.figure(figsize=(15, 5))
    
    # 1. 奖励曲线
    plt.subplot(1, 3, 1)
    plt.plot(rewards, label='Episode Reward', alpha=0.3)
    # 计算移动平均
    window = 50
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), moving_avg, label=f'Moving Avg ({window})', color='red')
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    
    # 2. Loss 曲线
    plt.subplot(1, 3, 2)
    plt.plot(losses, label='Loss', color='orange', alpha=0.5)
    plt.title('Training Loss')
    plt.xlabel('Step')
    plt.yscale('log')
    
    # 3. 成功率曲线
    plt.subplot(1, 3, 3)
    plt.plot(success_rates, label='Success Rate', color='green')
    plt.title('Success Rate (Last 100 eps)')
    plt.xlabel('Episode')
    plt.ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"训练图表已保存至: {filename}")

import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# ...existing code...

def create_demo_gif(env, agent1, agent2, device, filename="demo.gif"):
    """生成演示 GIF (图片素材版 - 完美解决乱码)"""
    print("正在生成演示 GIF...")
    state = env.reset()
    
    # 预先运行并收集状态数据
    states_data = []
    states_data.append(copy_env_state(env))
    
    for _ in range(MAX_STEPS):
        with torch.no_grad():
            s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            a1 = agent1.select_action(state, eval_mode=True).item()
            a2 = agent2.select_action(state, eval_mode=True).item()
        
        state, _, done, _ = env.step([a1, a2])
        states_data.append(copy_env_state(env))
        if done: break

    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 加载图片素材
    try:
        img_onion = mpimg.imread('assets/onion.png')
        img_pot_empty = mpimg.imread('assets/pot_empty.png')
        img_pot_cooking = mpimg.imread('assets/pot_cooking.png')
        img_pot_ready = mpimg.imread('assets/pot_ready.png')
        img_dish = mpimg.imread('assets/dish.png')
        img_serve = mpimg.imread('assets/serve.png')
        img_chef_man = mpimg.imread('assets/chef_man.png')
        img_chef_woman = mpimg.imread('assets/chef_woman.png')
    except FileNotFoundError:
        print("错误：未找到图片素材，请确保 assets 文件夹存在且包含所需图片。")
        return

    def add_image(ax, img, x, y, zoom=0.6):
        im = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(im, (x, y), frameon=False)
        ax.add_artist(ab)

    def animate(i):
        ax.clear()
        restore_env_state(env, states_data[i])
        
        # 设置坐标轴
        ax.set_xlim(-0.5, GRID_SIZE-0.5)
        ax.set_ylim(GRID_SIZE-0.5, -0.5)
        ax.set_xticks(np.arange(GRID_SIZE))
        ax.set_yticks(np.arange(GRID_SIZE))
        ax.grid(True, color='#cccccc', linestyle='--', alpha=0.3)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        # 绘制背景
        ax.add_patch(patches.Rectangle((-0.5, -0.5), GRID_SIZE, GRID_SIZE, color='#f9f9f9'))
        
        # --- 绘制静态设施 (图片) ---
        
        # 1. Onion (0,0)
        add_image(ax, img_onion, 0, 0, zoom=0.7)
        
        # 2. Pot (0, 4) -> Plot(4, 0)
        pot_img = img_pot_empty
        if env.pot_status == POT_COOKING: pot_img = img_pot_cooking
        if env.pot_status == POT_READY: pot_img = img_pot_ready
        add_image(ax, pot_img, GRID_SIZE-1, 0, zoom=0.7)
        
        # 3. Dish (4, 0) -> Plot(0, 4)
        add_image(ax, img_dish, 0, GRID_SIZE-1, zoom=0.7)
        
        # 4. Serve (4, 4) -> Plot(4, 4)
        add_image(ax, img_serve, GRID_SIZE-1, GRID_SIZE-1, zoom=0.7)

        # --- 绘制智能体 ---
        for idx, agent in enumerate(env.agents):
            # Agent Image
            chef_img = img_chef_man if idx == 0 else img_chef_woman
            add_image(ax, chef_img, agent['y'], agent['x'], zoom=0.8)
            
            # 编号
            ax.text(agent['y']-0.3, agent['x']-0.3, f"A{idx+1}", ha='center', va='center', fontsize=8, weight='bold', color='#333')
            
            # 持有物品 (在右下角显示小图标)
            if agent['holding'] != ITEM_NONE:
                item_img = None
                if agent['holding'] == ITEM_ONION: item_img = img_onion
                elif agent['holding'] == ITEM_DISH: item_img = img_dish
                elif agent['holding'] == ITEM_SOUP: item_img = img_pot_ready # 用汤的图代替
                
                if item_img is not None:
                    add_image(ax, item_img, agent['y']+0.25, agent['x']+0.25, zoom=0.35)

        ax.set_title(f"Step: {i} | Pot: {['Empty','Cooking','Ready'][env.pot_status]}", fontsize=12)
        return []

    anim = FuncAnimation(fig, animate, frames=len(states_data), interval=600)
    anim.save(filename, writer='pillow')
    print(f"演示 GIF 已保存至: {filename}")
    plt.close()

def copy_env_state(env):
    """深拷贝环境状态用于回放"""
    import copy
    return {
        'agents': copy.deepcopy(env.agents),
        'pot_status': env.pot_status,
        'pot_timer': env.pot_timer,
        'pot_contents': env.pot_contents
    }

def restore_env_state(env, state_dict):
    env.agents = state_dict['agents']
    env.pot_status = state_dict['pot_status']
    env.pot_timer = state_dict['pot_timer']
    env.pot_contents = state_dict['pot_contents']

# ==========================================
# 4. 主程序
# ==========================================
def main():
    print("启动 Overcooked RL 训练 (V2 - 增强版)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    env = SimpleOvercookedEnv()
    state_dim = 9 
    
    agent1 = DQNAgent(state_dim, NUM_ACTIONS, device)
    agent2 = DQNAgent(state_dim, NUM_ACTIONS, device)
    
    # 记录指标
    rewards_history = []
    loss_history = []
    success_history = [] # 1 if task completed else 0
    
    start_time = time.time()
    
    # 用于保存最优模型
    best_avg_reward = -float('inf')
    best_agent1_state = None
    best_agent2_state = None

    for i_episode in range(NUM_EPISODES):
        state = env.reset()
        total_reward = 0
        episode_loss = 0
        loss_count = 0
        
        for t in range(MAX_STEPS):
            action1 = agent1.select_action(state)
            action2 = agent2.select_action(state)
            
            next_state, rewards, done, info = env.step([action1.item(), action2.item()])
            
            reward_t = torch.tensor([rewards[0]], device=device)
            
            if done: next_state = None
            
            agent1.memory.push(state, action1, next_state, reward_t, done)
            agent2.memory.push(state, action2, next_state, reward_t, done)
            
            state = next_state if next_state is not None else state
            total_reward += rewards[0]
            
            l1 = agent1.optimize_model()
            l2 = agent2.optimize_model()
            
            if l1 is not None: 
                episode_loss += (l1 + l2)/2
                loss_count += 1
            
            if done:
                success_history.append(1 if info.get('task_completed') else 0)
                break
        else:
            success_history.append(0)
        
        # 更新目标网络
        if i_episode % TARGET_UPDATE == 0:
            agent1.update_target_net()
            agent2.update_target_net()
            
        rewards_history.append(total_reward)
        loss_history.append(episode_loss / max(1, loss_count))
        
        if (i_episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            success_rate = np.mean(success_history[-100:])
            print(f"Episode {i_episode+1}/{NUM_EPISODES} | Avg Reward: {avg_reward:.2f} | Success Rate: {success_rate:.2f} | Epsilon: {agent1.steps_done}")
            
            # 保存最优模型状态
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_agent1_state = agent1.policy_net.state_dict()
                best_agent2_state = agent2.policy_net.state_dict()
                print(f"  >>> 新纪录！保存最优模型 (Avg Reward: {best_avg_reward:.2f})")

    print(f"训练完成！耗时: {time.time() - start_time:.1f}s")
    
    # 加载最优模型进行演示和绘图
    if best_agent1_state is not None and best_agent2_state is not None:
        print(f"加载最优模型 (Avg Reward: {best_avg_reward:.2f})...")
        agent1.policy_net.load_state_dict(best_agent1_state)
        agent2.policy_net.load_state_dict(best_agent2_state)
    
    # 1. 绘制图表
    plot_metrics(rewards_history, loss_history, 
                 [np.mean(success_history[max(0, i-100):i+1]) for i in range(len(success_history))])
    
    # 2. 生成 GIF
    try:
        create_demo_gif(env, agent1, agent2, device)
    except Exception as e:
        print(f"生成 GIF 失败 (可能是缺少依赖): {e}")
        print("跳过 GIF 生成，直接运行终端演示。")

    # 3. 终端演示
    print("\n开始终端演示...")
    state = env.reset()
    env.render_text()
    time.sleep(1)
    
    for t in range(MAX_STEPS):
        with torch.no_grad():
            s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            a1 = agent1.select_action(state, eval_mode=True).item()
            a2 = agent2.select_action(state, eval_mode=True).item()
        
        print(f"\nStep {t+1}")
        state, rewards, done, _ = env.step([a1, a2])
        env.render_text()
        time.sleep(0.5)
        
        if done:
            print("演示结束!")
            break

if __name__ == "__main__":
    main()