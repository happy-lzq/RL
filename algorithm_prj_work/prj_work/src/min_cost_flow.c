#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#define MAX_NODES 1000
#define MAX_EDGES 10000
#define INF 0x3f3f3f3f

// 边结构体
typedef struct {
    int to;         // 目标节点
    int next;       // 下一条边的索引
    int capacity;   // 容量
    int flow;       // 当前流量
    int cost;       // 单位流量费用
} Edge;

Edge edges[MAX_EDGES * 2]; // 边数组，包含反向边
int head[MAX_NODES];       // 邻接表头
int edge_cnt = 0;          // 边计数

int dist[MAX_NODES];       // 源点到各点的最短距离（费用）
int in_queue[MAX_NODES];   // SPFA 队列标记
int pre_edge[MAX_NODES];   // 记录路径上的边
int pre_node[MAX_NODES];   // 记录路径上的前驱节点

// 初始化图
void init_graph() {
    memset(head, -1, sizeof(head));
    edge_cnt = 0;
}

// 添加边
void add_edge(int u, int v, int cap, int cost) {
    // 正向边
    edges[edge_cnt].to = v;
    edges[edge_cnt].capacity = cap;
    edges[edge_cnt].flow = 0;
    edges[edge_cnt].cost = cost;
    edges[edge_cnt].next = head[u];
    head[u] = edge_cnt++;

    // 反向边 (容量为0，费用为负)
    edges[edge_cnt].to = u;
    edges[edge_cnt].capacity = 0;
    edges[edge_cnt].flow = 0;
    edges[edge_cnt].cost = -cost;
    edges[edge_cnt].next = head[v];
    head[v] = edge_cnt++;
}

// SPFA 算法寻找最小费用路径
// 返回 1 如果存在增广路，否则返回 0
int spfa(int s, int t, int n) {
    for (int i = 0; i <= n; i++) {
        dist[i] = INF;
        in_queue[i] = 0;
        pre_edge[i] = -1;
        pre_node[i] = -1;
    }

    int queue[MAX_NODES * 100]; // 简单循环队列
    int front = 0, rear = 0;

    dist[s] = 0;
    queue[rear++] = s;
    in_queue[s] = 1;

    while (front != rear) {
        int u = queue[front++];
        in_queue[u] = 0;

        for (int i = head[u]; i != -1; i = edges[i].next) {
            int v = edges[i].to;
            // 如果残余容量大于0 且 能够松弛
            if (edges[i].capacity > edges[i].flow && dist[v] > dist[u] + edges[i].cost) {
                dist[v] = dist[u] + edges[i].cost;
                pre_node[v] = u;
                pre_edge[v] = i;

                if (!in_queue[v]) {
                    queue[rear++] = v;
                    in_queue[v] = 1;
                }
            }
        }
    }

    return dist[t] != INF;
}

// 最小费用最大流主函数
void min_cost_max_flow(int s, int t, int n, int *max_flow, int *min_cost) {
    *max_flow = 0;
    *min_cost = 0;

    while (spfa(s, t, n)) {
        // 寻找当前增广路上的最小残余容量
        int push = INF;
        int curr = t;
        while (curr != s) {
            int edge_idx = pre_edge[curr];
            int available = edges[edge_idx].capacity - edges[edge_idx].flow;
            if (available < push) {
                push = available;
            }
            curr = pre_node[curr];
        }

        // 更新流量和费用
        *max_flow += push;
        *min_cost += push * dist[t];
        
        curr = t;
        while (curr != s) {
            int edge_idx = pre_edge[curr];
            edges[edge_idx].flow += push;       // 正向边增加流量
            edges[edge_idx ^ 1].flow -= push;   // 反向边减少流量
            curr = pre_node[curr];
        }
    }
}

int main(int argc, char *argv[]) {
    init_graph();

    int n, m, s, t;
    FILE *input = stdin;

    // 检查是否提供了输入文件
    if (argc > 1) {
        input = fopen(argv[1], "r");
        if (input == NULL) {
            fprintf(stderr, "无法打开文件: %s\n", argv[1]);
            return 1;
        }
        printf("正在从文件 %s 读取数据...\n", argv[1]);
    } else {
        printf("=== 最小费用最大流 (Minimum Cost Maximum Flow) ===\n");
        printf("提示: 您也可以使用 ./min_cost_flow <filename> 来从文件读取数据\n");
        printf("请输入节点数 (n), 边数 (m), 源点 (s), 汇点 (t): ");
    }

    if (fscanf(input, "%d %d %d %d", &n, &m, &s, &t) != 4) {
        fprintf(stderr, "读取输入失败\n");
        if (input != stdin) fclose(input);
        return 1;
    }

    if (input == stdin) printf("请输入 %d 条边 (u v capacity cost):\n", m);
    
    for (int i = 0; i < m; i++) {
        int u, v, cap, cost;
        if (fscanf(input, "%d %d %d %d", &u, &v, &cap, &cost) != 4) break;
        add_edge(u, v, cap, cost);
    }

    if (input != stdin) fclose(input);

    int max_flow, min_cost;
    min_cost_max_flow(s, t, n, &max_flow, &min_cost);

    printf("\n结果:\n");
    printf("最大流量 (Max Flow): %d\n", max_flow);
    printf("最小费用 (Min Cost): %d\n", min_cost);

    return 0;
}
