import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置中文字体 - 直接指定字体文件路径以确保加载
font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
prop = fm.FontProperties(fname=font_path)

# 数据
text_lengths = [100000, 500000, 1000000, 5000000, 10000000]
times = [0.000402, 0.002049, 0.004140, 0.020750, 0.041000]

# 创建图表
plt.figure(figsize=(10, 6))
plt.plot(text_lengths, times, marker='o', linestyle='-', color='b', label='KMP 运行时间')

# 添加标题和标签
plt.title('KMP 算法性能分析: 运行时间 vs 文本长度', fontsize=16, fontproperties=prop)
plt.xlabel('文本长度 (N)', fontsize=12, fontproperties=prop)
plt.ylabel('耗时 (秒)', fontsize=12, fontproperties=prop)

# 添加网格
plt.grid(True, linestyle='--', alpha=0.7)

# 添加图例
plt.legend(prop=prop)

# 优化坐标轴显示
plt.ticklabel_format(style='plain', axis='x') # 禁用科学计数法
plt.xticks(text_lengths, rotation=45)

# 保存图片
plt.tight_layout()
plt.savefig('kmp_performance.png', dpi=300)
print("图表已保存为 kmp_performance.png")
