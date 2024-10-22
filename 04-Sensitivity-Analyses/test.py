from openseespy.opensees import *
import matplotlib.pyplot as plt
import numpy as np

# 初始化模型
wipe()
model('basic', '-ndm', 2, '-ndf', 3)  # 2D 模型, 每个节点有3个自由度 (平移x, 平移y, 转动)

# 定义桥梁参数
span_length = 144.0  # 桥梁长度 (单位: m)
num_nodes = 5  # 节点数量
node_spacing = span_length / (num_nodes - 1)

# 定义材料参数
E = 29.0  # 弹性模量
A = 10.0     # 截面积
I = 500.0    # 惯性矩

# 定义桥梁节点
for i in range(num_nodes):
    node(i+1, i * node_spacing, 0.0)

# 支座条件，约束两端
fix(1, 1, 1, 1)  # 固定节点1 (左端)
fix(num_nodes, 0, 1, 0)  # 固定节点 (右端)

# 定义梁单元
geomTransf('Linear', 1)  # 线性几何变换
for i in range(1, num_nodes):
    element('elasticBeamColumn', i, i, i+1, A, E, I, 1)

# 定义列车参数
train_speed = 10.0  # 列车速度 (单位: m/s)
train_axle_weights = [100.0, 150.0, 200.0]  # 车轴重量 (单位: kN)
train_axle_distances = [0.0, 10.0, 20.0]  # 车轴之间的距离 (单位: m)
time_step = 0.1  # 时间步长 (单位: s)
total_time = span_length / train_speed  # 列车完全通过桥梁所需的时间

# 定义时间序列
timeSeries('Linear', 1)

# 定义移动荷载模式
pattern('Plain', 1, 1)

# 静力分析设置
constraints('Plain')  # 约束处理器
numberer('RCM')  # 使用反向Cuthill-McKee算法进行编号
system('BandGen')  # 使用广义带状矩阵求解器
test('NormDispIncr', 1.0e-6, 10)  # 设定收敛标准
algorithm('Newton')  # 使用牛顿迭代法
integrator('LoadControl', 1.0)  # 载荷控制，增量1.0
analysis('Static')  # 静力分析类型

# 模拟列车移动
num_steps = int(total_time / time_step)
node_loads = []

for step in range(num_steps):
    current_time = step * time_step
    current_position = train_speed * current_time  # 当前列车的头部位置
    print(f"Time: {current_time}, Train position: {current_position}")

    # 清除之前的荷载
    loadConst('-time', 0.0)

    # 遍历车轴，计算每个车轴的荷载位置
    for axle, weight in zip(train_axle_distances, train_axle_weights):
        axle_position = current_position - axle

        # 找到与该车轴最近的桥节点
        if 0 <= axle_position <= span_length:
            closest_node = int(axle_position / node_spacing) + 1

            # 施加荷载到最近的节点
            if closest_node <= num_nodes:
                load(closest_node, weight, 0.0, 0.0)

    # 进行一步静力分析
    analyze(1)

    # 记录每一步节点的位移
    node_loads.append(nodeDisp(num_nodes))

# 输出最终结果
print('Final displacements at node:', node_loads)

# 可视化节点位移 (可选)
plt.plot(np.arange(0, total_time, time_step), node_loads, label='Node Displacement Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.title('Displacement of the Last Node Over Time as Train Passes')
plt.grid(True)
plt.legend()
plt.show()

# 清除模型数据
wipe()
