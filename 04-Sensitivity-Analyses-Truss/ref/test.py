# 导入 OpenSeesPy 模块
from openseespy.opensees import *
import matplotlib.pyplot as plt

# 创建模型
wipe()  # 清空之前的模型数据
model('basic', '-ndm', 2, '-ndf', 3)  # 2D 模型, 每个节点有3个自由度 (平移x, 平移y, 转动)

# 定义材料参数
E = 2900.0  # 弹性模量 (将其减小，增加变形)
A = 10.0     # 截面积
I = 50.0    # 惯性矩 (将其减小，增加变形)

# 定义节点
node(1, 0.0, 0.0)  # 节点1坐标 (0, 0)
node(2, 144.0, 0.0)  # 节点2坐标 (144, 0)

# 支座条件，约束节点
fix(1, 1, 1, 1)  # 固定节点1
fix(2, 0, 1, 0)  # 节点2可以沿x方向自由移动

# 定义梁单元
geomTransf('Linear', 1)  # 定义线性几何变换
element('elasticBeamColumn', 1, 1, 2, A, E, I, 1)  # 创建弹性梁柱单元

# 定义荷载
timeSeries('Linear', 1)  # 线性时间载荷
pattern('Plain', 1, 1)  # 平面载荷模式
load(2, 100000.0, 0.0, 0.0)  # 在节点2施加更大的力 (10万N)

# 进行静力分析
integrator('LoadControl', 1.0)  # 载荷控制，增量1.0
analysis('Static')  # 静力分析类型
analyze(1)  # 进行一步分析

# 输出节点位移结果
disp_node2 = nodeDisp(2)  # 节点2的位移
print('Node 2 displacement:', disp_node2)

# 初始节点坐标
x_initial = [0.0, 144.0]
y_initial = [0.0, 0.0]

# 变形后的节点坐标 (只考虑x方向的位移)
x_deformed = [0.0, 144.0 + disp_node2[0]]
y_deformed = [0.0, 0.0]

# 使用 Matplotlib 绘制初始和变形后的结构
plt.figure()

# 绘制初始结构
plt.plot(x_initial, y_initial, 'bo-', label='Initial Structure')

# 绘制变形后的结构
plt.plot(x_deformed, y_deformed, 'ro-', label='Deformed Structure')

# 标注和设置图形属性
plt.legend()
plt.title("Structure Deformation")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.grid(True)
plt.show()

# 清除模型数据
wipe()
