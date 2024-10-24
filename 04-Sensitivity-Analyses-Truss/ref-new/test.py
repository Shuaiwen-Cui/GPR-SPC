import openseespy.opensees as ops
import numpy as np
import matplotlib.pyplot as plt

# 初始化OpenSees模型
ops.wipe()
ops.model('basic', '-ndm', 3, '-ndf', 6)  # 三维6自由度模型

# 定义节点
ops.node(1, 0.0, 0.0, 0.0)  # A点，固定节点
ops.node(2, 5.0, 5.0, 0.0)  # B点，中间节点
ops.node(3, 10.0, 0.0, 0.0)  # C点，固定节点

# 定义约束条件 (铰接，A和C节点)
ops.fix(1, 1, 1, 1, 1, 1, 1)  # 节点A
ops.fix(3, 1, 1, 1, 1, 1, 1)  # 节点C

# 定义材料 (弹性材料，E为2100 MPa，A为0.02 m^2)
E = 2100.0e6  # 弹性模量，单位：Pa
A = 0.02      # 截面面积，单位：m^2
ops.uniaxialMaterial('Elastic', 1, E)

# 定义桁架单元
ops.element('Truss', 1, 1, 2, A, 1)  # A到B的桁架杆
ops.element('Truss', 2, 2, 3, A, 1)  # B到C的桁架杆

# 定义质量 (假设节点B有一个质量)
mass = 1000.0  # 单位：kg
ops.mass(2, mass, mass, mass, 0.0, 0.0, 0.0)

# 定义阻尼
xi = 0.05  # 阻尼比
omega = np.sqrt(E / mass)  # 自振频率估计
alphaM = 2 * xi * omega  # 质量比例阻尼
ops.rayleigh(alphaM, 0.0, 0.0, 0.0)

# 定义瞬时荷载 (在节点B上施加一个恒定的冲击荷载)
P = 10000.0  # 冲击力大小，单位：N
ops.timeSeries('Constant', 1)
ops.pattern('Plain', 1, 1)
ops.load(2, P, 0.0, 0.0, 0.0, 0.0, 0.0)  # 节点B施加冲击荷载

# 选择求解器和积分器
ops.system('BandGeneral')
ops.numberer('RCM')
ops.constraints('Transformation')
ops.integrator('Newmark', 0.5, 0.25)
ops.algorithm('Newton')
ops.analysis('Transient')

# 进行动态分析
dt = 0.01  # 时间步长，单位：秒
tMax = 5.0  # 最大分析时间
numSteps = int(tMax / dt)
time = np.arange(0, tMax, dt)
response = []

# 开始分析
for i in range(numSteps):
    ops.analyze(1, dt)
    dispB = ops.nodeDisp(2)  # 获取节点B的位移
    response.append(dispB[0])  # 记录x方向的位移

# 绘制位移时间曲线
plt.figure()
plt.plot(time, response)
plt.xlabel('Time [s]')
plt.ylabel('Displacement at Node B [m]')
plt.title('Free Vibration Response with Damping')
plt.grid(True)
plt.show()
