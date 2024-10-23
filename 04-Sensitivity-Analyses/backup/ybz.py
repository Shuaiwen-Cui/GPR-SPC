def model_4(E_1, E_2, E_3, E_4, density2, density3, density4, density5, density6):
    import openseespy.opensees as ops
    import openseespy.postprocessing.Get_Rendering as opsplt
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy.signal import resample
    # 材料属性
    E = 3e4 # 板的模量
    density = 2.7e3 # 柱的密度
    # 每层柱的模量
    G_1 = E_1/(2*(0.33+1));
    G_2 = E_2/(2*(0.33+1));
    G_3 = E_3/(2*(0.33+1));
    G_4 = E_4/(2*(0.33+1));

    # G_2 = E/(2*(0.25+1));
    density_1 = density * 1e-12;  # 钢的密度（kg/m^3）
    # 每层板的密度
    density_2 = density2 * 1e-12;
    density_3 = density3 * 1e-12;
    density_4 = density4 * 1e-12;
    density_5 = density5 * 1e-12;
    density_6 = density6 * 1e-12;
    # 几何参数
    Ly = 200;   # 楼板长度 (mm)
    Lx = 180;   # 楼板宽度 (mm)
    Lz = 21;    # 楼板厚度 (mm)
    column_width = 30;  # 柱的宽度 (mm)
    column_thickness = 2;  # 柱的厚度 (mm)
    column_height = 150;   # 柱的高度 (mm)
    m_2 = Lx * Ly * Lz * density_2  # slab mass
    m_3 = Lx * Ly * Lz * density_3
    m_4 = Lx * Ly * Lz * density_4
    m_5 = Lx * Ly * Lz * density_5
    m_6 = Lx * Ly * Lz * density_6

    m_1 = column_width * column_thickness * column_height * density_1 # column mass

    # 创建模型
    ops.wipe()
    ops.model('basic', '-ndm', 3, '-ndf', 6)

    # 创建节点和分配质量
    for i in range(9):
        # 创建底部和顶部节点
        ops.node(101+i, 0, 0, i*column_height/2)
        ops.node(201+i, Lx, 0, i*column_height/2)
        ops.node(301+i, Lx, Ly, i*column_height/2)
        ops.node(401+i, 0, Ly, i*column_height/2)

    # fix the bottom
    ops.fix(101, 1, 1, 1, 1, 1, 1)       
    ops.fix(201, 1, 1, 1, 1, 1, 1)
    ops.fix(301, 1, 1, 1, 1, 1, 1)
    ops.fix(401, 1, 1, 1, 1, 1, 1)

    transfTag = 1   
    sec_Tag_1 = 1 
    mat_Tag_1 = 1
    sec_Tag_2 = 2 
    mat_Tag_2 = 2
    sec_Tag_3 = 3 
    mat_Tag_3 = 3
    sec_Tag_4 = 4 
    mat_Tag_4 = 4

    ops.uniaxialMaterial('Elastic', mat_Tag_1, E_1)
    ops.uniaxialMaterial('Elastic', mat_Tag_2, E_2)
    ops.uniaxialMaterial('Elastic', mat_Tag_3, E_3)
    ops.uniaxialMaterial('Elastic', mat_Tag_4, E_4)
    numSubdivY = 8
    numSubdivZ = 2 
    integrationTag_1 = 1
    integrationTag_2 = 2
    integrationTag_3 = 3
    integrationTag_4 = 4
    ops.geomTransf('Linear', transfTag, 1, 0, 0)

    ops.section('Fiber',sec_Tag_1,'-GJ',G_1)
    ops.patch('rect', mat_Tag_1, numSubdivY, numSubdivZ, -column_width/2, -column_thickness/2, column_width/2, column_thickness/2)  
    ops.beamIntegration('Lobatto', integrationTag_1, sec_Tag_1, 5)
    for i in range(2):
        ops.element('dispBeamColumn', i+121, i+101, i+102, transfTag, integrationTag_1)
        ops.element('dispBeamColumn', i+221, i+201, i+202, transfTag, integrationTag_1)
        ops.element('dispBeamColumn', i+321, i+301, i+302, transfTag, integrationTag_1)
        ops.element('dispBeamColumn', i+421, i+401, i+402, transfTag, integrationTag_1)

    ops.section('Fiber',sec_Tag_2,'-GJ',G_2)
    ops.patch('rect', mat_Tag_2, numSubdivY, numSubdivZ, -column_width/2, -column_thickness/2, column_width/2, column_thickness/2)
    ops.beamIntegration('Lobatto', integrationTag_2, sec_Tag_2, 5)
    for i in range(2,4):
        ops.element('dispBeamColumn', i+121, i+101, i+102, transfTag, integrationTag_2)
        ops.element('dispBeamColumn', i+221, i+201, i+202, transfTag, integrationTag_2)
        ops.element('dispBeamColumn', i+321, i+301, i+302, transfTag, integrationTag_2)
        ops.element('dispBeamColumn', i+421, i+401, i+402, transfTag, integrationTag_2)

    ops.section('Fiber',sec_Tag_3,'-GJ',G_3)
    ops.patch('rect', mat_Tag_3, numSubdivY, numSubdivZ, -column_width/2, -column_thickness/2, column_width/2, column_thickness/2)
    ops.beamIntegration('Lobatto', integrationTag_3, sec_Tag_3, 5)
    for i in range(4,6):
        ops.element('dispBeamColumn', i+121, i+101, i+102, transfTag, integrationTag_3)
        ops.element('dispBeamColumn', i+221, i+201, i+202, transfTag, integrationTag_3)
        ops.element('dispBeamColumn', i+321, i+301, i+302, transfTag, integrationTag_3)
        ops.element('dispBeamColumn', i+421, i+401, i+402, transfTag, integrationTag_3)

    ops.section('Fiber',sec_Tag_4,'-GJ',G_4)
    ops.patch('rect', mat_Tag_4, numSubdivY, numSubdivZ, -column_width/2, -column_thickness/2, column_width/2, column_thickness/2)
    ops.beamIntegration('Lobatto', integrationTag_4, sec_Tag_4, 5)
    for i in range(6,8):
        ops.element('dispBeamColumn', i+121, i+101, i+102, transfTag, integrationTag_4)
        ops.element('dispBeamColumn', i+221, i+201, i+202, transfTag, integrationTag_4)
        ops.element('dispBeamColumn', i+321, i+301, i+302, transfTag, integrationTag_4)
        ops.element('dispBeamColumn', i+421, i+401, i+402, transfTag, integrationTag_4)

    # add mass for column
    for i in range(4): #N_story+1
        ops.mass(i*2+102, m_1, m_1, m_1, 0, 0, 0)
        ops.mass(i*2+202, m_1, m_1, m_1, 0, 0, 0)
        ops.mass(i*2+302, m_1, m_1, m_1, 0, 0, 0)
        ops.mass(i*2+402, m_1, m_1, m_1, 0, 0, 0)

    transfTag = 2
    sec_tag_1 = 5
    mat_tag_1 = 5
    sec_tag_2 = 6
    mat_tag_2 = 6
    sec_tag_3 = 7
    mat_tag_3 = 7
    sec_tag_4 = 8
    mat_tag_4 = 8
    sec_tag_5 = 9
    mat_tag_5 = 9
    nu = 0.25 # Poisson's ratio of GFRP(Glass Fiber Reinforced Plastic)
    thickness = 21 # 
    # 定义材料
    ops.nDMaterial('ElasticIsotropic', mat_tag_1, E, nu, density_2)
    ops.nDMaterial('ElasticIsotropic', mat_tag_2, E, nu, density_3)
    ops.nDMaterial('ElasticIsotropic', mat_tag_3, E, nu, density_4)
    ops.nDMaterial('ElasticIsotropic', mat_tag_4, E, nu, density_5)
    ops.nDMaterial('ElasticIsotropic', mat_tag_5, E, nu, density_6)
    # numSubdivY = 5
    # numSubdivZ = 5 
    # ops.section('ElasticMembranePlateSection',sec_Tag, E_2, nu, thickness, density_2)
    ops.section('PlateFiber', sec_tag_1, mat_tag_1, thickness)
    ops.section('PlateFiber', sec_tag_2, mat_tag_2, thickness)
    ops.section('PlateFiber', sec_tag_3, mat_tag_3, thickness)
    ops.section('PlateFiber', sec_tag_4, mat_tag_4, thickness)
    ops.section('PlateFiber', sec_tag_5, mat_tag_5, thickness)
    # ops.patch('rect', matTag_2, numSubdivY, numSubdivZ, 0, 0, Ly, Lz)
    taglist = [sec_tag_1, sec_tag_2, sec_tag_3, sec_tag_4, sec_tag_5]
    masslist = [m_2, m_3, m_4, m_5, m_6]
    ops.geomTransf('Linear', transfTag, 0, 0, 1)

    for i in range(5):
        # 创建楼板 (壳单元)
        ops.element("ShellMITC4", 601+i, 101+i*2, 201+i*2, 301+i*2, 401+i*2, taglist[i], transfTag)

    # add mass for slab
    for i in range(5):
        ops.mass(101+i*2, masslist[i]/4, masslist[i]/4, masslist[i]/4, 0, 0, 0)
        ops.mass(201+i*2, masslist[i]/4, masslist[i]/4, masslist[i]/4, 0, 0, 0)
        ops.mass(301+i*2, masslist[i]/4, masslist[i]/4, masslist[i]/4, 0, 0, 0)
        ops.mass(401+i*2, masslist[i]/4, masslist[i]/4, masslist[i]/4, 0, 0, 0)
    
    # 可视化模型
    # opsplt.plot_model()

    numEigen = 6  # 要求的模态数量
    eigenvalues = ops.eigen(numEigen)
    mode_shapes = []
    node_list = [103, 105, 107, 109]
    for i in range(1, 7):
        mode_shape = []
        for node in node_list:
            ux = ops.nodeEigenvector(node, i, 1)
            # uy = ops.nodeEigenvector(node, i, 2)
            # uz = ops.nodeEigenvector(node, i, 3)
            mode_shape.append(ux)
        mode_shapes.append(mode_shape)
        # print(mode_shape)  
    mode_shapes = np.array(mode_shapes)
    mode_shapes = mode_shapes.T
    # print(mode_shapes)

    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)
    # add weight for column
    for i in range(4):
        ops.load(102+2*i, 0, 0, -9.8*m_1, 0, 0, 0) 
        ops.load(202+2*i, 0, 0, -9.8*m_1, 0, 0, 0) 
        ops.load(302+2*i, 0, 0, -9.8*m_1, 0, 0, 0) 
        ops.load(402+2*i, 0, 0, -9.8*m_1, 0, 0, 0)

    # add weight for slab
    for g in range(5):
        ops.load(101+2*i, 0, 0, -9.8*(masslist[g]/4), 0, 0, 0) 
        ops.load(201+2*i, 0, 0, -9.8*(masslist[g]/4), 0, 0, 0) 
        ops.load(301+2*i, 0, 0, -9.8*(masslist[g]/4), 0, 0, 0) 
        ops.load(401+2*i, 0, 0, -9.8*(masslist[g]/4), 0, 0, 0)

    # add earthquake time-domain data
    csv_file = 'earthquake_load.csv'  # CSV文件路径
    earthquake = pd.read_csv(csv_file)
    time_series = earthquake['A0'].values  # 假设CSV中有加速度列
    dt = earthquake['Time'].values[0]  # 假设每行时间步长一致
    time_series = time_series[:1200]
    # print(time_series)
    # print(dt)

    ops.timeSeries('Path', 2, '-values', *time_series, '-dt', dt)
    # ops.pattern('MultipleSupport', 2)
    ops.pattern('UniformExcitation', 2, 1, '-accel', 2)  # 在X方向施加地震加速度
    # ops.imposedMotion(103, 1, 2)  # add load in x axis for node 101
    # ops.imposedMotion(201, 1, 2)  # add load in x axis for node 201
    # ops.imposedMotion(301, 1, 2)  # add load in x axis for node 301
    # ops.imposedMotion(401, 1, 2)  # add load in x axis for node 401

    # 记录柱子的某个位置应变，例如，截面中心位置 (xLoc, yLoc) 为 0.0
    # ops.recorder('Element', '-file', 'strainOutput3D.txt', '-time', '-ele', 321, 'section', 1, 'fiber', 1, 'strain')
    # 定义分析类型
    ops.integrator('Newmark', 0.5, 0.25)
    ops.system('BandGeneral')
    ops.numberer('RCM')
    ops.constraints('Plain')
    ops.test('NormDispIncr', 1.0e-6, 10)
    ops.algorithm('Newton')
    ops.analysis('Transient')
    # 记录并输出加速度数据
    accel_data = []
    accel_data2 = []

    data_60Hz = resample(earthquake['Acc5'].values, 18000)
    Top_acc = data_60Hz
    # 执行分析
    nSteps = int(1200)
    for i in range(nSteps):
        ops.analyze(1, dt)
        accel_data.append(ops.nodeAccel(209, 1))
        # if int(time/recordfreq) * recordfreq == time:
        #     ops.record()
    print(accel_data)
    # print(Top_acc[:1000])
    plt.plot(earthquake['Time'].values[:1200], accel_data, 'b')
    plt.plot(earthquake['Time2'].values[:40960], earthquake['Acc5'].values[:40960], 'r')
    # plt.plot((earthquake['dt'].values)[:10000], earthquake['acceleration2'].values[:10000], 'r')
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [m/s^2]')
    plt.title('Acceleration Time History at Node 1')
    plt.grid(True)
    plt.show()

    ops.wipe()

    # 读取应变输出
    # with open('strainOutput3D.txt', 'r') as f:
    #     strain_data = f.readlines()
    #     strain = float(strain_data[9][2:])

    # 绘制前四阶模态
    # fig, axs = plt.subplots(3, 2, figsize=(12, 6))
    
    # floor_tag = [0, 1, 2, 3, 4]

    # axs[0, 0].plot(mode_shapes[0, :], floor_tag, marker='o', linestyle='-', color='b', label=f'Mode {1}')
    # axs[0, 0].set_xlabel('Displacement')
    # axs[0, 0].set_ylabel('Floor')
    # axs[0, 0].legend()
    # plt.grid(True)

    # axs[0, 1].plot(mode_shapes[1, :], floor_tag, marker='o', linestyle='-', color='b', label=f'Mode {2}')
    # axs[0, 1].set_xlabel('Displacement')
    # axs[0, 1].set_ylabel('Floor')
    # axs[0, 1].legend()
    # plt.grid(True)

    # axs[1, 0].plot(mode_shapes[2, :], floor_tag, marker='o', linestyle='-', color='b', label=f'Mode {3}')
    # axs[1, 0].set_xlabel('Displacement')
    # axs[1, 0].set_ylabel('Floor')
    # axs[1, 0].legend()
    # plt.grid(True)

    # axs[1, 1].plot(mode_shapes[3, :], floor_tag, marker='o', linestyle='-', color='b', label=f'Mode {4}')
    # axs[1, 1].set_xlabel('Displacement')
    # axs[1, 1].set_ylabel('Floor')
    # axs[1, 1].legend()
    # plt.grid(True)

    # axs[2, 0].plot(mode_shapes[4, :], floor_tag, marker='o', linestyle='-', color='b', label=f'Mode {4}')
    # axs[2, 0].set_xlabel('Displacement')
    # axs[2, 0].set_ylabel('Floor')
    # axs[2, 0].legend()
    # plt.grid(True)

    # axs[2, 1].plot(mode_shapes[5, :], floor_tag, marker='o', linestyle='-', color='b', label=f'Mode {4}')
    # axs[2, 1].set_xlabel('Displacement')
    # axs[2, 1].set_ylabel('Floor')
    # axs[2, 1].legend()
    # plt.grid(True)

    # plt.tight_layout()
    # plt.show()
    # 输出模态频率（单位：Hz）
    frequencies = np.sqrt(eigenvalues) / (2 * np.pi)
    return frequencies, mode_shapes

model_4(37217, 32825, 33840, 36481, 1406, 1421, 1431, 1449, 1431)