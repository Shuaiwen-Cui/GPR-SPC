
# ------------------------------
# DEPENDENCIES
# ------------------------------
import openseespy.opensees as ops
import matplotlib.pyplot as plt
import numpy as np

def trial(reduction, noise_level):
    
    # ensure reduction is between 0 and 1
    if reduction < 0 or reduction > 1:
        raise ValueError("Reduction factor must be between 0 and 1")
    
    # ensure noise_level is above 0
    if noise_level < 0:
        raise ValueError("Noise level must be above 0")
    
    # ------------------------------
    # MODEL CONSTRUCTION
    # ------------------------------
    
    ops.wipe()  # Reset OpenSees model
    ops.model('basic', '-ndm', 3, '-ndf', 3)  # 3D model with 3 DOF (X, Y, Z translations)
    
    # -Bridge dimensions
    length_span = 6.0  # Length of one span (m)
    width_bridge = 5.0  # Width of the bridge (m)
    height_pier = 5.0  # Height of the bridge piers (m)
    num_span = 4  # Number of spans
    
    # <Helper Variables>
    node_id = 1  # Index for node and its initial value
    node_list = []  # List to store all nodes
    node_deck = []  # List to store all nodes of the bridge deck
    node_deck_left = []  # List to store left nodes of the bridge deck
    node_deck_right = []  # List to store right nodes of the bridge deck
    node_pier = []  # List to store all nodes of the bridge piers
    node_pier_left = []  # List to store left nodes of the bridge piers
    node_pier_right = []  # List to store right nodes of the bridge piers
    
    # - Deck

    # Loop over each span to create the nodes for the bridge deck
    for i in range(num_span + 1):
        x = i * length_span  # X coordinate based on the span
        ops.node(node_id, x, 0.0, 0.0)  # Left node of the span
        ops.node(node_id + 1, x, width_bridge, 0.0)  # Right node of the span
        node_deck_left.append(node_id)
        node_deck_right.append(node_id + 1)
        node_deck.append(node_id)
        node_deck.append(node_id + 1)
        node_id += 2  # Move to the next
    
    # - Piers

    # Loop over each span to create the nodes for the bridge piers
    for i in range(1, num_span):
        x = i * length_span  # X coordinate based on the span
        ops.node(node_id, x, 0.0, -height_pier)  # Left node of the span
        ops.node(node_id + 1, x, width_bridge, -height_pier)  # Right node of the span
        node_pier_left.append(node_id)
        node_pier_right.append(node_id + 1)
        node_pier.append(node_id)
        node_pier.append(node_id + 1)
        node_id += 2  # Move to the next
    
    # - Nodes
    node_list = node_deck + node_pier
    
    # <Material Properties>
    E = 20000  # Elastic modulus for timber (MPa)
    A = 0.12   # Cross-sectional area (m^2)
    
    # <Material Definition>
    ops.uniaxialMaterial("Elastic", 1, E)
    
    # <Helper Variables>
    element_id = 1  # Index for element and its initial value
    element_list = []  # List to store all elements
    element_deck = []  # List to store all elements of the bridge deck
    element_pier = []  # List to store all elements of the bridge piers
    
    # <Create Elements (Truss) for the Bridge>

    # - Deck Truss Elements

    # Loop over each span to create the elements for the bridge deck
    for i in range(num_span):
        # Nodes
        left_near_node = node_deck_left[i]
        right_near_node = node_deck_right[i]
        left_far_node = node_deck_left[i + 1]
        right_far_node = node_deck_right[i + 1]

        # Y-direction truss elements
        if i == 0:
            ops.element("truss", element_id, left_near_node, right_near_node, A, 1)
            element_deck.append(element_id)
            element_id += 1
        ops.element("truss", element_id, left_far_node, right_far_node, A, 1)
        element_deck.append(element_id)
        element_id += 1

        # X-direction truss elements
        ops.element("truss", element_id, left_near_node, left_far_node, A, 1)
        element_deck.append(element_id)
        element_id += 1
        ops.element("truss", element_id, right_near_node, right_far_node, A, 1)
        element_deck.append(element_id)
        element_id += 1

        # Diagonal truss elements
        ops.element("truss", element_id, left_near_node, right_far_node, A, 1)
        element_deck.append(element_id)
        element_id += 1 
        ops.element("truss", element_id, left_far_node, right_near_node, A, 1)
        element_deck.append(element_id)
        element_id += 1

    # - Pier Truss Elements

    # strength reduction factor

    # Loop over each span to create the elements for the bridge piers
    for i in range(num_span - 1):
        # Nodes
        left_high_node = node_deck_left[i + 1]
        right_high_node = node_deck_right[i + 1]
        left_low_node = node_pier_left[i]
        right_low_node = node_pier_right[i]

        if i == 2:
            reduction_factor = reduction # passed by the user when calling the function
        else:
            reduction_factor = 1


        # Z-direction truss elements
        ops.element("truss", element_id, left_high_node, left_low_node, reduction_factor*3.5*A, 1)
        element_pier.append(element_id)
        element_id += 1
        ops.element("truss", element_id, right_high_node, right_low_node, reduction_factor*3.5*A, 1)
        element_pier.append(element_id)
        element_id += 1

        # Diagonal truss elements
        ops.element("truss", element_id, left_high_node, right_low_node, reduction_factor*3.5*A, 1)
        element_pier.append(element_id)
        element_id += 1
        ops.element("truss", element_id, right_high_node, left_low_node, reduction_factor*3.5*A, 1)
        element_pier.append(element_id)
        element_id += 1

    # - Elements
    element_list = element_deck + element_pier

    mass_x = 100
    mass_y = 100
    mass_z = 100

    # Set Up Mass for Nodes, assume
    for node in node_list:
        ops.mass(node, mass_x, mass_y, mass_z)

    # Modal analysis to obtain the first two modal frequencies (eigenvalues)
    eigenvalues = ops.eigen(2)  # Compute the first two eigenvalues
    omega1 = np.sqrt(eigenvalues[0])  # First modal frequency (angular)
    omega2 = np.sqrt(eigenvalues[1])  # Second modal frequency (angular)

    # print(f"First Modal Frequency: {omega1:.2f} rad/s")
    # print(f"Second Modal Frequency: {omega2:.2f} rad/s")

    # Damping ratio, assuming a x% damping ratio
    xi = 0.05  

    # Calculate Rayleigh damping coefficients
    alphaM = 2 * xi * omega1 * omega2 / (omega1 + omega2)
    betaK = 2 * xi / (omega1 + omega2)

    # Add Rayleigh damping to the model
    ops.rayleigh(alphaM, betaK, 0.0, 0.0)

    # [BOUNDARY CONDITIONS]

    # 1. Fix the leftmost and rightmost deck nodes in XYZ directions
    ops.fix(node_deck_left[0], 1, 1, 1)  # Fix node at left end
    ops.fix(node_deck_right[0], 1, 1, 1)  # Fix node at left end
    ops.fix(node_deck_left[-1], 1, 1, 1)  # Fix node at right end
    ops.fix(node_deck_right[-1], 1, 1, 1)  # Fix node at right end

    # 2. Fix all the pier bottom nodes (both left and right)
    for i in range(len(node_pier_left)):
        ops.fix(node_pier_left[i], 1, 1, 1)  # Fix left bottom pier nodes
        ops.fix(node_pier_right[i], 1, 1, 1)  # Fix right bottom pier nodes

    # ------------------------------
    # LOAD APPLICATION
    # ------------------------------

    # Apply random white noise dynamic load to nodes 5
    dt = 0.01  # Time step in seconds
    time_steps = 2000  # Number of time steps
    time = np.linspace(0, dt * time_steps, time_steps)

    # Generate white noise excitation for nodes 3 and 8
    white_noise = np.random.normal(0, 1, time_steps)  # White noise at node 5

    # Apply white noise to node 3 (left side of the second span)
    ops.timeSeries('Path', 1, '-dt', dt, '-values', *white_noise)
    ops.pattern('Plain', 1, 1)
    # ops.load(5, 0.3, 0.05, 0.8)  # Apply white noise in Z direction
    for node in node_deck:
        ops.load(node, 0.0, 0.0, 0.5 * np.random.normal(0, 1))

    # Define gravity
    g = 9.81  # m/s^2, gravitational acceleration

    # Apply gravity to all nodes in the Z direction
    for node in node_list:

        # Compute gravity force in Z direction: F_z = mass_z * g
        gravity_force_z = mass_z * g

        # Apply the gravity load to the node in Z direction
        ops.load(node, 0.0, 0.0, -gravity_force_z)  # X, Y, Z components
    
    # ------------------------------
    # ANALYSES & RECORDING
    # ------------------------------

    # Recorder for nodes 3 and 8 displacements
    ops.recorder('Node', '-file', 'disp_node_3.txt', '-time', '-node', 3, '-dof', 1, 2, 3, 'disp')
    ops.recorder('Node', '-file', 'disp_node_8.txt', '-time', '-node', 8, '-dof', 1, 2, 3, 'disp')

    # Define analysis options
    ops.algorithm('Newton')
    ops.system('BandGeneral')
    ops.numberer('Plain')
    ops.constraints('Plain')
    ops.integrator('Newmark', 0.5, 0.25)
    ops.analysis('Transient')

    # Run analysis
    ops.analyze(time_steps, dt)
    
    # Clean up
    ops.wipe()

    # ------------------------------
    # POST-PROCESSING
    # ------------------------------
    point_A = np.loadtxt('disp_node_3.txt')
    point_B = np.loadtxt('disp_node_8.txt')
    
    disp_XA = point_A[:, 1]
    disp_XB = point_B[:, 1]
    
    disp_YA = point_A[:, 2]
    disp_YB = point_B[:, 2]
    
    disp_ZA = point_A[:, 3]
    disp_ZB = point_B[:, 3]
    
    # add noise to the displacement data
    if noise_level != 0:
        noise_A = np.random.normal(0, noise_level, len(disp_XA))
        noise_B = np.random.normal(0, noise_level, len(disp_XB))
    else:
        noise_A = np.zeros(len(disp_XA))
        noise_B = np.zeros(len(disp_XB))
    
    disp_XA += noise_A
    disp_XB += noise_B
    
    disp_YA += noise_A
    disp_YB += noise_B
    
    disp_ZA += noise_A
    disp_ZB += noise_B
    
    max_XA = np.max(np.abs(disp_XA))
    max_XB = np.max(np.abs(disp_XB))

    max_YA = np.max(np.abs(disp_YA))
    max_YB = np.max(np.abs(disp_YB))
    
    max_ZA = np.max(np.abs(disp_ZA))
    max_ZB = np.max(np.abs(disp_ZB))
    
    return max_XA, max_XB, max_YA, max_YB, max_ZA, max_ZB