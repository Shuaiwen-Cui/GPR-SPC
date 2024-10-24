# =================================================
# DEPENDENCIES
# =================================================

import openseespy.opensees as ops
import matplotlib.pyplot as plt
import numpy as np

# =================================================
# Function to extract coordinates of nodes
# =================================================

def get_node_coords(node_ids):
    x_coords = [ops.nodeCoord(node_id)[0] for node_id in node_ids]
    y_coords = [ops.nodeCoord(node_id)[1] for node_id in node_ids]
    z_coords = [ops.nodeCoord(node_id)[2] for node_id in node_ids]
    return np.array(x_coords), np.array(y_coords), np.array(z_coords)

# =================================================
# MODEL CONSTRUCTION 
# =================================================

# [INITIALIZATION]
ops.wipe()  # Reset OpenSees model
ops.model('basic', '-ndm', 3, '-ndf', 3)  # 3D model with 3 DOF (X, Y, Z translations)

# [NODES CONSTRUCTION]

# <Configuration>

# -Bridge dimensions
length_span = 10.0  # Length of one span (m)
width_bridge = 6.0  # Width of the bridge (m)
height_pier = 8.0  # Height of the bridge piers (m)
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

# <Create Nodes for the Bridge>

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

# [MATERIAL CONSTRUCTION]

# <Material Properties>
E = 12000  # Elastic modulus for timber (MPa)
A = 0.04   # Cross-sectional area (m^2)

# <Material Definition>
ops.uniaxialMaterial("Elastic", 1, E)

# [ELEMENT CONSTRUCTION]

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

# Loop over each span to create the elements for the bridge piers
for i in range(num_span - 1):
    # Nodes
    left_high_node = node_deck_left[i + 1]
    right_high_node = node_deck_right[i + 1]
    left_low_node = node_pier_left[i]
    right_low_node = node_pier_right[i]
    
    # Z-direction truss elements
    ops.element("truss", element_id, left_high_node, left_low_node, A, 1)
    element_pier.append(element_id)
    element_id += 1
    ops.element("truss", element_id, right_high_node, right_low_node, A, 1)
    element_pier.append(element_id)
    element_id += 1
    
    # Diagonal truss elements
    ops.element("truss", element_id, left_high_node, right_low_node, A, 1)
    element_pier.append(element_id)
    element_id += 1
    ops.element("truss", element_id, right_high_node, left_low_node, A, 1)
    element_pier.append(element_id)
    element_id += 1

# - Elements
element_list = element_deck + element_pier

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

# =================================================
# WHITE NOISE EXCITATION (APPLIED TO ALL NODES IN X, Y, Z DIRECTIONS)
# =================================================

dt = 0.01  # Time step in seconds
time_steps = 1000  # Number of time steps
time = np.linspace(0, dt * time_steps, time_steps)

# Generate white noise excitation for X, Y, Z directions
white_noise_X = np.random.normal(0, 1, time_steps) * 100  # White noise in X direction
white_noise_Y = np.random.normal(0, 1, time_steps) * 100  # White noise in Y direction
white_noise_Z = np.random.normal(0, 1, time_steps) * 100  # White noise in Z direction

# Apply the same white noise to all nodes in X, Y, Z directions
for node in node_list:
    # Apply white noise in X, Y, Z directions
    ops.timeSeries('Path', node, '-dt', dt, '-values', *white_noise_X)
    ops.pattern('Plain', node, node)
    ops.load(node, 1.0, 0.0, 0.0)  # White noise in X

    ops.timeSeries('Path', node + 1000, '-dt', dt, '-values', *white_noise_Y)
    ops.pattern('Plain', node + 1000, node + 1000)
    ops.load(node, 0.0, 1.0, 0.0)  # White noise in Y

    ops.timeSeries('Path', node + 2000, '-dt', dt, '-values', *white_noise_Z)
    ops.pattern('Plain', node + 2000, node + 2000)
    ops.load(node, 0.0, 0.0, 1.0)  # White noise in Z

# =================================================
# RECORD AND ANALYZE RESPONSE
# =================================================

# Recorder for displacements of all nodes (X, Y, Z)
ops.recorder('Node', '-file', 'disp_all_nodes.txt', '-time', '-nodeRange', 1, len(node_list), '-dof', 1, 2, 3, 'disp')

# Define analysis options
ops.algorithm('Newton')
ops.system('BandGeneral')
ops.numberer('Plain')
ops.constraints('Plain')
ops.integrator('Newmark', 0.5, 0.25)
ops.analysis('Transient')

# Run analysis
ops.analyze(time_steps, dt)

# =================================================
# PLOT THE RESULTS
# =================================================

# Load the recorded displacements for all nodes
disp_all_nodes = np.loadtxt('disp_all_nodes.txt')

# Create a figure for side-by-side comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Plot the model structure
x_coords, y_coords, z_coords = get_node_coords(node_list)
ax1.scatter(x_coords, y_coords, z_coords, c='b', marker='o', label='Nodes')

# Add node labels
for node_id in node_list:
    x, y, z = ops.nodeCoord(node_id)
    # Ensure the coordinates are valid before adding text
    if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
        ax1.text(x, y, z, f'{node_id}', fontsize=10, color='black')  # Node number

# Plot the elements (lines connecting nodes)
for elem_id in element_list:
    nodes = ops.eleNodes(elem_id)
    x_elem, y_elem, z_elem = get_node_coords(nodes)
    ax1.plot(x_elem, y_elem, z_elem, 'r-')  # Red lines for elements

# Set axis limits and the viewing angle
ax1.set_xlim([0, 40])
ax1.set_ylim([-1, 7])
ax1.set_zlim([-10, 2])
ax1.view_init(elev=30, azim=-45)
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_zlabel('Z (m)')
ax1.set_title('3D Truss Bridge Structure')

# Right: Plot the displacement response of all nodes in Z direction (for example)
ax2.plot(disp_all_nodes[:, 0], disp_all_nodes[:, 3], label='Node 1 Z-displacement', color='blue')
ax2.set_xlabel('Time (seconds)')
ax2.set_ylabel('Displacement (meters)')
ax2.set_title('Z-displacement Response at All Nodes')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()

# =================================================
# CLEAN UP
# =================================================
ops.wipe()
