import openseespy.opensees as ops
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------
# Modeling 
# ------------------------------

# Initialize OpenSees model
ops.wipe()
ops.model('basic', '-ndm', 3, '-ndf', 6)  # 3D model with 6 DOF (3 translations, 3 rotations)

# Define material properties (example for steel)
E = 12000  # Elastic modulus for steel (MPa)
A = 0.04   # Cross-sectional area (m^2)

# Bridge dimensions
span_length = 10.0  # Length of one span (m)
width_bridge = 6.0  # Width of the bridge (m)
height_pier = 8.0  # Height of the bridge piers (m)
num_spans = 4  # Number of spans

# Create nodes for all spans (both bottom and top nodes of bridge deck)
node_id = 1
element_id = 1  # Separate variable to keep track of element IDs
nodes_top = []  # List to store top nodes of the bridge
nodes_bottom = []  # List to store bottom nodes of the bridge
pier_nodes = []  # List to store all pier nodes

# Loop over each span to create the nodes for the bridge deck
for i in range(num_spans + 1):  # We need num_spans + 1 nodes in the X direction
    x = i * span_length  # X coordinate based on the span
    # Create bottom and top nodes for each span
    ops.node(node_id, x, 0.0, 0.0)  # Bottom node of the span
    ops.node(node_id + 1, x, width_bridge, 0.0)  # Top node of the span
    nodes_bottom.append(node_id)
    nodes_top.append(node_id + 1)
    node_id += 2  # Move to the next set of node IDs

# Define material for truss elements
ops.uniaxialMaterial("Elastic", 1, E)

# Create elements (trusses) between nodes within each span and between spans
for i in range(num_spans):
    # Nodes for the current span (bottom-left, bottom-right, top-left, top-right)
    bottom_left, top_left = nodes_bottom[i], nodes_top[i]  # Current span's left side nodes
    bottom_right, top_right = nodes_bottom[i + 1], nodes_top[i + 1]  # Next span's right side nodes

    # Create horizontal elements (span trusses)
    ops.element("truss", element_id, bottom_left, bottom_right, A, 1)  # Bottom truss (X direction)
    element_id += 1
    ops.element("truss", element_id, top_left, top_right, A, 1)  # Top truss (X direction)
    element_id += 1

    # Create vertical elements
    ops.element("truss", element_id, bottom_left, top_left, A, 1)  # Left vertical
    element_id += 1
    ops.element("truss", element_id, bottom_right, top_right, A, 1)  # Right vertical
    element_id += 1

    # Create diagonal elements (cross-bracing)
    ops.element("truss", element_id, bottom_left, top_right, A, 1)  # Diagonal from bottom-left to top-right
    element_id += 1
    ops.element("truss", element_id, top_left, bottom_right, A, 1)  # Diagonal from top-left to bottom-right
    element_id += 1

# Create bridge piers with X-bracing under each junction between spans
for i in range(1, num_spans):  # Piers under every junction between spans
    x = i * span_length  # Pier located at the junction of each span
    top_left = nodes_bottom[i]  # Use existing bottom node of the current span as pier top-left node
    top_right = nodes_top[i]  # Use existing top node of the current span as pier top-right node
    pier_bottom_left = node_id  # New bottom-left node for the pier
    pier_bottom_right = node_id + 1  # New bottom-right node for the pier
    ops.node(pier_bottom_left, x, 0.0, -height_pier)  # Create bottom node of the pier
    ops.node(pier_bottom_right, x, width_bridge, -height_pier)  # Bottom-right node of the pier
    pier_nodes.extend([pier_bottom_left, pier_bottom_right])

    # Connect the pier with vertical and diagonal elements (X-bracing)
    ops.element("truss", element_id, top_left, pier_bottom_left, A, 1)  # Left vertical pier
    element_id += 1
    ops.element("truss", element_id, top_right, pier_bottom_right, A, 1)  # Right vertical pier
    element_id += 1
    ops.element("truss", element_id, top_left, pier_bottom_right, A, 1)  # Diagonal left-top to right-bottom
    element_id += 1
    ops.element("truss", element_id, top_right, pier_bottom_left, A, 1)  # Diagonal right-top to left-bottom
    element_id += 1

    # Update node_id for next pier
    node_id += 2

# ------------------------------
# Apply boundary conditions (fixed constraints)
# ------------------------------
# Fix the four outermost nodes of the bridge deck (top layer)
ops.fix(nodes_top[0], 1, 1, 1, 0, 0, 0)  # Fix node at one end of the bridge (top-left)
ops.fix(nodes_top[-1], 1, 1, 1, 0, 0, 0)  # Fix node at the other end of the bridge (top-right)
ops.fix(nodes_bottom[0], 1, 1, 1, 0, 0, 0)  # Fix node at one end of the bridge (bottom-left)
ops.fix(nodes_bottom[-1], 1, 1, 1, 0, 0, 0)  # Fix node at the other end of the bridge (bottom-right)

# Fix all bottom pier nodes (both sides of the pier)
for pier_node in pier_nodes:
    ops.fix(pier_node, 1, 1, 1, 0, 0, 0)  # Fix pier nodes in all 3 translational and 3 rotational DOFs

# ------------------------------
# Dynamic Analysis (White noise loading)
# ------------------------------

# Define analysis parameters
duration = 30.0  # 30 seconds
time_step = 0.01  # Time step for dynamic analysis
num_steps = int(duration / time_step)  # Number of time steps

# Generate white noise excitation
np.random.seed(42)  # Set random seed for reproducibility
white_noise = np.random.normal(0, 1, num_steps)  # White noise with mean 0 and std deviation 1

# Apply white noise excitation at node 3 and node 7 (Y-direction)
ops.timeSeries('Path', 1, '-dt', time_step, '-values', *white_noise.tolist())
ops.pattern('Plain', 1, 1)
ops.load(3, 0.0, 1.0, 0.0)  # Apply load in Y-direction at node 3
ops.load(7, 0.0, 1.0, 0.0)  # Apply load in Y-direction at node 7

# Define dynamic analysis
ops.constraints('Plain')
ops.numberer('RCM')
ops.system('BandGeneral')
ops.algorithm('Newton')
ops.integrator('Newmark', 0.5, 0.25)  # Newmark-Beta method
ops.analysis('Transient')

# ------------------------------
# Observation and Recording
# ------------------------------

# Record displacements at node 3 and node 7
node_3_disp = []
node_7_disp = []

for step in range(num_steps):
    ops.analyze(1, time_step)
    node_3_disp.append(ops.nodeDisp(3))  # Record displacement at node 3
    node_7_disp.append(ops.nodeDisp(7))  # Record displacement at node 7

# Convert displacement data to numpy arrays for easier manipulation
node_3_disp = np.array(node_3_disp)
node_7_disp = np.array(node_7_disp)

# ------------------------------
# Visualization (Model and Displacement Curves)
# ------------------------------

# Get coordinates of the nodes
def get_node_coords(node_ids):
    x_coords = [ops.nodeCoord(node_id)[0] for node_id in node_ids]
    y_coords = [ops.nodeCoord(node_id)[1] for node_id in node_ids]
    z_coords = [ops.nodeCoord(node_id)[2] for node_id in node_ids]
    return x_coords, y_coords, z_coords

# Plot the bridge model
fig = plt.figure(figsize=(14, 6))
ax = fig.add_subplot(121, projection='3d')

# Plot the truss elements (connecting all node pairs)
for element in range(1, element_id):
    nodes = ops.eleNodes(element)
    x_coords, y_coords, z_coords = get_node_coords(nodes)
    ax.plot(x_coords, y_coords, z_coords, 'r-')  # Plot the truss as a red line

# Plot nodes (bridge deck and pier nodes)
all_node_ids = nodes_bottom + nodes_top + pier_nodes  # Include pier nodes as well
x_coords, y_coords, z_coords = get_node_coords(all_node_ids)
ax.scatter(x_coords, y_coords, z_coords, c='b', marker='o', label='Nodes')

# Add node labels
for node_id in all_node_ids:
    x, y, z = ops.nodeCoord(node_id)
    ax.text(x, y, z, f'{node_id}', fontsize=8, color='black')  # Node number

# Set equal aspect ratio for all axes
max_range = np.array([max(x_coords) - min(x_coords), max(y_coords) - min(y_coords), max(z_coords) - min(z_coords)]).max() / 2.0
mid_x = (max(x_coords) + min(x_coords)) * 0.5
mid_y = (max(y_coords) + min(y_coords)) * 0.5
mid_z = (max(z_coords) + min(z_coords)) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range)
ax.set_zlim(mid_z - max_range)

# Labels and title
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('4-span Bridge Deck with Piers and X-bracing')

# Plot displacement results for node 3 and node 7
time = np.linspace(0, duration, num_steps)

ax2 = fig.add_subplot(122)

# Node 3 displacements
ax2.plot(time, node_3_disp[:, 1], label='Node 3 Y-displacement', color='b')
# Node 7 displacements
ax2.plot(time, node_7_disp[:, 1], label='Node 7 Y-displacement', color='g')

ax2.set_title('Displacement over Time (Y-axis)')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Displacement (m)')
ax2.legend()

plt.tight_layout()
plt.show()
