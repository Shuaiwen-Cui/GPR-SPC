import openseespy.opensees as ops
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ------------------------------
# Modeling (Model Geometry and Elements)
# ------------------------------

# Initialize OpenSees model
ops.wipe()
ops.model('basic', '-ndm', 3, '-ndf', 6)  # 3D model with 6 DOF (3 translations, 3 rotations)

# Define material properties
E = 12000  # Elastic modulus (MPa)
A = 0.04   # Cross-sectional area (m^2)

# Bridge parameters
span_length = 10.0  # Length of each span (m)
num_spans = 4       # Number of spans
H_pier = 8.0        # Pier height (m)

# Define nodal coordinates (x, y, z)
# Bottom row of nodes (supports and deck)
for i in range(num_spans + 1):
    x = i * span_length
    ops.node(i + 1, x, 0.0, 0.0)  # Deck nodes
    ops.node(i + num_spans + 2, x, 0.0, -H_pier)  # Pier base nodes

# Fix pier base nodes (pinned)
for i in range(num_spans + 2, 2 * (num_spans + 1) + 1):
    ops.fix(i, 1, 1, 1, 0, 0, 0)  # Pin support at pier base nodes

# Define truss elements (deck and piers)
ops.uniaxialMaterial("Elastic", 1, E)

# Bridge deck trusses
for i in range(num_spans):
    ops.element("truss", i + 1, i + 1, i + 2, A, 1)  # Horizontal deck trusses

# Trusses connecting deck to piers
for i in range(num_spans + 1):
    ops.element("truss", num_spans + i + 1, i + 1, i + num_spans + 2, A, 1)  # Vertical pier trusses

# ------------------------------
# Loading (Define External Loads)
# ------------------------------

# Apply vertical load at the center of the bridge
ops.timeSeries('Linear', 1)
ops.pattern('Plain', 1, 1)
center_node = (num_spans // 2) + 1
ops.load(center_node, 0.0, 0.0, -10.0)  # Apply vertical load (10kN)

# ------------------------------
# Observation and Recording (Output Displacements)
# ------------------------------

# You can add more observation or record commands here if needed
# (e.g., element forces, reactions, etc.)
disp_node_center = ops.nodeDisp(center_node)  # Displacement at the center node
print(f"Displacement at center node: {disp_node_center}")

# ------------------------------
# Analysis (Static Analysis)
# ------------------------------

# Define analysis options
ops.system('BandGeneral')
ops.numberer('RCM')
ops.constraints('Plain')
ops.integrator('LoadControl', 1.0)
ops.algorithm('Newton')
ops.analysis('Static')

# Perform the analysis
ops.analyze(1)

# ------------------------------
# Visualization (Plot the Original and Deformed Shapes)
# ------------------------------

# Original coordinates of the nodes (X, Y, Z)
x_coords = [ops.nodeCoord(i)[0] for i in range(1, num_spans + 2)]
y_coords = [ops.nodeCoord(i)[1] for i in range(1, num_spans + 2)]
z_coords = [ops.nodeCoord(i)[2] for i in range(1, num_spans + 2)]

# Deformed coordinates of the nodes (after applying load)
x_disp = [ops.nodeCoord(i)[0] + ops.nodeDisp(i)[0] for i in range(1, num_spans + 2)]
y_disp = [ops.nodeCoord(i)[1] + ops.nodeDisp(i)[1] for i in range(1, num_spans + 2)]
z_disp = [ops.nodeCoord(i)[2] + ops.nodeDisp(i)[2] for i in range(1, num_spans + 2)]

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot original truss
ax.plot(x_coords, y_coords, z_coords, 'b-o', label='Original shape')

# Plot deformed truss
ax.plot(x_disp, y_disp, z_disp, 'r-o', label='Deformed shape')

# Labels and title
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('3D Truss Bridge Visualization with Multiple Spans')
ax.legend()

# Show plot
plt.show()
