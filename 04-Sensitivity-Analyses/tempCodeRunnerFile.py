# - Pier Truss Elements

# for i in range(1, num_span):
#     # Nodes for the current span (bottom-left, bottom-right, top-left, top-right)
#     left_node_high = node_deck_left[i]
#     right_node_high = node_deck_right[i]
#     left_node_low = node_pier_left[i - 1]
#     right_node_low = node_pier_right[i - 1]
    
#     # Z-direction truss elements
#     ops.element("truss", element_id, left_node_high, left_node_low, A, 1) # left vertical pier
#     element_pier.append(element_id)
#     element_id += 1
#     ops.element("truss", element_id, right_node_high, right_node_low, A, 1) # right vertical pier
#     element_pier.append(element_id)
#     element_id += 1
    
#     # Diagonal truss elements
#     ops.element("truss", element_id, left_node_high, right_node_low, A, 1) # diagonal left-top to right-bottom
#     element_pier.append(element_id)
#     element_id += 1
#     ops.element("truss", element_id, right_node_high, left_node_low, A, 1) # diagonal right-top to left-bottom
#     element_pier.append(element_id)
#     element_id += 1

# print("element_pier: ", element_pier)