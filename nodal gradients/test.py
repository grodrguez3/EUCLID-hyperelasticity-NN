import numpy as np
import pandas as pd

# Define Gauss quadrature points for Hex8 (2x2x2 rule)
gauss_points = np.array([
    [-1/np.sqrt(3), -1/np.sqrt(3), -1/np.sqrt(3)],
    [ 1/np.sqrt(3), -1/np.sqrt(3), -1/np.sqrt(3)],
    [ 1/np.sqrt(3),  1/np.sqrt(3), -1/np.sqrt(3)],
    [-1/np.sqrt(3),  1/np.sqrt(3), -1/np.sqrt(3)],
    [-1/np.sqrt(3), -1/np.sqrt(3),  1/np.sqrt(3)],
    [ 1/np.sqrt(3), -1/np.sqrt(3),  1/np.sqrt(3)],
    [ 1/np.sqrt(3),  1/np.sqrt(3),  1/np.sqrt(3)],
    [-1/np.sqrt(3),  1/np.sqrt(3),  1/np.sqrt(3)]
])

def shape_function_derivatives_hex8(r, s, t):
    """
    Computes the shape function derivatives for an 8-node hexahedral element.

    Parameters:
    - r, s, t: Isoparametric coordinates (Gauss quadrature points)

    Returns:
    - dN: (8,3) matrix with derivatives [∂N/∂r, ∂N/∂s, ∂N/∂t] for all 8 nodes  of a hex8 element.
    """
    dN = np.zeros((8, 3))

    dN[:, 0] = [-0.125*(1 - s)*(1 - t),  0.125*(1 - s)*(1 - t),
                 0.125*(1 + s)*(1 - t), -0.125*(1 + s)*(1 - t),
                -0.125*(1 - s)*(1 + t),  0.125*(1 - s)*(1 + t),
                 0.125*(1 + s)*(1 + t), -0.125*(1 + s)*(1 + t)]

    dN[:, 1] = [-0.125*(1 - r)*(1 - t), -0.125*(1 + r)*(1 - t),
                 0.125*(1 + r)*(1 - t),  0.125*(1 - r)*(1 - t),
                -0.125*(1 - r)*(1 + t), -0.125*(1 + r)*(1 + t),
                 0.125*(1 + r)*(1 + t),  0.125*(1 - r)*(1 + t)]

    dN[:, 2] = [-0.125*(1 - r)*(1 - s), -0.125*(1 + r)*(1 - s),
                -0.125*(1 + r)*(1 + s), -0.125*(1 - r)*(1 + s),
                 0.125*(1 - r)*(1 - s),  0.125*(1 + r)*(1 - s),
                 0.125*(1 + r)*(1 + s),  0.125*(1 - r)*(1 + s)]

    return dN

def compute_jacobian_hex8(nodal_coords, r, s, t):
    """
    Computes the Jacobian matrix J for a hexahedral element.

    Parameters:
    - nodal_coords: (8, 3) array of nodal coordinates (x, y, z) for an 8-node hex element.
 
    Returns:
    - J: (3, 3) Jacobian matrix
    - detJ: Determinant of J
    - invJ: Inverse of the Jacobian
    """
    dN_dxi = shape_function_derivatives_hex8(r, s, t)

    J = np.zeros((3, 3))
    for a in range(8):
        J[:, 0] += dN_dxi[a, 0] * nodal_coords[a, :]
        J[:, 1] += dN_dxi[a, 1] * nodal_coords[a, :]
        J[:, 2] += dN_dxi[a, 2] * nodal_coords[a, :]

    detJ = np.linalg.det(J)
    invJ = np.linalg.inv(J) if detJ != 0 else None

    return J, detJ, invJ

# Define nodal coordinates for multiple elements
mesh_elements = [
    np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Bottom four nodes
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # Top four nodes
    ]),
    np.array([
        [1, 0, 0], [2, 0, 0], [2, 1, 0], [1, 1, 0],  # Bottom four nodes
        [1, 0, 1], [2, 0, 1], [2, 1, 1], [1, 1, 1]   # Top four nodes
    ])
]

# Prepare CSV output data
data = []

for elem_idx, nodal_coords in enumerate(mesh_elements):
    J, detJ, invJ = compute_jacobian_hex8(nodal_coords, 0, 0, 0)  # Compute at element center

    dN_local = shape_function_derivatives_hex8(0, 0, 0)
    dN_global = (invJ.T @ dN_local.T).T if invJ is not None else None

    row = []
    for node in nodal_coords:  # Store all 8 nodes' (x, y, z)
        row.extend(node[:3])  # Store x, y, z

    if dN_global is not None:
        for grad in dN_global:  # Store all 8 nodes' gradients
            row.extend(grad[:3])  # Store grad x, grad y, grad z

    data.append(row)

# Convert to DataFrame
columns = []
for i in range(1, 9):  # 8 nodes
    columns.extend([f"node{i}_x", f"node{i}_y", f"node{i}_z"])
for i in range(1, 9):
    columns.extend([f"gradNa_node{i}_x", f"gradNa_node{i}_y", f"gradNa_node{i}_z"])

df = pd.DataFrame(data, columns=columns)

# Save to CSV
df.to_csv("./hexahedral_element_data.csv", index=False)

# Display the DataFrame
#import acetools as tools
#tools.display_dataframe_to_user(name="Hexahedral Element Data", dataframe=df)
