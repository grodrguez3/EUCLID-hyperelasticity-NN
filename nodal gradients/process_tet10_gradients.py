import numpy as np
import pandas as pd

# --- Tet10 shape function derivative functions (from FEBio) ---

def shape_function_derivatives_tet10(r, s, t):
    """
    Computes the shape function derivatives for a 10-node tetrahedral element
    using the FEBio source code formulas.
    
    Parameters:
      r, s, t : Isoparametric coordinates (tetrahedral reference system)
    
    Returns:
      dN: A (10,3) NumPy array where each row i is [∂N_i/∂r, ∂N_i/∂s, ∂N_i/∂t]
    """
    dN = np.zeros((10, 3))
    # Derivatives with respect to r (Hr)
    dN[0, 0] = -3.0 + 4.0*r + 4.0*(s + t)
    dN[1, 0] = 4.0*r - 1.0
    dN[2, 0] = 0.0
    dN[3, 0] = 0.0
    dN[4, 0] = 4.0 - 8.0*r - 4.0*(s + t)
    dN[5, 0] = 4.0*s
    dN[6, 0] = -4.0*s
    dN[7, 0] = -4.0*t
    dN[8, 0] = 4.0*t
    dN[9, 0] = 0.0

    # Derivatives with respect to s (Hs)
    dN[0, 1] = -3.0 + 4.0*s + 4.0*(r + t)
    dN[1, 1] = 0.0
    dN[2, 1] = 4.0*s - 1.0
    dN[3, 1] = 0.0
    dN[4, 1] = -4.0*r
    dN[5, 1] = 4.0*r
    dN[6, 1] = 4.0 - 8.0*s - 4.0*(r + t)
    dN[7, 1] = -4.0*t
    dN[8, 1] = 0.0
    dN[9, 1] = 4.0*t

    # Derivatives with respect to t (Ht)
    dN[0, 2] = -3.0 + 4.0*t + 4.0*(r + s)
    dN[1, 2] = 0.0
    dN[2, 2] = 0.0
    dN[3, 2] = 4.0*t - 1.0
    dN[4, 2] = -4.0*r
    dN[5, 2] = 0.0
    dN[6, 2] = -4.0*s
    dN[7, 2] = 4.0 - 8.0*t - 4.0*(r + s)
    dN[8, 2] = 4.0*r
    dN[9, 2] = 4.0*s

    return dN

def compute_jacobian_tet10(nodal_coords, r, s, t):
    """
    Computes the Jacobian matrix for a tet10 element at a given integration point.
    
    Parameters:
      nodal_coords: (10,3) NumPy array of nodal coordinates for the tet10 element.
      r, s, t: Isoparametric coordinates in the tetrahedral reference system.
    
    Returns:
      J: (3,3) Jacobian matrix.
      detJ: Determinant of J.
      invJ: Inverse of J (or None if detJ is zero).
    """
    dN_dxi = shape_function_derivatives_tet10(r, s, t) #(10,3)
    J = np.zeros((3, 3))
    for a in range(10):
        J[:, 0] += dN_dxi[a, 0] * nodal_coords[a, :]
        J[:, 1] += dN_dxi[a, 1] * nodal_coords[a, :]
        J[:, 2] += dN_dxi[a, 2] * nodal_coords[a, :]
    detJ = np.linalg.det(J)
    invJ = np.linalg.inv(J) if detJ != 0 else None
    return J, detJ, invJ

# --- End tet10 functions ---

# --- Read parsed mesh CSV and compute global gradients ---

def process_tet10_mesh_csv(input_csv, output_csv, integration_point=(0.25, 0.25, 0.25)):
    """
    Reads a CSV file with tet10 mesh data (nodal coordinates) and computes the global
    shape function gradients at a specified integration point.
    
    The input CSV is expected to have columns:
      Element_ID, node1_x, node1_y, node1_z, node2_x, node2_y, node2_z, ..., node10_x, node10_y, node10_z
     
    The output CSV will include additional columns for:
      gradNa_node1_x, gradNa_node1_y, gradNa_node1_z, ..., gradNa_node10_x, gradNa_node10_y, gradNa_node10_z
    
    Parameters:
      input_csv (str): Path to the input CSV file.
      output_csv (str): Path to the output CSV file.
      integration_point (tuple): (r, s, t) coordinates in the tetrahedral reference system.
    """
    df = pd.read_csv(input_csv)
    r, s, t = integration_point
    
    # Prepare lists to hold gradient data for each element
    gradient_data = []

    # Process each element (each row)
    for index, row in df.iterrows():
        # Extract nodal coordinates for the 10 nodes into a (10,3) array.
        # The columns are assumed to be in the order: node1_x, node1_y, node1_z, node2_x, ...
        nodes = []
        for i in range(1, 11):
            x = row[f"node{i}_x"]
            y = row[f"node{i}_y"]
            z = row[f"node{i}_z"]
            nodes.append([x, y, z])
        nodal_coords = np.array(nodes)  # shape (10,3)
        
        # Compute the Jacobian and its inverse at the given integration point
        J, detJ, invJ = compute_jacobian_tet10(nodal_coords, r, s, t)
        # Compute local shape function derivatives
        dN_local = shape_function_derivatives_tet10(r, s, t)
        # Transform to global gradients: ∇_x N = J^{-T} * (∇_ξ N)
        dN_global = (invJ.T @ dN_local.T).T if invJ is not None else np.full((10,3), np.nan)
        
        # Flatten the gradients into a list [grad_node1_x, grad_node1_y, grad_node1_z, ...]
        grad_list = dN_global.flatten().tolist()
        gradient_data.append(grad_list)
    
    # Create new DataFrame with gradient columns
    grad_columns = []
    for i in range(1, 11):
        grad_columns.extend([f"gradNa_node{i}_x", f"gradNa_node{i}_y", f"gradNa_node{i}_z"])
    grad_df = pd.DataFrame(gradient_data, columns=grad_columns)
    
    # Concatenate the original DataFrame with the new gradient DataFrame
    df_out = pd.concat([df, grad_df], axis=1)
    df_out.to_csv(output_csv, index=False)
    print(f"Processed mesh with gradients exported to: {output_csv}")

if __name__ == "__main__":
    # Replace with your actual CSV file that was produced by your FEBio mesh parser.
    input_csv = "mesh_elements.csv"
    output_csv = "tet10_mesh_with_gradients.csv"
    
    process_tet10_mesh_csv(input_csv, output_csv)
