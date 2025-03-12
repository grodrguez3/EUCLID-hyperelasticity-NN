import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

def parse_febio_mesh(feb_file):
    """
    Parses a FEBio .feb file to extract nodal coordinates and tet10 element connectivity.
    
    Parameters:
      feb_file (str): Path to the FEBio .feb file.
    
    Returns:
      mesh_elements (list): List of NumPy arrays (each of shape (10,3)), one per element.
      element_ids (list): List of element IDs (as strings) corresponding to each element.
    """
    # Parse the FEBio XML file
    tree = ET.parse(feb_file)
    root = tree.getroot()
    
    # Extract nodes: create a dictionary mapping node id to its coordinate (as a NumPy array)
    nodes = {}
    nodes_element = root.find(".//Mesh/Nodes")
    if nodes_element is None:
        raise ValueError("No <Nodes> section found in the FEBio file.")
    
    for node in nodes_element.findall("node"):
        node_id = node.attrib.get("id")
        coord_text = node.text.strip()
        # Assume coordinates are comma-separated
        coords = list(map(float, coord_text.split(",")))
        nodes[node_id] = np.array(coords)
    
    # Extract elements: here we look for tet10 elements.
    mesh_elements = []
    element_ids = []
    elements_element = root.find(".//Mesh/Elements")
    if elements_element is None:
        raise ValueError("No <Elements> section found in the FEBio file.")
    
    # In your sample, the <Elements> tag has type="tet10".
    elem_type = elements_element.attrib.get('type', '').lower()
    if elem_type != "tet10":
        print(f"Warning: Expected element type 'tet10' but found '{elem_type}'.")
    
    for elem in elements_element.findall("elem"):
        # Get the list of node IDs for this element.
        # Replace commas with spaces and then split.
        node_ids = elem.text.strip().replace(",", " ").split()
        if len(node_ids) != 10:
            print(f"Warning: Element id {elem.attrib.get('id')} has {len(node_ids)} nodes (expected 10). Skipping.")
            continue
        # Build an array (10x3) with the nodal coordinates
        coords_array = np.array([nodes[node_id] for node_id in node_ids])
        mesh_elements.append(coords_array)
        element_ids.append(elem.attrib.get("id"))
    
    return mesh_elements, element_ids

def export_mesh_to_csv(mesh_elements, element_ids, output_csv):
    """
    Exports the mesh elements to a CSV file.
    
    Each row corresponds to one element and has the following format:
      Element_ID, node1_x, node1_y, node1_z, node2_x, node2_y, node2_z, ..., node10_x, node10_y, node10_z
    
    Parameters:
      mesh_elements (list): List of NumPy arrays (each element is an array of shape (10,3)).
      element_ids (list): List of element IDs.
      output_csv (str): Path to the output CSV file.
    """
    rows = []
    for elem_id, coords in zip(element_ids, mesh_elements):
        row = [elem_id]
        # Flatten the (10,3) coordinate array into a 1D list (node1_x, node1_y, node1_z, node2_x, ... )
        row.extend(coords.flatten().tolist())
        rows.append(row)
    
    # Build column headers
    columns = ["Element_ID"]
    for i in range(1, 11):
        columns.extend([f"node{i}_x", f"node{i}_y", f"node{i}_z"])
    
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"Mesh exported to CSV file: {output_csv}")

if __name__ == "__main__":
    # Replace with the path to your FEBio .feb file
    feb_file = "mesh.feb"  
    output_csv = "mesh_elements.csv"
    
    try:
        mesh_elements, element_ids = parse_febio_mesh(feb_file)
        export_mesh_to_csv(mesh_elements, element_ids, output_csv)
    except Exception as e:
        print("Error parsing FEBio mesh:", e)
