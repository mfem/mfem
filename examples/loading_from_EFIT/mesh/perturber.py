import numpy as np

# Path to the mesh file
mesh_file_path = "./new_2d_mesh_unalign.mesh"

# Read the mesh file as text
with open(mesh_file_path, 'r') as file:
    lines = file.readlines()

# Function to find a section in the mesh file
def find_section(lines, section_name):
    """Find the starting line of a section in the mesh file"""
    for idx, line in enumerate(lines):
        if line.strip() == section_name:
            return idx
    return -1

# Parameters for perturbation
alpha = 0.05  # Perturbation strength for x coordinates
beta = 0.05   # Perturbation strength for y coordinates

# Find and extract vertices
vertices = None
vertices_idx = find_section(lines, "vertices")
if vertices_idx != -1:
    num_vertices = int(lines[vertices_idx + 1].strip())
    vertex_dim = int(lines[vertices_idx + 2].strip())
    
    # Read vertex coordinates into a numpy array
    vertices = np.zeros((num_vertices, vertex_dim))
    for i in range(num_vertices):
        if vertices_idx + 3 + i < len(lines):
            coords = lines[vertices_idx + 3 + i].strip().split()
            if len(coords) == vertex_dim:
                vertices[i] = [float(coord) for coord in coords]
    
    # Verify the shape
    assert vertices.shape == (num_vertices, vertex_dim)
    
    # Apply perturbation: x = x + alpha * sin(x), y = y + beta * sin(y)
    if vertex_dim >= 2:  # Make sure we have at least x and y coordinates
        vertices[:, 0] = vertices[:, 0] + alpha * np.sin(vertices[:, 0])  # Perturb x
        vertices[:, 1] = vertices[:, 1] + beta * np.sin(vertices[:, 1])   # Perturb y
    
    # Optional: Write the perturbed vertices back to a new mesh file
    output_mesh_path = "./new_2d_mesh_unalign_p.mesh"
    with open(output_mesh_path, 'w') as outfile:
        # Copy the file up to the vertices section
        for j in range(vertices_idx + 3):
            outfile.write(lines[j])
        
        # Write the perturbed vertices
        for i in range(num_vertices):
            if vertex_dim == 2:
                outfile.write(f"{vertices[i, 0]} {vertices[i, 1]}\n")
            elif vertex_dim == 3:
                outfile.write(f"{vertices[i, 0]} {vertices[i, 1]} {vertices[i, 2]}\n")
        
        # Copy the rest of the file
        for j in range(vertices_idx + 3 + num_vertices, len(lines)):
            outfile.write(lines[j])
    
    print(f"Perturbed mesh saved to {output_mesh_path}")

# Now 'vertices' contains all perturbed vertex coordinates as a numpy array or None if not found
# You can use this array for further processing