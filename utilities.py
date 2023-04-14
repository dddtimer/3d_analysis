from PIL import Image

def get_vertex_colors(mesh, texture, mesh_file_prefix):
    """
    Extracts the vertex colors from the given mesh and texture.

    The function calculates the color for each vertex of the mesh by using the
    UV coordinates stored in the mesh's point data. It then samples the colors
    from the texture image at the corresponding (x, y) positions.

    Args:
        mesh (vedo.Mesh): A mesh object containing the 3D model. The mesh should have UV coordinates
                          in the `mesh.pointdata` dictionary.
        texture (PIL.Image.Image): A texture image that maps onto the mesh. The image should have
                                   dimensions corresponding to the mesh UV coordinates.
        mesh_file_prefix (str): The prefix of the mesh filename used to identify the UV coordinates
                                in `mesh.pointdata`.

    Returns:
        list: A list of vertex colors as (R, G, B) tuples.
    """
    uv_coords = mesh.pointdata[mesh_file_prefix]
    if uv_coords is None:
        uv_coords=mesh.pointdata['TCoords']
    width, height = texture.size  # Get the dimensions of the texture image
    vertex_colors = []

    # Loop through each UV coordinate to calculate the corresponding vertex color
    for uv in uv_coords:
        # Convert the UV coordinates to (x, y) pixel coordinates in the texture image
        x = int(uv[0] * (width - 1))
        y = int((1 - uv[1]) * (height - 1))
        # Sample the color from the texture image at the (x, y) position
        color = texture.getpixel((x, y))
        # Append the color to the vertex_colors list
        vertex_colors.append(color)

    return vertex_colors