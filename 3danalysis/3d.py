import open3d as o3d
import numpy as np
from scipy.stats import entropy
import sys


def calculate_spatial_color_entropy(mesh_file, texture_file, voxel_size):
    # Load the mesh
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    print(mesh)
    print('Vertices:')
    print(np.asarray(mesh.vertices))
    print('Triangles:')
    print(np.asarray(mesh.triangles))
    #o3d.visualization.draw_geometries([mesh])
    
    # Load the texture
    texture = o3d.io.read_image(texture_file)

    # Compute the 3D voxel grid
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=voxel_size)

    # Extract the voxel colors
    voxel_colors = []
    for voxel in voxel_grid.get_voxels():
        voxel_coord = voxel.grid_index
        voxel_center = voxel_grid.get_voxel_center(voxel_coord)
        voxel_color = texture[int(voxel_center[1]), int(voxel_center[0])]
        voxel_colors.append(voxel_color)

    # Compute the color histogram
    color_hist, _ = np.histogramdd(voxel_grid.get_locations(), bins=voxel_grid.get_resolution(), weights=voxel_colors)

    # Normalize the color histogram
    color_hist /= np.sum(color_hist)

    # Compute the entropy of the color histogram
    color_entropy = entropy(color_hist.flatten())

    return color_entropy

def visualize_voxels(mesh_file, texture_file, voxel_size):
    # Load the mesh
    mesh = o3d.io.read_triangle_mesh(mesh_file)

    # Load the texture
    texture = o3d.io.read_image(texture_file)

    # Compute the 3D voxel grid
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=voxel_size)

    # Visualize the voxel grid
    o3d.visualization.draw_geometries([voxel_grid])

if __name__ == '__main__':
    mesh_file = 'fletcheri_JJ89.obj'
    texture_file = 'fletcheri_JJ89.jpg'
    voxel_size = 0.05
    
    color_entropy = calculate_spatial_color_entropy(mesh_file, texture_file, voxel_size)
    print(f'Color entropy: {color_entropy}')

    #visualize_voxels(mesh_file, texture_file, voxel_size)
