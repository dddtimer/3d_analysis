import vtk
import os
import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


#Error: qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in...
#set: export QT_QPA_PLATFORM=offscreen 
def calculate_spatial_color_entropy(obj_file, mtl_file, nbins, voxel_size=0.25, sigma=0.5):
    # Load the obj and mtl files and extract the vertex and face data
    vertices = []
    faces = []
    with open(obj_file, 'r') as f:
        for line in f:
            if line.startswith('v '):
                x, y, z = map(float, line.split()[1:4]) #map applies float to all elements returned by split
                vertex = [x, y, z] #new list object containing x,y,z
                vertices.append(vertex) #list of vertices
            elif line.startswith('f '):
                #the f line in the obj file lists the corresponding line number (index) for all three vertices
                face = line.split()[1:]
                face = [int(x.split('/')[0]) for x in face] #extracts the vertex indices only e.g., [1/223, 2/566, 3/5666] becomes [1,2,3]
                faces.append(face) #adds face to the list of faces
                   
    #Shift the coordinates of all vertices so there are no negative values
    #to ensure the correct grid size is generated
    min_val = np.min(vertices)
    vertices -= min_val    
    # Load the texture image and convert to HSV
    # Load the MTL file and find the texture image file name
    with open(mtl_file, 'r') as f:
        lines = f.readlines()
    for line in lines: #searches the mtl file for the line containing the reference to the texture file
        if line.startswith('map_Kd'):
            texture_file = line.split()[1] #stores the file name of the texture file
            break
    # Load the texture image file
    texture_path = os.path.join(os.getcwd(), texture_file)
    texture_img = cv2.imread(texture_path)
    hsv_texture_img = cv2.cvtColor(texture_img, cv2.COLOR_BGR2HSV)#changes color space of texture file from rgb to hsv

    # Set the size of the 3D grid by the difference in the maximum and minimum coordinate values along each axis
    max_xyz = np.max(vertices, axis=0)
    max_xyz = max_xyz+0.1*max_xyz
    min_xyz = np.min(vertices, axis=0)
    min_xyz = min_xyz    
    grid_size = np.ceil((max_xyz)/voxel_size).astype(int)
    print(grid_size)
    # Create an empty 3D texture of the calculated size
    texture_3d = np.zeros((grid_size[0], grid_size[1], grid_size[2], 3))
    voxel_weight = np.zeros((grid_size[0], grid_size[1], grid_size[2], 1))

    #######
    min_coords = [float('inf'), float('inf'), float('inf')]
    max_coords = [float('-inf'), float('-inf'), float('-inf')]

    for vertex in vertices:
        x, y, z = vertex
        min_coords[0] = min(min_coords[0], x)
        min_coords[1] = min(min_coords[1], y)
        min_coords[2] = min(min_coords[2], z)
        max_coords[0] = max(max_coords[0], x)
        max_coords[1] = max(max_coords[1], y)
        max_coords[2] = max(max_coords[2], z)

    print("grid_size:", grid_size)
    print("max_xyz:", max_xyz)
    print("Minimum vertex coordinates:", min_coords)
    print("Maximum vertex coordinates:", max_coords)
    if any(max_coords[i] >= max_xyz[i] for i in range(3)):
        print("Some vertices are falling outside of the grid")
    #######

    # Assign color values to each voxel in the 3D texture
    for face in faces:
        for i in range(3):
            # Get the vertex coordinates and color value
            #print(np.floor(np.array([x, y, z]) / voxel_size).astype(int))
            x, y, z = vertices[face[i]-1]
            color = hsv_texture_img[int(y), int(x)]

            # Round the vertex coordinates to the nearest integer to determine the voxel position
            x_idx, y_idx, z_idx = np.floor(np.array([x, y, z]) / voxel_size).astype(int)
            print(f"x_idx {x_idx} y_idx {y_idx} z_idx {z_idx}")
            # Calculate weights for each vertex based on Euclidean distance to the center of the voxel
            dist_x, dist_y, dist_z = x - (x_idx + 0.5) * voxel_size, y - (y_idx + 0.5) * voxel_size, z - (z_idx + 0.5) * voxel_size
            dist = np.sqrt(dist_x**2 + dist_y**2 + dist_z**2)
            w = 1 - dist / voxel_size

            # Update the color for this voxel based on the weighted contribution of the current vertex
            texture_3d[x_idx, y_idx, z_idx, :] += color * w
            # Update the weight of the voxel at this position
            # A voxel may have multiple vertices contributing to its color, so we need to track the total weight
            voxel_weight[x_idx, y_idx, z_idx, :] += w

    # Smooth the 3D texture using a Gaussian filter
    # This helps to reduce noise and ensure that colors blend smoothly across adjacent voxels
    texture_3d_smoothed = gaussian_filter(texture_3d, sigma=sigma)
    
    # print(voxel_weight)

    # Normalize each voxel by its weight
    # By dividing each voxel by its total weight, we ensure that the final colors are averaged correctly across all contributing vertices
    texture_3d_normalized = np.divide(texture_3d_smoothed, voxel_weight+1e-10)

    # Compute the histogram of the normalized texture
    hist_h, _ = np.histogram(texture_3d_normalized[..., 0].ravel(), bins=nbins, range=(0, 180), density=True)
    hist_s, _ = np.histogram(texture_3d_normalized[..., 1].ravel(), bins=nbins, range=(0, 256), density=True)
    hist_v, _ = np.histogram(texture_3d_normalized[..., 2].ravel(), bins=nbins, range=(0, 256), density=True)
    hist = np.concatenate((hist_h, hist_s, hist_v))
    # Compute the entropy of the histogram
    #entropy = -np.sum(hist*np.log2(hist+1e-10))
    entropy = -np.sum(hist * np.log(hist + 1e-10)) #/ np.log(nbins)    
    # Count the number of non-zero weights in each voxel
    num_vertices_per_voxel = np.count_nonzero(voxel_weight, axis=-1)

    # Flatten the num_vertices_per_voxel array to a 1D array
    num_vertices_per_voxel = num_vertices_per_voxel.ravel()

    # Count the number of voxels with non-zero weights
    num_voxels_with_vertices = np.count_nonzero(num_vertices_per_voxel)

    # Print the number of voxels with non-zero weights and the number of vertices per voxel
    print(f"Number of voxels with non-zero weights: {num_voxels_with_vertices}")
    print(f"Number of vertices per voxel: {num_vertices_per_voxel}")
    return entropy

def calculate_color_entropy(obj_file_name: str, mtl_file_name: str, nbins: int) -> float:
    # Load OBJ file with texture coordinates
    obj_importer = vtk.vtkOBJImporter()
    obj_importer.SetFileName(obj_file_name)
    obj_importer.SetFileNameMTL(mtl_file_name)
    obj_importer.Read()

    # Get output from importer
    actor = obj_importer.GetRenderer().GetActors().GetLastActor()
    polydata = actor.GetMapper().GetInput()

    # Get color data from MTL file
    with open(mtl_file_name, 'r') as f:
        lines = f.readlines()

    # Find the material name and associated texture file
    texture_file = None
    material_name = None
    for line in lines:
        if line.startswith('newmtl'):
            material_name = line.split()[1]
        elif material_name and line.startswith('map_Kd'):
            texture_file = line.split()[1]

    # If a texture file is found, use it to calculate color histogram
    if texture_file:
        # Load texture image
        reader = vtk.vtkJPEGReader()
        reader.SetFileName(texture_file)
        reader.Update()

        # Calculate color histogram
        color_data = vtk.util.numpy_support.vtk_to_numpy(reader.GetOutput().GetPointData().GetScalars())
        hist, _ = np.histogram(color_data, bins=nbins, range=(0, 255))
        hist = hist / np.sum(hist)
        entropy = -np.sum(hist * np.log2(hist + (hist == 0)))

    # If no texture file is found, return 0.0
    else:
        entropy = 0.0

    return entropy


def calculate_texture_entropy(obj_file_name: str, mtl_file_name: str, nbins: int) -> float:
    # Load OBJ file with texture coordinates
    obj_importer = vtk.vtkOBJImporter()
    obj_importer.SetFileName(obj_file_name)
    obj_importer.SetFileNameMTL(mtl_file_name)
    obj_importer.Read()

    # Get output from importer
    actor = obj_importer.GetRenderer().GetActors().GetLastActor()
    polydata = actor.GetMapper().GetInput()

    # Get texture data from MTL file
    with open(mtl_file_name, 'r') as f:
        mtl_data = f.readlines()

    texture_file_name = None
    for line in mtl_data:
        if 'map_Kd' in line:
            texture_file_name = line.split()[-1]

    if texture_file_name is None:
        raise ValueError("No texture data found in MTL file.")

    # Load texture file
    reader = vtk.vtkJPEGReader()
    reader.SetFileName(texture_file_name)
    reader.Update()

    # Create texture histogram
    texture_data = reader.GetOutput().GetPointData().GetScalars()
    texture_array = vtk.util.numpy_support.vtk_to_numpy(texture_data)
    texture_hist, _ = np.histogram(texture_array, bins=nbins, density=True)
    texture_hist = texture_hist[texture_hist > 0]

    # Calculate texture entropy
    texture_entropy = -np.sum(texture_hist * np.log2(texture_hist))

    return texture_entropy


def plot_entropy_vs_nbins(obj_file_name: str, mtl_file_name: str, nbins_range: range):
    """
    Plots the texture entropy of a 3D image given the file names of the obj and mtl files, and a range of values for the
    number of bins in the histogram.

    Parameters:
    obj_file_name (str): The file name of the .obj file containing the 3D image and its texture coordinates.
    mtl_file_name (str): The file name of the .mtl file containing the materials and textures of the 3D image.
    nbins_range (range): A range of values for the number of bins to use in the histogram calculation.
    """
    prefix=obj_file_name.split(".")[0]
    entropy_values = []
    for nbins in nbins_range:
        entropy = calculate_texture_entropy(obj_file_name, mtl_file_name, nbins)
        entropy_values.append(entropy)
    plt.plot(list(nbins_range), entropy_values)
    plt.xlabel('Number of Bins')
    plt.ylabel('Texture Entropy')
    plt.title('Texture Entropy vs Number of Bins for '+ prefix)
    # Save the plot as an image file
    filename=prefix+'_entropy_vs_nbins.png'
    plt.savefig(filename)
    print('Image saved to: '+os.getcwd()+"/"+filename)
    # show the plot
    plt.show()

def main():
     # Calculate entropy for a given set of parameters
    nbins = 256
    prefix='fletcheri_JJ89'
   
    entropy = calculate_spatial_color_entropy(prefix+'.obj', prefix+'.mtl',nbins)
    print(f"Spatial color entropy for {prefix} with {nbins} bins: {entropy}")
    
    #entropy = calculate_color_entropy(prefix+'.obj', prefix+'.mtl', nbins)
    #print(f"Non-spatial color entropy for {prefix} with {nbins} bins: {entropy}")

    prefix='prezewalskii_JJ14'

    #entropy = calculate_spatial_color_entropy(prefix+'.obj', prefix+'.mtl',nbins)
    #print(f"Spatial color entropy for {prefix} with {nbins} bins: {entropy}")
    
    #entropy = calculate_color_entropy(prefix+'.obj', prefix+'.mtl', nbins)
    #print(f"Non-spatial color entropy for {prefix} with {nbins} bins: {entropy}")

    """  entropy = calculate_texture_entropy(prefix+'.obj', prefix+'.mtl', nbins)
    print(f"Texture entropy for {prefix} with {nbins} bins: {entropy}")   

    entropy = calculate_color_entropy(prefix+'.obj', prefix+'.mtl', nbins)
    print(f"Color entropy for {prefix} with {nbins} bins: {entropy}")

    prefix='prezewalskii_JJ14'

    entropy = calculate_texture_entropy(prefix+'.obj', prefix+'.mtl', nbins)
    print(f"Texture entropy for {prefix} with {nbins} bins: {entropy}")   

    entropy = calculate_color_entropy(prefix+'.obj', prefix+'.mtl', nbins)
    print(f"Color entropy for {prefix} with {nbins} bins: {entropy}") """

    #entropy = calculate_texture_entropy(prefix+'.obj', prefix+'.mtl', nbins)
    #print(f"Texture entropy for {prefix} with {nbins} bins: {entropy}")
    
    #prefix='prezewalskii_JJ14'
    #entropy = calculate_texture_entropy(prefix+'.obj', prefix+'.mtl', nbins)
    #print(f"Texture entropy for {prefix} with {nbins} bins: {entropy}")
    
    
    
    
    
    # plot entropy by bin number
    # Define the number of bins
    #nbins_range = range(10, 500, 25)
    # Call the function and plot the results
    #plot_entropy_vs_nbins(prefix+'.obj', prefix+'.mtl, nbins_range)

if __name__ == '__main__':
    main()