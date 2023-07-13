import os
import numpy as np
from vtk.util import numpy_support
from vtk.util.numpy_support import vtk_to_numpy
import vtk
import matplotlib.pyplot as plt


import vtk
import numpy as np
from scipy.ndimage import gaussian_filter
import cv2

import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter

def calculate_spatial_color_entropy(obj_file, mtl_file, nbins, sigma=0.5):
    # Load the obj and mtl files and extract the vertex and face data
    vertices = []
    faces = []
    with open(obj_file, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertex = list(map(float, line.split()[1:]))
                vertices.append(vertex)
            elif line.startswith('f '):
                face = line.split()[1:]
                face = [int(x.split('/')[0]) for x in face]
                faces.append(face)

    # Load the texture image and convert to HSV
    # Load the MTL file and find the texture image file name
    with open(mtl_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('map_Kd'):
            texture_file = line.split()[1]
            break
    # Load the texture image file
    texture_path = os.path.join(os.getcwd(), texture_file)
    texture_img = cv2.imread(texture_path)
    hsv_texture_img = cv2.cvtColor(texture_img, cv2.COLOR_BGR2HSV)
    # Create an empty 3D array to represent the 3D texture
    max_x, max_y, max_z = np.max(vertices, axis=0)[:3].astype(int)
    min_x, min_y, min_z = np.min(vertices, axis=0)[:3].astype(int)

    grid_size = np.array([max_x - min_x, max_y - min_y, max_z - min_z]) + 1

    # Shift all values by a constant factor such that the minimum value becomes zero
    vertices = np.array(vertices) # must be a numpy
    shift_x = int(np.abs(np.min(vertices[:, 0])))
    shift_y = int(np.abs(np.min(vertices[:, 1])))
    shift_z = int(np.abs(np.min(vertices[:, 2])))
    grid_size = np.ceil(np.max(vertices, axis=0) - np.min(vertices, axis=0)).astype(int)
    grid_size[grid_size == 0] = 1  # Handle zero dimension
    grid_size += np.array([shift_x, shift_y, shift_z])
    texture_3d = np.zeros((grid_size[0], grid_size[1], grid_size[2], 3))

    # Assign color values to each voxel in the 3D texture
    for face in faces:
        for i in range(3):
            x, y, z = vertices[face[i]-1]
            color = hsv_texture_img[int(y), int(x)]
            texture_3d[int(x)+shift_x, int(y)+shift_y, int(z)+shift_z, :] = color

            #texture_3d[int(x), int(y), int(z), :] = color
    # Smooth the 3D texture using a Gaussian filter
    texture_3d_smoothed = gaussian_filter(texture_3d, sigma=sigma)

    # Compute the histogram of the smoothed texture
    hist_h, _ = np.histogram(texture_3d_smoothed[..., 0].ravel(), bins=nbins, range=(0, 180), density=True)
    hist_s, _ = np.histogram(texture_3d_smoothed[..., 1].ravel(), bins=nbins, range=(0, 256), density=True)
    hist_v, _ = np.histogram(texture_3d_smoothed[..., 2].ravel(), bins=nbins, range=(0, 256), density=True)
    hist = np.concatenate((hist_h, hist_s, hist_v))

    # Compute the entropy of the histogram
    entropy = -np.sum(hist*np.log2(hist+1e-10))

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
    #plt.show()

def main():
     # Calculate entropy for a given set of parameters
    nbins = 256
    prefix='fletcheri_JJ89'
   
    #entropy = calculate_spatial_color_entropy(prefix+'.obj', prefix+'.mtl',nbins)
    #print(f"Spatial color entropy for {prefix} with {nbins} bins: {entropy}")
    
    #entropy = calculate_color_entropy(prefix+'.obj', prefix+'.mtl', nbins)
    #print(f"Non-spatial color entropy for {prefix} with {nbins} bins: {entropy}")

    prefix='prezewalskii_JJ14'

    entropy = calculate_spatial_color_entropy(prefix+'.obj', prefix+'.mtl',nbins)
    print(f"Spatial color entropy for {prefix} with {nbins} bins: {entropy}")
    
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