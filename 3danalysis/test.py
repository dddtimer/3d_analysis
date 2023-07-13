import numpy as np
from vtk.util import numpy_support
from vtk.util.numpy_support import vtk_to_numpy
import vtk

# set the prefix of the input OBJ file (without file extension)
prefix='prezewalskii_JJ14'

# Define the number of bins for the texture histogram
nbins = 256

# Load OBJ file with texture coordinates using VTK OBJ importer
obj_importer = vtk.vtkOBJImporter()
obj_importer.SetFileName(prefix+'.obj')
obj_importer.SetFileNameMTL(prefix+'.mtl')
obj_importer.Read()

# Get the output actor from the VTK OBJ importer
actor = obj_importer.GetRenderer().GetActors().GetLastActor()

# Get the polydata from the actor
polydata = actor.GetMapper().GetInput()

# Extract texture coordinates from the polydata
texture_coords = vtk_to_numpy(polydata.GetPointData().GetTCoords())

# Compute the histogram of texture coordinates using NumPy histogramdd function
# texture_coords is a 2D NumPy array with each row representing a texture coordinate (u, v) in [0, 1] range
texture_hist, bins = np.histogramdd(texture_coords, bins=nbins, range=[[0, 1], [0, 1]])

# Normalize the histogram so that it sums to 1
texture_prob = texture_hist / np.sum(texture_hist)

# Add a small constant to avoid taking the logarithm of 0
texture_prob += 1e-8

# Compute the entropy of the texture distribution using the normalized histogram
texture_entropy = -np.sum(texture_prob * np.log2(texture_prob))

# Print the computed texture entropy
print("Texture entropy:", texture_entropy)

""" numpy: NumPy is a popular Python library for numerical computing with arrays and matrices. We use it to compute the histogram of texture coordinates and perform some basic array operations.
    
vtk: The Visualization Toolkit (VTK) is a popular open-source software system for 3D computer graphics, image processing, and visualization. We use it to load the OBJ file containing the 3D model and its texture coordinates.
    
prefix: The prefix is the file name prefix (without file extension) for the input OBJ file.
nbins: The number of bins in the texture histogram. This determines the granularity of the histogram, with more bins giving a more detailed distribution.

obj_importer: The VTK OBJ importer is used to load the OBJ file and its associated MTL file, which contains information about the texture image.

actor: In VTK, an actor is an object that represents a graphical object that can be rendered. In this case, the actor represents the 3D model loaded from the OBJ file.

polydata: Polydata is a data structure in VTK that represents geometric objects, such as triangles or lines. In this case, the polydata is extracted from the actor and contains the texture coordinates.

texture_coords: Texture coordinates are 2D coordinates (u, v) that map onto the surface of a 3D model to specify how a texture image should be applied to the model. The texture coordinates are extracted from the polydata and stored in a NumPy array.

texture_hist: The texture histogram is a 2D histogram that shows the frequency of occurrence of each texture coordinate in the [0, 1] range. We compute it using NumPy's histogramdd function, which creates a multi-dimensional histogram.

bins: The bin edges for the texture histogram, which are returned """
