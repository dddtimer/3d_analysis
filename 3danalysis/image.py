import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

# Load the 3D model data
reader = vtk.vtkOBJReader()
reader.SetFileName("prezewalskii_JJ14.obj")
reader.Update()
poly_data = reader.GetOutput()

# Get the texture coordinates
texture_coords = vtk_to_numpy(poly_data.GetPointData().GetTCoords())

# Calculate the texture diversity using the entropy approach
unique_coords, counts = np.unique(texture_coords, axis=0, return_counts=True)
probabilities = counts / len(texture_coords)
entropy = -np.sum(probabilities * np.log2(probabilities))
texture_diversity = 1 - entropy

# Define the texture as a 1D numpy array
texture = np.zeros(len(texture_coords))

# Create the texture image
texture_image = vtk.vtkImageData()
texture_image.SetDimensions(1, len(texture_coords), 1)
texture_image.SetSpacing(1, 1, 1)
texture_image.SetOrigin(0, 0, 0)
texture_image.AllocateScalars(vtk.VTK_FLOAT, 1)
texture_image.GetPointData().GetScalars().Fill(0)
texture_dims = [100000, 100000]
for i in range(len(texture_coords)):
    for j in range(texture_dims[1]):
        tex_coord = [float(i) / len(texture_coords), float(j) / texture_dims[1], 0.0]
        pt_id = poly_data.FindPoint(tex_coord)
        if pt_id >= 0:
            tex_val = texture[pt_id]
            texture_image.SetScalarComponentFromFloat(i, j, 0, 0, tex_val)


# Calculate the GLCM and its entropy
histogram = vtk.vtkImageHistogramStatistics()
histogram.SetInputData(texture_image)
histogram.SetAutoRangePercentiles([0, 99])
histogram.SetComputeEntropy(True)
histogram.Update()

histogram.GetOutput().GetPointData().GetScalars().SetName("ImageScalars")
image_array = vtk_to_numpy(histogram.GetOutput().GetPointData().GetScalars())
probabilities = np.histogram(image_array, bins=256)[0] / float(len(image_array))
entropy = -np.sum(probabilities * np.log2(probabilities))
