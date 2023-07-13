
import numpy as np
import vtk

# Load 3D object file
reader = vtk.vtkOBJReader()
#reader.SetFileName('prezewalskii_JJ14.obj')
reader.SetFileName('fletcheri_JJ89.obj')
reader.SetFileNameMTL('fletcheri_JJ89.mtl')
reader.Update()

# Extract texture coordinates
polydata = reader.GetOutput()
texture_coords = polydata.GetPointData().GetTCoords()

# Calculate entropy of texture coordinates
texture_hist, texture_bins = np.histogram(texture_coords, bins=256, range=(0.0, 1.0))
texture_prob = texture_hist / np.sum(texture_hist)
texture_entropy = -np.sum(texture_prob * np.log2(texture_prob))
print("Texture entropy:", texture_entropy)
