#Good version

import numpy as np
import vtk

# Load 3D object and MTL files
importer = vtk.vtkOBJImporter()
importer.SetFileName("fletcheri_JJ89.obj")
importer.SetFileNameMTL('fletcheri_JJ89.mtl')
importer.Read()
importer.Update()


# Get output from importer
actor = importer.GetRenderer().GetActors().GetLastActor()
polydata = actor.GetMapper().GetInput()
# Extract texture coordinates
texture_coords = polydata.GetPointData().GetTCoords()

# Calculate entropy of texture coordinates
texture_hist, texture_bins = np.histogram(texture_coords, bins=256, range=(0.0, 1.0))
texture_prob = texture_hist / np.sum(texture_hist)
texture_entropy = -np.sum(texture_prob * np.log2(texture_prob))
print("Texture entropy:", texture_entropy)
