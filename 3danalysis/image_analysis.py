import vtk

reader = vtk.vtkOBJReader()
import vtk

# Load the 3D model
reader = vtk.vtkOBJReader()
reader.SetFileName("bunny10k.obj")
reader.Update()

# Create a texture image of the model
texture = vtk.vtkTexture()
texture.SetInputConnection(reader.GetOutputPort())

def calculate_entropy(histogram):
    # Calculate the entropy of a histogram
    num_bins = histogram.GetNumberOfBins()

    # Compute the total number of voxels in the histogram
    voxel_count = histogram.GetVoxelCount()

    # Compute the entropy
    entropy = 0.0
    for i in range(num_bins):
        # Compute the probability of voxel i
        pi = histogram.GetFrequency(i) / voxel_count

        # If pi is zero, skip this bin
        if pi == 0.0:
            continue

        # Compute the entropy contribution of this voxel
        entropy -= pi * math.log2(pi)

    return entropy

################################################
#Texture diversity

# Convert the color texture to grayscale
color_to_gray = vtk.vtkImageLuminance()
color_to_gray.SetInputConnection(texture.GetOutputPort())

# Create a histogram of the grayscale image
histogram = vtk.vtkImageHistogram()
histogram.SetInputConnection(color_to_gray.GetOutputPort())
histogram.SetHistogramBinMinimum(0)
histogram.SetHistogramBinMaximum(255)
histogram.SetHistogramBinCount(256)
histogram.Update()

# Calculate the entropy of the histogram
entropy = calculate_entropy(histogram)

# Print the result
print("Entropy: ", entropy)
################################################
#texture energy

# Create a texture energy filter
texture_energy = vtk.vtkTextureEnergy()
texture_energy.SetInputConnection(texture.GetOutputPort())
texture_energy.Update()

# Print the texture energy value
print("Texture energy:", texture_energy.GetOutput().GetScalarComponentAsDouble(0, 0, 0, 0))

################################################
# Color entropy

# Extract the surface mesh
surface_filter = vtk.vtkDataSetSurfaceFilter()
surface_filter.SetInputConnection(reader.GetOutputPort())
surface_filter.Update()

# Compute the color histogram
histogram = vtk.vtkImageHistogramStatistics()
histogram.SetInputConnection(surface_filter.GetOutputPort())
histogram.SetComponentExtent(0, 255, 0, 0, 0, 0)
histogram.SetComponentOrigin(0.0, 0.0, 0.0)
histogram.SetComponentSpacing(1.0, 1.0, 1.0)
histogram.Update()

# Calculate the entropy of the color histogram
entropy = 0
for i in range(256):
    freq = histogram.GetFrequency(i, 0, 0)
    if freq > 0:
        p_i = freq / histogram.GetCount()
        entropy -= p_i * math.log2(p_i)

print("Color entropy:", entropy)