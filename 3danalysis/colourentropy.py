import vtk

# Load the input data
reader = vtk.vtkOBJReader()
#reader.SetFileName('prezewalskii_JJ14.obj')
reader.SetFileName('fletcheri_JJ89.obj')

reader.Update()

# Get the scalar range of the input data
scalar_range = reader.GetOutput().GetScalarRange()

# Create a color histogram
color_hist = vtk.vtkImageHistogram()
color_hist.SetInputConnection(reader.GetOutputPort())
color_hist.SetNumberOfBins(256)
color_hist.SetAutoCalculateRange(True)
color_hist.Update()

# Calculate the entropy of the color histogram
entropy = 0
total_count = color_hist.GetCount()
for bin_index in range(color_hist.GetNumberOfBins()):
    bin_count = color_hist.GetBinCount(bin_index)
    if bin_count == 0:
        continue
    probability = bin_count / total_count
    entropy -= probability * vtk.vtkMath.Log(probability, 2)

print(f"Color entropy: {entropy}")
