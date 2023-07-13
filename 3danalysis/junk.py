from mayavi import mlab
from mayavi.core.api import Engine

# create an engine
engine = Engine()
engine.start()

# create a new scene and add it to the engine
scene = engine.new_scene()

# create a figure and set its scene to the new scene
fig = mlab.figure(engine=engine)
fig.scene = scene

# Generate some example 3D texture data
texture_3d = np.random.rand(64, 64, 64)
from mayavi import mlab
mlab.options.backend = 'test'
# Create a scalar field from the texture data
src = mlab.pipeline.scalar_field(texture_3d)

# Create a volume rendering of the scalar field
vol = mlab.pipeline.volume(src)
mlab.show()
# Save the figure to a PNG file
mlab.savefig('figure.png')

# Close the figure and engine
mlab.close(fig)
engine.stop()
