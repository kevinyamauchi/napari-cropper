import numpy as np
import napari


nz = 128
ny = 128
nx = 128
data = np.random.random((nz, ny, nz))


with napari.gui_qt():
    viewer = napari.view_image(data)
    viewer.dims.ndisplay = 3
