from napari_plugin_engine import napari_hook_implementation

from .qt_cropper_widget import QtCropperWidget


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return (QtCropperWidget, {"area": "right", "name": "cropping widget"})
