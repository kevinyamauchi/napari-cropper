import napari
import numpy as np
from napari.layers.shapes._shapes_utils import inside_triangles
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox, QPushButton, QVBoxLayout, QWidget

from .constants import FACE_NORMALS
from .utils import (
    bounding_box_to_face_vertices,
    get_view_direction_in_scene_coordinates,
)


class QtCropperWidget(QWidget):
    """The QWdiget containing the controls for the cropper tool

    Parameters
    ----------
    viewer : napari.viewer.Viewer
        The parent napari viewer

    Attributes
    ----------
    """

    def __init__(self, viewer: napari.viewer.Viewer):
        super().__init__()
        self.viewer = viewer

        # create the surface layer for
        vertices = np.array([[0, 0], [0, 20], [10, 0], [10, 10]])
        faces = np.array([[0, 1, 2], [1, 2, 3]])
        values = np.linspace(0, 1, len(vertices))
        surface_layer = viewer.add_surface(
            (vertices, faces, values), name="surface", opacity=0.5, colormap="bop blue"
        )
        surface_layer.mouse_drag_callbacks.append(self._on_click)
        self._bounding_box_surface = surface_layer

        # Create a combobox for selecting layers
        self.layer_combo_box = QComboBox(self)
        self._selected_layer = None
        self.layer_combo_box.currentIndexChanged.connect(self.on_layer_selection)
        self.initialize_layer_combobox()

        self.viewer.layers.events.inserted.connect(self.on_add_layer)
        self.viewer.layers.events.removed.connect(self.on_remove_layer)

        # create a button for cropping the data
        self.crop_button = QPushButton("crop")
        self.crop_button.clicked.connect(self.crop_layer)

        # create the layout
        self.vbox_layout = QVBoxLayout()
        self.vbox_layout.addWidget(self.layer_combo_box)
        self.vbox_layout.addWidget(self.crop_button)

        self.setLayout(self.vbox_layout)

    @property
    def selected_layer(self):
        return self._selected_layer

    @selected_layer.setter
    def selected_layer(self, selected_layer):
        self._selected_layer = selected_layer

    @property
    def bbox_face_coords(self):
        if self.selected_layer is not None:
            bbox = self.viewer.layers[self.selected_layer]._view_bounding_box
            face_coords = bounding_box_to_face_vertices(bbox)
        else:
            face_coords = None
        return face_coords

    def crop_layer(self):
        if self.selected_layer is not None:
            layer = self.viewer.layers[self.selected_layer]
            bbox = layer._bounding_box.astype(np.int)
            bbox[:, 1] = bbox[:, 1] + 1
            cropped_data = layer.data[
                bbox[0, 0] : bbox[0, 1],
                bbox[1, 0] : bbox[1, 1],
                bbox[2, 0] : bbox[2, 1],
            ]
            self.viewer.add_image(
                cropped_data,
                translate=(bbox[0, 0], bbox[1, 0], bbox[2, 0]),
                colormap="bop blue",
            )

    def initialize_layer_combobox(self):
        """Populates the combobox with all layers that contain properties"""
        layer_names = [
            layer.name
            for layer in self.viewer.layers
            if type(layer).__name__ == "Image"
        ]
        self.layer_combo_box.addItems(layer_names)

        if self.selected_layer is None:
            self.selected_layer = layer_names[0]
        index = self.layer_combo_box.findText(self.selected_layer, Qt.MatchExactly)
        self.layer_combo_box.setCurrentIndex(index)

    def on_add_layer(self, event):
        """Callback function that updates the layer list combobox
        when a layer is added to the viewer LayerList.
        """
        layer_name = event.value.name
        layer = self.viewer.layers[layer_name]
        if type(layer).__name__ == "Image":
            self.layer_combo_box.addItem(layer_name)

    def on_remove_layer(self, event):
        """Callback function that updates the layer list combobox
        when a layer is removed from the viewer LayerList.
        """
        layer_name = event.value.name

        index = self.layer_combo_box.findText(layer_name, Qt.MatchExactly)
        # findText returns -1 if the item isn't in the ComboBox
        # if it is in the ComboBox, remove it
        if index != -1:
            self.layer_combo_box.removeItem(index)

            # get the new layer selection
            index = self.layer_combo_box.currentIndex()
            layer_name = self.layer_combo_box.itemText(index)
            if layer_name != self.selected_layer:
                self.selected_layer = layer_name

    def on_layer_selection(self, index: int):
        """Callback function that updates the table when a
        new layer is selected in the combobox.
        """
        if index != -1:
            layer_name = self.layer_combo_box.itemText(index)
            self.selected_layer = layer_name

    def _on_click(self, layer, event):
        if self.selected_layer is not None:
            print("hi")
            view_box = self.viewer._window.qt_viewer.view
            view_dir = get_view_direction_in_scene_coordinates(view_box)
            print(view_dir)
            coords = []

            all_vertices = []
            all_triangles = []
            triangle_offset = 0

            self.selected_face = None
            for k, v in FACE_NORMALS.items():
                if (np.dot(view_dir, v) + 0.001) < 0:
                    vertices = self.bbox_face_coords[k]

                    tform = self.viewer.window.qt_viewer.layer_to_visual[
                        self.viewer.layers[0]
                    ].node.get_transform(map_from="visual", map_to="canvas")

                    # convert the vertex coordinates to canvas coordinates
                    vertices_canv = tform.map(np.asarray(vertices)[:, [2, 1, 0]])
                    vertices_canv = vertices_canv[:, :2] / vertices_canv[:, 3:]
                    triangle_vertices_canv = np.stack(
                        (vertices_canv[[0, 1, 2]], vertices_canv[[0, 2, 3]])
                    )
                    click_pos_canv = event.pos
                    in_triangles = inside_triangles(
                        triangle_vertices_canv - click_pos_canv
                    )
                    if in_triangles.sum() > 0:
                        coords += vertices.tolist()
                        print(k)

                        all_vertices += vertices.tolist()
                        triangles = np.array([[0, 1, 2], [0, 2, 3]]) + triangle_offset
                        all_triangles += triangles.tolist()
                        triangle_offset += 4
                        self.selected_face = k

            if self.selected_face is not None:
                self._bounding_box_surface._vertex_values = np.ones(len(all_vertices))
                self._bounding_box_surface._faces = np.asarray(all_triangles)
                self._bounding_box_surface.vertices = np.asarray(all_vertices)
                self._bounding_box_surface.interactive = False

                selected_face_normal = FACE_NORMALS[self.selected_face]
                bounding_box_axis = np.squeeze(np.argwhere(selected_face_normal))
                original_bounding_box = self.viewer.layers[
                    self.selected_layer
                ]._view_bounding_box[bounding_box_axis]
                if selected_face_normal.sum() > 0:
                    # modify the max, if in the positive direction
                    bbox_index = 1
                else:
                    # modify the max, if in the negative direction
                    bbox_index = 0

            else:
                self._bounding_box_surface.interactive = True
            dragged = False

            yield
            # drag stuff
            if self.selected_face is not None:
                while event.type == "mouse_move":
                    print(self.selected_face)
                    end_pos_canv = event.pos
                    drag_vector_canv = end_pos_canv - click_pos_canv
                    tform = self.viewer.window.qt_viewer.layer_to_visual[
                        self.viewer.layers[0]
                    ].node.get_transform(map_from="visual", map_to="canvas")
                    face_normal_data = 10 * FACE_NORMALS[self.selected_face]
                    face_normal_scene = face_normal_data[[2, 1, 0]]
                    coords_to_convert = np.stack([face_normal_scene, [0, 0, 0]])
                    face_normal_points_homogeneous = tform.map(coords_to_convert)
                    face_normal_points_canv = (
                        face_normal_points_homogeneous[:, :2]
                        / face_normal_points_homogeneous[:, 3:]
                    )
                    face_normal_canv = (
                        face_normal_points_canv[0] - face_normal_points_canv[1]
                    )
                    face_normal_canv = face_normal_canv / np.linalg.norm(
                        face_normal_canv
                    )

                    drag_displacement = np.dot(drag_vector_canv, face_normal_canv)
                    # print('face_norm: ', face_normal_data)
                    # print('canv: ', face_normal_canv)
                    # print(drag_displacement)
                    # print('\n')
                    new_bbox = original_bounding_box.copy()
                    if bbox_index == 0:
                        new_lim = original_bounding_box[bbox_index] - drag_displacement
                        new_lim = np.max([new_lim, 0])
                        new_lim = np.min([new_lim, original_bounding_box[1]])
                        new_bbox[bbox_index] = new_lim
                    else:
                        new_lim = original_bounding_box[bbox_index] + drag_displacement
                        new_lim = np.min(
                            [
                                new_lim,
                                self.viewer.layers[
                                    self.selected_layer
                                ]._data_view.shape[bounding_box_axis],
                            ]
                        )
                        new_lim = np.max([new_lim, original_bounding_box[0]])
                        new_bbox[bbox_index] = new_lim

                    self.viewer.layers[self.selected_layer]._set_bbox_lim(
                        new_bbox, bounding_box_axis
                    )

                    # update the selected surface
                    vertices = self.bbox_face_coords[self.selected_face]
                    triangles = np.array([[0, 1, 2], [0, 2, 3]])
                    values = np.ones(len(vertices))
                    self._bounding_box_surface._vertex_values = values
                    self._bounding_box_surface._faces = triangles
                    self._bounding_box_surface.vertices = vertices

                    dragged = True
                    yield

            # end drag stuff
            verts = self._bounding_box_surface.vertices
            self._bounding_box_surface.vertex_values = np.zeros(len(verts))
            self._bounding_box_surface.vertices = verts
            if dragged:
                print("drag end")
            else:
                print("clicked!")

            self._bounding_box_surface.interactive = True
