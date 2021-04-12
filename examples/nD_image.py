import numpy as np
import napari
from napari.layers.shapes._shapes_utils import triangulate_face

nz = 128
ny = 128
nx = 128
data = np.random.random((nz, ny, nz))

face_normals = {
    'x_pos': np.array([0, 0, 1]),
    'x_neg': np.array([0, 0, -1]),
    'y_pos': np.array([0, 1, 0]),
    'y_neg': np.array([0, -1, 0]),
    'z_pos': np.array([1, 0, 0]),
    'z_neg': np.array([-1, 0, 0])
}

face_coords = {
    'x_pos': np.array(
        [
            [0, 0, nx],
            [0, ny, nx],
            [nz, ny, nx],
            [nz, 0, nx],
        ]
    ),
    'x_neg': np.array(
        [
            [0, 0, 0],
            [0, ny, 0],
            [nz, ny, 0],
            [nz, 0, 0],
        ]
    ),
    'y_pos': np.array(
        [
            [0, ny, 0],
            [0, ny, nx],
            [nz, ny, nx],
            [nz, ny, 0],
        ]
    ),
    'y_neg': np.array(
        [
            [0, 0, 0],
            [0, 0, nx],
            [nz, 0, nx],
            [nz, 0, 0],
        ]
    ),
    'z_pos': np.array(
        [
            [nz, 0, 0],
            [nz, 0, nx],
            [nz, ny, nx],
            [nz, ny, 0],
        ]
    ),
    'z_neg': np.array(
        [
            [0, 0, 0],
            [0, 0, nx],
            [0, ny, nx],
            [0, ny, 0],
        ]
    )
}


with napari.gui_qt():

    viewer = napari.view_image(data)
    points_layer = viewer.add_points(name='points', ndim=3)
    vertices = np.array([[0, 0], [0, 20], [10, 0], [10, 10]])
    faces = np.array([[0, 1, 2], [1, 2, 3]])
    values = np.linspace(0, 1, len(vertices))
    surface_layer = viewer.add_surface((vertices, faces, values), name='surface')


    @viewer.mouse_drag_callbacks.append
    def get_event(viewer, event):
        # print('\n')
        # print(viewer.camera)
        # print(viewer.cursor)
        # print(event.pos)
        # print(viewer._window.qt_viewer._canvas_corners_in_world)
        # print('\n')

        view_box = viewer._window.qt_viewer.view
        view_dir = get_view_direction_in_scene_coordinates(view_box)
        print(view_dir)
        coords = []


        all_vertices = []
        all_triangles = []
        triangle_offset = 0
        for k, v in face_normals.items():
            if (np.dot(view_dir, v) + 0.001) < 0:
                vertices = face_coords[k].tolist()
                coords += vertices
                print(k)

                all_vertices += vertices
                triangles = np.array([[0, 1, 2], [0, 2, 3]]) + triangle_offset
                all_triangles += triangles.tolist()
                triangle_offset += 4

        viewer.layers['points'].data = coords

        viewer.layers['surface']._vertex_values = np.ones((len(all_vertices)))
        viewer.layers['surface']._faces = np.asarray(all_triangles)
        viewer.layers['surface'].vertices = np.asarray(all_vertices)

        # get the transform from the visual - this can probably come from the camera?
        tform = viewer.window.qt_viewer.layer_to_visual[viewer.layers[0]].node.get_transform(map_from='visual', map_to='canvas')

        # convert the vertex coordinates to canvas coordinates
        coords_canv = tform.map(np.asarray(all_vertices)[:, [2, 1, 0]])
        coords_canv = coords_canv[:, :2] / coords_canv[:, 3:]
        print(coords_canv)
        print(event.pos)



    def get_view_direction_in_scene_coordinates(view):
        tform = view.scene.transform
        w, h = view.canvas.size
        screen_center = np.array([w / 2, h / 2, 0, 1])  # in homogeneous screen coordinates
        d1 = np.array([0, 0, 1, 0])  # in homogeneous screen coordinates
        point_in_front_of_screen_center = screen_center + d1  # in homogeneous screen coordinates
        p1 = tform.imap(point_in_front_of_screen_center)  # in homogeneous scene coordinates
        p0 = tform.imap(screen_center)  # in homogeneous screen coordinates
        assert (abs(p1[3] - 1.0) < 1e-5)  # normalization necessary before subtraction
        assert (abs(p0[3] - 1.0) < 1e-5)
        d2 = p1 - p0  # in homogeneous screen coordinates
        assert (abs(d2[3]) < 1e-5)
        d3 = d2[0:3]  # in 3D screen coordinates
        d4 = d3 / np.linalg.norm(d3)
        return d4[[2, 1, 0]]