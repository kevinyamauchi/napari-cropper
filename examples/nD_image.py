import numpy as np
import napari
from napari.layers.shapes._shapes_utils import inside_triangles

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
            [-0.5, -0.5, nx],
            [-0.5, ny, nx],
            [nz, ny, nx],
            [nz, -0.5, nx],
        ]
    ),
    'x_neg': np.array(
        [
            [-0.5, -0.5, -0.5],
            [-0.5, ny, -0.5],
            [nz, ny, -0.5],
            [nz, -0.5, -0.5],
        ]
    ),
    'y_pos': np.array(
        [
            [-0.5, ny, -0.5],
            [-0.5, ny, nx],
            [nz, ny, nx],
            [nz, ny, -0.5],
        ]
    ),
    'y_neg': np.array(
        [
            [-0.5, -0.5, -0.5],
            [-0.5, -0.5, nx],
            [nz, -0.5, nx],
            [nz, -0.5, -0.5],
        ]
    ),
    'z_pos': np.array(
        [
            [nz, -0.5, -0.5],
            [nz, -0.5, nx],
            [nz, ny, nx],
            [nz, ny, -0.5],
        ]
    ),
    'z_neg': np.array(
        [
            [-0.5, -0.5, -0.5],
            [-0.5, -0.5, nx],
            [-0.5, ny, nx],
            [-0.5, ny, -0.5],
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
                vertices = face_coords[k]

                tform = viewer.window.qt_viewer.layer_to_visual[viewer.layers[0]].node.get_transform(map_from='visual',
                                                                                                     map_to='canvas')

                # convert the vertex coordinates to canvas coordinates
                vertices_canv = tform.map(np.asarray(vertices)[:, [2, 1, 0]])
                vertices_canv = vertices_canv[:, :2] / vertices_canv[:, 3:]
                triangle_vertices_canv = np.stack((vertices_canv[[0, 1, 2]], vertices_canv[[0, 2, 3]]))
                click_pos_canv = event.pos
                in_triangles = inside_triangles(triangle_vertices_canv - click_pos_canv)
                if in_triangles.sum() > 0:
                    coords += vertices.tolist()
                    print(k)

                    all_vertices += vertices.tolist()
                    triangles = np.array([[0, 1, 2], [0, 2, 3]]) + triangle_offset
                    all_triangles += triangles.tolist()
                    triangle_offset += 4

        viewer.layers['points'].data = coords

        if len(all_vertices) > 0:
            viewer.layers['surface']._vertex_values = np.ones((len(all_vertices)))
            viewer.layers['surface']._faces = np.asarray(all_triangles)
            viewer.layers['surface'].vertices = np.asarray(all_vertices)

            # # get the transform from the visual - this can probably come from the camera?
            # tform = viewer.window.qt_viewer.layer_to_visual[viewer.layers[0]].node.get_transform(map_from='visual', map_to='canvas')
            #
            # # convert the vertex coordinates to canvas coordinates
            # coords_canv = tform.map(np.asarray(all_vertices)[:, [2, 1, 0]])
            # coords_canv = coords_canv[:, :2] / coords_canv[:, 3:]
            # print(coords_canv)
            # print(event.pos)



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