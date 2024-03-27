# Author :udit
# Created on : 22/03/24
import os

import cv2
import numpy as np
import open3d as o3d

vec3d = o3d.utility.Vector3dVector
vec3i = o3d.utility.Vector3iVector

vec2d = o3d.utility.Vector2dVector
vec2i = o3d.utility.Vector2iVector


def NormalViz(geometrylist, _ui=0):
    """
    This function shows the geometries in open3d visualizer
    :param geometrylist: list containing all the geometries
    :param _ui: enables with UI
    """
    if not _ui:
        o3d.visualization.draw_geometries(geometrylist)
    else:
        o3d.visualization.draw(geometrylist, show_ui=True)
    return


def mesh_io(path, save=None):
    if save is None:
        return o3d.io.read_triangle_mesh(path, enable_post_processing=True, print_progress=True)
    o3d.io.write_triangle_mesh(path, print_progress=True, write_ascii=True)
    return


def axis_mesh(origin=(0, 0, 0), size=5):
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin)


def _tolineset(points=None, lines=None, colors=np.asarray([1, 0, 0])):
    """ create linesets """
    lineset = o3d.geometry.LineSet()
    if points is None: return lineset
    lineset.points = vec3d(np.asarray(points))
    if len(colors.shape) < 2:
        lineset.paint_uniform_color(colors)
    else:
        lineset.colors = vec3d(np.asarray(colors))
    if lines is None: return lineset
    lineset.lines = vec2i(np.asarray(lines))
    return lineset


def _drawPoses(campoints):
    """plots the poseses"""
    nline = len(campoints) - 1
    col = (np.column_stack((np.arange(0, nline), np.arange(nline, 0, -1), np.zeros(nline))) / nline)
    return _tolineset(points=campoints, colors=col, lines=[[i, i + 1] for i in range(nline)])


def _topcd(points=None, colors=np.asarray([0, 0, 1]), normals=None, filepath=None):
    """creates the pointcloud from points"""
    pcd = o3d.geometry.PointCloud()
    if points is None: return pcd
    pcd.points = vec3d(np.asarray(points))
    if type(colors) == 'list': colors = np.asarray(colors)
    if colors.ndim == 1:
        pcd.paint_uniform_color(colors)
    else:
        pcd.colors = vec3d(colors)
    if normals is not None: pcd.normals = vec3d(normals)
    if filepath is not None:
        o3d.io.write_point_cloud(filename=filepath, pointcloud=pcd, write_ascii=1, print_progress=1)
    return pcd


def images_topcd(depth_im, rgb_im, intr, extr):
    rgb_im = o3d.geometry.Image((rgb_im * 255.).astype(np.uint8))
    depth_im = o3d.geometry.Image(depth_im)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color=rgb_im, depth=depth_im, depth_trunc=10000,
                                                              depth_scale=1000, convert_rgb_to_intensity=False)
    pointcloud = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd, intrinsic=intr, extrinsic=extr)
    pointcloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return pointcloud, rgbd


def _tomesh(vertices=None, triangles=None, normals=False, colors=np.asarray([0, 1, 0]), anticlock=0, filepath=None,
            *args, **kwargs):
    """creates mesh from vertices and triangles"""
    mesh = o3d.geometry.TriangleMesh()
    if vertices is None: return mesh
    mesh.vertices = vec3d(np.asarray(vertices))
    if len(colors.shape) < 2:
        mesh.paint_uniform_color(colors)
    else:
        mesh.vertex_colors = vec3d(colors)
    if triangles is None: return mesh
    _anticlock = [tri[::-1] for tri in triangles] if anticlock else []  # triangles[...,::-1]
    mesh.triangles = vec3i(np.asarray(_anticlock + triangles))
    if not normals: return mesh
    mesh.compute_vertex_normals(normalized=1)
    mesh.compute_triangle_normals(normalized=1)
    if filepath is not None:
        o3d.io.write_triangle_mesh(mesh=mesh, write_ascii=True, filename=filepath, print_progress=True)
    return mesh


# CV2 or 2D images area!

def show(image, windowname='depth', waitkey=1000, dest=True):
    cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
    cv2.imshow(windowname, image)
    cv2.waitKey(waitkey)
    if waitkey != 0 or dest:
        cv2.destroyWindow(windowname)
    return


def write_(image, path, filename='1', ext='.png'):
    assert image.ndim in (1, 2, 3)
    assert image.dtype in [np.uint8, np.float32, np.float64]
    if image.dtype == np.float32 or image.dtype == np.float64:
        min_, max_ = image.min(), image.max()
        image = np.round(image * 255).astype(np.uint8)
    return cv2.imwrite(path + os.sep + filename + ext, image)


def process_depth(image, depthscale=1000, depth_trunc=0):
    image[np.where(image == np.inf)] = depth_trunc
    return (image * depthscale).astype(np.uint16)


def write_depth(image, path=None, filename='1', ext='.png'):
    if path is not None: cv2.imwrite(os.path.join(path, filename) + ext, image)
    return image
