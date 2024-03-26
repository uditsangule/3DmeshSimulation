# Author :udit
# Created on : 23/03/24
# Features :
import numpy as np

import utility.linearalgebra as l3d
from utility import _RTS
from utility import open3d_util as u3d
from utility._filepath import read_json
from utility.open3d_util import o3d


def get_fov(focalpt, dimm): return np.rad2deg(2 * np.arctan2(dimm, 2 * focalpt))


class Camera:
    """Camera class to put all parameters related to its locations , rays """

    def __init__(self, position=(0, 0, 0), lookat=_RTS.zaxis, up=_RTS.yaxis, calibfile=None):
        """
        Camera Setup initilatizer
        :param position: initial position of camera
        :param lookat: looking vector , direction of camera
        :param up: up vector of camera
        :param calibfile: inputpath for the calibration file in .json format, includes K , E matrix.
        """
        self.position = position
        self.lookat = lookat
        self.up = up
        # Todo: if calib not found then ways to predict,load default parameters!
        if calibfile is not None:
            calib = read_json(calibfile)
            self.imgdimm = calib['Camera']['imageshape']
            self.params = o3d.camera.PinholeCameraParameters()
            self.params.intrinsic = o3d.camera.PinholeCameraIntrinsic(width=self.imgdimm[0], height=self.imgdimm[1],
                                                                      intrinsic_matrix=calib['Camera']['intrinsic'])
            self.params.extrinsic = calib['Camera']['extrinsic']
            self.projectionMat = self.params.intrinsic.intrinsic_matrix.dot(
                np.hstack([self.params.extrinsic[:3, :3], np.asarray(self.position)[:, None]]))
        return

    def update_locations(self, position=None, lookat=None, up=None , Tmat=None):
        """
        updates new cordinated of camera!
        :param position: new position  if any
        :param lookat: new lookat vector  if any
        :param up: new up vector if any
        :param Tmat: Transformation Matrix is already given update according to it
        :return: None , updates internal parameters only
        """
        if Tmat is None:
            self.position = position if position is not None else self.position
            self.lookat = lookat if lookat is not None else self.lookat
            self.up = up if up is not None else self.up
            Tmat = np.eye(4)
            Tmat[:3, 3] = self.position
            Tmat[:3, :3] = _RTS.get_rotmat(vec1=_RTS.zaxis, vec2=self.lookat)

        self.params.extrinsic = Tmat
        self.projectionMat = self.params.intrinsic.intrinsic_matrix.dot(
            np.hstack([self.params.extrinsic[:3, :3], np.asarray(self.position)[:, None]]))
        return

    def get_rays(self):
        """
        Generates Rays in direction of camera!
        :return: (source points [N,3] , ray direction vectors( Normalized ) [N,3])
        """
        invk = np.linalg.inv(self.params.intrinsic.intrinsic_matrix)
        Rot, t = self.params.extrinsic[:3, :3], self.params.extrinsic[:3, 3]
        Source = np.dot(Rot, t)
        x, y = np.meshgrid(np.arange(self.imgdimm[0]), np.arange(self.imgdimm[1]))
        px = np.stack((x + 0.5, y + 0.5, np.ones_like(x)), axis=-1)
        for i in range(len(px)): px[i] = np.dot(np.dot(Rot, invk), px[i].T).T
        px /= np.linalg.norm(px, axis=2, keepdims=True)
        return np.full_like(a=px, fill_value=Source), px

    def draw_rays(self, sources, direction, distance=1, colors=(0, 1, 1)):
        """
        Draws Rays from its source points to rays direction. keeping calibration matrix to consideration
        :param sources: source points of rays
        :param direction: rays direction vectors
        :param distance: lengths of rays to vizualize. default = 1
        :param colors: colors of rays
        :return: o3d.geometry.Linesets
        """
        if sources.ndim != 2: sources = sources[np.newaxis, :]
        direction = l3d.tounit(direction)
        targetpts = sources + distance * direction

        # reducing points density for vizualizer
        sources = sources[::2000, :]
        targetpts = targetpts[::2000, :]

        lines = np.stack((np.arange(0, len(sources)), np.arange(len(sources), len(sources) * 2))).T
        points = np.vstack((sources, targetpts))
        return u3d._tolineset(points=points, lines=lines, colors=np.asarray(colors))

    def show(self, ret_=False):
        """
        Normal vizualiser for showing camera, rays!
        :param ret_: if True = show externally!
        :return: list of geometry.Lineset
        """
        camviz = o3d.geometry.LineSet.create_camera_visualization(view_width_px=self.imgdimm[0],
                                                                  view_height_px=self.imgdimm[1],
                                                                  intrinsic=self.params.intrinsic.intrinsic_matrix,
                                                                  extrinsic=self.params.extrinsic, )
        if ret_:
            return [camviz]
        src, dirs = self.get_rays()
        lset = self.draw_rays(sources=np.vstack(src), direction=np.vstack(dirs), colors=[1, 0, 0])
        o3d.visualization.draw_geometries([camviz] + [lset])
        return


if __name__ == '__main__':
    cam = Camera(calibfile='./data/CameraParam.json')
    cam.show()
