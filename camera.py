# Author :udit
# Created on : 23/03/24
# Features :
import numpy as np
import os
from utility import _RTS
from utility._filepath import read_json
from utility.open3d_util import o3d

def get_fov(focalpt , dimm): return np.rad2deg(2 * np.arctan2(dimm , 2 * focalpt))

class Camera:

    def __init__(self, position=(0,0,0) , lookat = _RTS.zaxis , up=_RTS.yaxis , calibfile=None):
        self.position = position
        self.lookat = lookat
        self.up = up
        if calibfile is not None:
            calib =  read_json(calibfile)
            self.imgdimm = calib['Camera']['imageshape']
            self.params = o3d.camera.PinholeCameraParameters()
            self.params.intrinsic = o3d.camera.PinholeCameraIntrinsic( width = self.imgdimm[0] ,
                                                                height = self.imgdimm[1], intrinsic_matrix=calib['Camera']['intrinsic'])
            #self.params.extrinsic = calib['Camera']['extrinsic']

        return

    def update_locations(self , position , lookat , up=None):
        self.position = position
        self.lookat = lookat
        if up is not None: self.up = up
        return

    def get_rays(self):
        np.meshgrid(self.imgdimm[0] , self.imgdimm[1])
        n_rays = self.imgdimm[0] * self.imgdimm[1]
        rays_src = np.repeat(self.position , repeats=n_rays,axis=0)
        return

if __name__ == '__main__':
    cam = Camera(calibfile='./data/CameraParam.json')
    print(cam.params.extrinsic , cam.params.intrinsic.intrinsic_matrix)