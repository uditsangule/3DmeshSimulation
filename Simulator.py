import os
from open3d import core as o3c
from camera import Camera
from utility import open3d_util as o3d
import numpy as np

class UnProjector:
    def __init__(self, inputdir, outputdir, filename=None , pose_format = 'raw'):
        self.inputdir = inputdir
        self.outputdir = outputdir

        # reading mesh from path , else empty mesh will be loaded, however creating mesh may be injected in future!
        self.mesh = o3d.mesh_io(self.inputdir + os.sep + filename if filename is not None else './data/Simple1/mesh.obj')
        self.cam = Camera(calibfile=self.inputdir + os.sep + 'CameraParam.json')
        return

    def _create_scenes_(self):
        scene = o3d.o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.o3d.t.geometry.TriangleMesh.from_legacy(self.mesh))
        return scene

    def _export(self, outdir=None):
        outdir = self.outputdir if outdir is None else outdir
        # save rgb , depth images
        # save calib.yaml
        # save poses.txt
        # save pcd , mesh of frame # if any
        return

    def validate(self):
        return

    def show(self):
        o3d.NormalViz([self.mesh] + [o3d.axis_mesh(size=1)])
        return

    def run(self, camlocations=None, camlooks=None, save=False):
        scene = self._create_scenes_()
        for _ in range(10):
            rays = scene.create_rays_pinhole(intrinsic_matrix= o3c.Tensor(self.cam.params.intrinsic.intrinsic_matrix),
                                             extrinsic_matrix=o3c.Tensor(self.cam.params.extrinsic) ,width_px=self.cam.imgdimm[0] ,height_px= self.cam.imgdimm[1])
            intersections = scene.count_intersections(rays).numpy()
            if np.count_nonzero(intersections) > 1:
                print(1)
            ans = scene.cast_rays(rays)
            img = ans['t_hit'].numpy()
            k=1
        if save: self._export()
        return


if __name__ == '__main__':
    inputdir = outdir = './data'
    App = UnProjector(inputdir, outdir)
    App.show()
    App.run(save=True)
