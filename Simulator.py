import os
import time

import numpy as np
import pandas as pd
from open3d import core as o3c

from camera import Camera
from utility import _RTS
from utility import _filepath as fp
from utility import open3d_util as o3d


def get_pixel():
    return


class UnProjector:
    def __init__(self, inputdir, outputdir, filename=None, pose_format='slam'):
        self.inputdir = inputdir
        self.outputdir = outputdir
        self.pose_format = pose_format
        # reading mesh from path , else empty mesh will be loaded, however creating mesh may be injected in future!
        self.mesh = o3d.mesh_io(
            self.inputdir + os.sep + filename)  # if filename is not None else './data/Simple1/mesh.obj')
        self.textureimg = np.asarray(self.mesh.textures[0])
        self.bbox = self.mesh.get_oriented_bounding_box()
        self.bbox.color = (1, 0, 0)
        self.cam = Camera(calibfile=self.inputdir + os.sep + 'CameraParam.json', position=self.mesh.get_center())
        self.viz = o3d.o3d.visualization.Visualizer()
        return

    def _create_scenes_(self):
        scene = o3d.o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.o3d.t.geometry.TriangleMesh.from_legacy(self.mesh))
        return scene

    def _export(self, pose_dict, outdir=None):
        outdir = fp.Fstatus(self.outputdir if outdir is None else outdir)
        # saving depth images:
        for d in pose_dict:
            o3d.write_depth(image=d['depth_im'], path=fp.Fstatus(outdir + os.sep + '__depth'), filename=str(d['id']))
        # save calib.yaml
        calibfile = []
        for d in pose_dict:
            data = {'camera': {'id': d['id'], 'image_dimm': d['imagedimm']},
                    'intrinsic': {'n_rows': 3, 'n_cols': 3, 'matrix': d['intr'].flatten().tolist()}}
            fp.save_calibration(outputdir=fp.Fstatus(outdir + os.sep + '_calib'), data=data, camid=d['id'])

        # save poses.csv
        D1 = pd.DataFrame.from_dict(pose_dict).drop(columns=['poseext', 'transformation', 'imagedimm', 'intr'])
        if self.pose_format == 'raw':
            cols = ['Rx1', 'Rx2', 'Rx3', 'Tx', 'Ry1', 'Ry2', 'Ry3', 'Ty', 'Rz1', 'Rz2', 'Rz3', 'Tz']
            P = [d['transformation'].flatten()[:-4].tolist() for d in pose_dict]
        elif self.pose_format == 'slam':
            P = []
            cols = ['qw', 'qx', 'qy', 'qz', 'x', 'y', 'z']
            for d in pose_dict:
                t = d['transformation'][:3, 3].tolist();
                Q = _RTS.Tmatrix_to_quats(d['transformation']);
                P.append([Q.w, Q.x, Q.y, Q.z, t[0], t[1], t[2]])
        else:
            # Todo: addition of more pose format!
            cols = None
        pd.concat((D1, pd.DataFrame(P, columns=cols)), axis=1).to_csv(outdir + os.sep + '__poses.csv', sep=',',
                                                                      index=False)

        return

    def validate(self):
        # Todo: to validate if depth frame and Transformation can form back the mesh!
        return

    def show(self):
        """
        Visualizes the geometry with its boundingbox
        :return:
        """
        o3d.NormalViz([self.mesh] + [o3d.axis_mesh(size=1)] + self.cam.show(ret_=True) + [self.bbox])
        return

    def run(self, save=False, show=False, iter_=20, waitkey=1000):
        """
        Ray Casting to mesh to get the depth images , poses , etc.
        :param save: enables saving the results
        :param show: enables visualizer to check intersections and depth_image.
        :return:
        """
        scene = self._create_scenes_()
        triangles = np.asarray(self.mesh.triangles)
        vertices = np.asarray(self.mesh.vertices)
        pose_dict = []
        tic = time.time_ns()
        for i in range(1, iter_):
            print(i)
            tmat = _RTS.get_random_Tmatrix(extends=(-5, 5))
            self.cam.update_locations(Tmat=tmat)

            # rays = o3c.Tensor(np.concatenate(self.cam.get_rays(), axis=-1).astype(np.float32))
            rays = scene.create_rays_pinhole(intrinsic_matrix=o3c.Tensor(self.cam.params.intrinsic.intrinsic_matrix),
                                             extrinsic_matrix=o3c.Tensor(self.cam.params.extrinsic),
                                             width_px=self.cam.imgdimm[0], height_px=self.cam.imgdimm[1])

            intersections = scene.count_intersections(rays).numpy()
            if np.count_nonzero(intersections) / intersections.size < .1:
                # if less than 10% intersection occurs then continue
                print("less than 10% intersections found!")
                continue
            ans = scene.cast_rays(rays)
            face_ids = ans['primitive_ids'].numpy().flatten()
            fidx = face_ids[np.where(face_ids != scene.INVALID_ID)[0]]
            uq_tri = np.unique(triangles[fidx], axis=0)
            points3d = o3d._topcd(points=np.vstack(vertices[uq_tri]))
            depth_im = o3d.process_depth(ans['t_hit'].numpy())

            pose_dict.append({'id': i, 'transformation': tmat, 'poseext': self.cam.params.extrinsic,
                              'intr': self.cam.params.intrinsic.intrinsic_matrix, 'timestamp': time.time_ns() - tic,
                              'imagedimm': self.cam.imgdimm, 'depth_im': depth_im})
            if show:
                o3d.show(image=depth_im, waitkey=waitkey)
                o3d.NormalViz([points3d] + [self.mesh] + self.cam.show(True) + [o3d.axis_mesh(size=1)] + [
                    self.cam.draw_rays(direction=np.vstack(rays.numpy()[:, :, 3:]),
                                       sources=np.vstack(rays.numpy()[:, :, :3]))])
        if save:
            self._export(pose_dict)
        return


if __name__ == '__main__':
    inputdir = os.getcwd() + f'{os.sep}data'
    outdir = os.getcwd() + f'{os.sep}data{os.sep}Arway'
    App = UnProjector(inputdir, outdir, filename='OfficeArway.glb')
    App.run(save=True, show=True)
