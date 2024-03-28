import os
import time

import numpy as np
import pandas as pd
from open3d import core as o3c

from camera import Camera
from utility import _RTS
from utility import _filepath as fp
from utility import open3d_util as o3d


def get_random(min_bb, max_bb, size=30):
    """
    Generates random trajectory within bounding box
    :param min_bb:
    :param max_bb:
    :param size:
    :return:
    """
    x, y, z = np.arange(min_bb[0], max_bb[0]), np.arange(min_bb[1], max_bb[1]), np.arange(min_bb[2], max_bb[2])
    campos = np.vstack((np.sort(np.random.choice(x, size=size, replace=True)),
                        np.random.choice(y, size=size, replace=True),
                        np.sort(np.random.choice(z, size=size, replace=True)))).T

    return campos


def UV2rgb(texture_image, triangle_uvs, vertices, triangles):
    """
    Converts uv map to rgb color by extracting colors from image
    :param texture_image: texture image
    :param triangle_uvs: uv mapping of texture to triangles , 3N,2
    :param vertices: vertices of mesh
    :param triangles: triangles of mesh
    :return: vertex colors , triangles colors
    """
    texture_height, texture_width, _ = texture_image.shape

    # Converting uv w.r.t to image idx
    uv_indices = np.round(triangle_uvs * np.array([texture_width, texture_height])).astype(int)

    # Extract colors from the texture image
    triangle_colors = texture_image[uv_indices[:, 1], uv_indices[:, 0]]
    triangle_colors = triangle_colors / 255.
    triangle_colors = triangle_colors.reshape((len(triangles), 3, 3))

    # Initialize an array to store vertex colors , triangle colors
    vertex_colors = np.zeros((len(vertices), 3), dtype=np.float32)
    vertex_colors[triangles] = triangle_colors

    return vertex_colors, np.mean(triangle_colors, axis=1)[:, [2, 1, 0]]


class UnProjector:
    def __init__(self, inputdir, outputdir, filename=None, pose_format='slam'):
        """
        :param inputdir:
        :param outputdir:
        :param filename:
        :param pose_format:
        """
        self.inputdir = inputdir
        self.outputdir = outputdir
        self.pose_format = pose_format

        # reading mesh from path , else empty mesh will be loaded, however creating mesh may be injected in future!
        self.mesh = o3d.mesh_io(
            self.inputdir + os.sep + filename)  # if filename is not None else './data/Simple1/mesh.obj')
        self.textureimg = np.asarray(self.mesh.textures[0])
        self.bbox = self.mesh.get_oriented_bounding_box()
        self.bbox.color = (1, 0, 0)
        self.campose = get_random(min_bb=self.bbox.get_min_bound(), max_bb=self.bbox.get_max_bound())

        self.cam = Camera(calibfile=self.inputdir + os.sep + 'CameraParam.json', position=self.mesh.get_center())
        self.viz = o3d.o3d.visualization.Visualizer()
        self.vcol, self.tricolor = UV2rgb(triangles=np.asarray(self.mesh.triangles),
                                          vertices=np.asarray(self.mesh.vertices), texture_image=self.textureimg,
                                          triangle_uvs=np.asarray(self.mesh.triangle_uvs))
        return

    def _create_scenes_(self):
        scene = o3d.o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.o3d.t.geometry.TriangleMesh.from_legacy(self.mesh))
        return scene

    def _export(self, pose_dict, outdir=None):
        print("Saving Files on ...")
        outdir = fp.Fstatus(self.outputdir if outdir is None else outdir)
        if len(pose_dict) < 1:
            print("NO files to save!")
            return

        # saving depth ,rgb images , pcd
        for d in pose_dict:
            o3d.write_depth(image=d['depth_im'], path=fp.Fstatus(outdir + os.sep + '__depth'), filename=str(d['id']))
            o3d.write_(image=d['rgb'], path=fp.Fstatus(outdir + os.sep + '__rgb'), filename=str(d['id']))
            pcd, rgbd = o3d.images_topcd(depth_im=d['depth_im'], rgb_im=d['rgb'], intr=self.cam.params.intrinsic,
                                         extr=self.cam.params.extrinsic)

            o3d.o3d.io.write_point_cloud(filename=fp.Fstatus(outdir + os.sep + '__pcd') + os.sep + f'{d["id"]}.ply',
                                         write_ascii=True, print_progress=True, pointcloud=pcd)

        # save calib.yaml
        calibfile = []
        for d in pose_dict:
            data = {'camera': {'id': d['id'], 'image_dimm': d['imagedimm']},
                    'intrinsic': {'n_rows': 3, 'n_cols': 3, 'matrix': d['intr'].flatten().tolist()}}
            fp.save_calibration(outputdir=fp.Fstatus(outdir + os.sep + '_calib'), data=data, camid=d['id'])

        # save poses.csv
        D1 = pd.DataFrame.from_dict(pose_dict).drop(
            columns=['poseext', 'transformation', 'depth_im', 'rgb', 'imagedimm', 'intr'])
        if self.pose_format == 'raw':
            cols = ['Rx1', 'Rx2', 'Rx3', 'Tx', 'Ry1', 'Ry2', 'Ry3', 'Ty', 'Rz1', 'Rz2', 'Rz3', 'Tz']
            P = [d['transformation'][:3,:].flatten().tolist() for d in pose_dict]
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
        print(f"Done! Saved on :{outdir}")
        return

    def validate(self):
        # Todo: to validate if depth frame and Transformation can form back the mesh!
        return

    def show(self):
        """
        Visualizes the geometry with its boundingbox
        :return:
        """
        o3d.NormalViz([self.mesh] + [self.bbox] + [o3d.axis_mesh(size=3, origin=[0, 0, 0])] + self.cam.show(ret_=True))
        return

    def run(self, save=False, show=False, N=4, waitkey=1000):
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
        campose = [[0, 0, 0]]

        for i in range(1, N):
            print(i)
            # tmat[:3,:3] = _RTS.get_rotmat(vec1=[0,0,1] , vec2= [0,0,i*5 if i < 3 else 2])
            tmat = np.eye(4)
            tmat[0, :3] = [1, 0, 0]
            tmat[1, :3] = [0, -1, 0]
            tmat[:3, 3] = [1, 1, i]

            self.cam.update_locations(Tmat=tmat)

            # rays1 = o3c.Tensor(np.concatenate(self.cam.get_rays(), axis=-1).astype(np.float32))
            rays = scene.create_rays_pinhole(intrinsic_matrix=o3c.Tensor(self.cam.params.intrinsic.intrinsic_matrix),
                                             extrinsic_matrix=o3c.Tensor(self.cam.params.extrinsic),
                                             width_px=self.cam.imgdimm[0], height_px=self.cam.imgdimm[1])

            intersections = scene.count_intersections(rays).numpy()
            if np.count_nonzero(intersections) / intersections.size < .1:
                # if less than 10% intersection occurs then continue
                print("less than 10% intersections found!")
                continue
            ans = scene.cast_rays(rays)
            hits = ans['primitive_ids'].numpy()

            # getting rgbimage from triangle colors
            inv_idx = np.where(hits == scene.INVALID_ID)
            hits[inv_idx] = 0
            rgb_im = self.tricolor[hits]

            # getting depth image
            face_ids = hits.flatten()
            fidx = face_ids[np.where(face_ids != scene.INVALID_ID)[0]]
            uq_tri = np.unique(triangles[fidx], axis=0)

            depth_im = o3d.process_depth(ans['t_hit'].numpy())

            pose_dict.append({'id': i, 'transformation': tmat.copy(), 'poseext': self.cam.params.extrinsic,
                              'intr': self.cam.params.intrinsic.intrinsic_matrix, 'timestamp': time.time_ns() - tic,
                              'imagedimm': self.cam.imgdimm, 'depth_im': depth_im, 'rgb': rgb_im})
            if show:
                points3d = o3d._topcd(points=np.vstack(vertices[uq_tri]), colors=np.vstack(self.vcol[uq_tri]))
                pcd, _ = o3d.images_topcd(depth_im=depth_im, rgb_im=rgb_im, intr=self.cam.intrinsic,
                                          extr=self.cam.params.extrinsic)
                o3d.show(image=depth_im, waitkey=0, dest=False, windowname='imdepth')
                o3d.show(image=rgb_im, waitkey=0, dest=False, windowname='rgb')
                # o3d.show(image=ans['primitive_normals'].numpy(), dest=False, waitkey=0, windowname='normalsmap')
                o3d.NormalViz([points3d] + [self.mesh] + self.cam.show(True) + [
                    self.cam.draw_rays(direction=np.vstack(rays.numpy()[:, :, 3:]),
                                       sources=np.vstack(rays.numpy()[:, :, :3]))] + [o3d.axis_mesh(size=1)])
                o3d.NormalViz([pcd] + [o3d.axis_mesh(size=1)])
                o3d.cv2.destroyAllWindows()

        if save:
            self._export(pose_dict)
        return


if __name__ == '__main__':
    inputdir = os.getcwd() + f'{os.sep}data'
    outdir = os.getcwd() + f'{os.sep}data{os.sep}Arway'
    App = UnProjector(inputdir, outdir, filename='OfficeArway.glb', pose_format='raw')
    # App.show()
    App.run(save=True, show=False)
