import os
import json
import shutil
import pandas as pd
import numpy as np
import nibabel as nb
from scipy.spatial import distance_matrix
import regsegm_utils as reg
from regsegm_logging import logmess


class CtRegSegmentor():
    def __init__(self, resized_data_dir='resized_data',
                 reg_dir='registration_py',
                 reg_files_storage='regFiles',
                 lungs_proj_filename='lungproj_xyzb_130_py.txt'):

        self.resized_data_dir = resized_data_dir
        self.reg_dir = reg_dir
        self.reg_files_storage = reg_files_storage
        self.lungs_proj_filename = lungs_proj_filename

    def process_file(self, file_path):
        os.makedirs(self.reg_dir, exist_ok=True)

        logmess(self.reg_dir)

        self.copy_files_from_storage()

        with open('config.json', 'r') as f:
            config = json.load(f)

        num_nearest = config['num_nearest']
        resz = config['slice_resize']
        pos_val = config['positive_value']

        logmess(self.reg_dir, 'Reading lung projections from ' + self.lungs_proj_filename)
        df = pd.read_csv(self.lungs_proj_filename, header=None)
        data = df.get_values()
        xyz_bounds = data[:, 300:306]
        lung_projs = data[:, 0:300]

        self.log('Reading 3D image from ' + file_path)
        im, voxel_dimensions, affine = reg.adv_analyze_nii_read(file_path)

        self.log('Coarse extraction of lungs')
        lungs = reg.catch_lungs(im, voxel_dimensions)
        self.log('Calculating lung projections')
        projections, bounds = reg.calculate_lung_projections(lungs, voxel_dimensions)

        distances = distance_matrix(projections, lung_projs).flatten()
        idx = np.argsort(distances)
        if distances[idx[0]] < 0.001:
            ids = idx[1:num_nearest + 1] + 1
        else:
            ids = idx[0:num_nearest] + 1

        fixed = reg.make_uint8(im[::resz, ::resz, :]).copy()
        mean_mask = (fixed * 0).astype(np.float32)
        for j in range(num_nearest):
            bounds_moving = xyz_bounds[ids[j] - 1, :]

            path = os.path.join(self.resized_data_dir, 'id%03i_img.npz' % ids[j])
            self.log('Similar image #%i: Reading image from %s' % (j + 1, path))
            data = np.load(path)
            moving = data[data.files[0]]
            moving = self.shift3(moving, fixed.shape, bounds_moving // resz, bounds // resz).astype(np.uint8)

            path = os.path.join(self.resized_data_dir, 'id%03i_msk.npz' % ids[j])
            self.log('Similar image #%i: Reading mask from %s' % (j + 1, path))
            data = np.load(path)
            mask = data[data.files[0]]
            mask = self.shift3(mask, fixed.shape, bounds_moving // resz, bounds // resz).astype(np.uint8)

            self.log('Similar image #%i: Registration' % (j + 1))
            moved_mask = reg.register3d(moving, fixed, mask, self.reg_dir)

            mean_mask += moved_mask.astype(np.float32) / pos_val / num_nearest

        self.log('Resizing procedures')
        mean_mask[mean_mask < 0.5] = 0
        mean_mask[mean_mask >= 0.5] = 1
        mean_mask = mean_mask[:, ::-1, :]
        mean_mask = np.swapaxes(mean_mask, 0, 1)

        mean_mask = reg.imresize(mean_mask, im.shape, order=0)
        mean_mask = mean_mask.astype(np.int16)
        affine = np.abs(affine) * np.eye(4, 4)
        affine[1, 1] = -affine[1, 1]
        affine[0, 0] = -affine[0, 0]
        nii = nb.Nifti1Image(mean_mask, affine)

        fno = file_path[:-7] + '_regsegm_py.nii.gz'
        self.log('Saving result to ' + fno)
        nb.save(nii, fno)

        return 0

    def process_dir(self, dir_path):
        file_ending = '.nii.gz'
        files = os.listdir(dir_path)

        for file in files:
            file_path = os.path.join(dir_path, file)
            out_path = os.path.join(dir_path, file[:-7] + '_regsegm_py.nii.gz')

            if file.endswith(file_ending) and '_regsegm' not in file:
                if os.path.isfile(out_path):
                    print('File ' + out_path + ' already exists. Skipping.')
                else:
                    print('Processing "' + file + '"')
                    self.process_file(file_path)

    def log(self, message=None):
        if message is None:
            logmess(self.reg_dir)
        else:
            logmess(self.reg_dir, message)

    @staticmethod
    def shift3(im, sz, xyzsrc, xyztrg):
        mult = 1.0e-5
        im = im.astype(np.float32) * mult

        for dim in [2, 1, 3]:
            srange = xyzsrc[2 * dim - 2:2 * dim].astype(np.float32)
            trange = xyztrg[2 * dim - 2:2 * dim].astype(np.int)

            resz = (trange[1] - trange[0]) / (srange[1] - srange[0])
            newsz = (int(round(im.shape[0] * resz)), im.shape[1], im.shape[2])
            im1 = reg.imresize(im, newsz, order=0)

            if dim == 3:
                shp = (sz[2], im.shape[1], im.shape[2])
                im = np.zeros(shp)
            else:
                im = im * 0

            n = trange[1] - trange[0]

            strt = int(round(srange[0] * resz))
            if strt - trange[0] + 1 > 0:
                im[0:trange[0], :, :] = im1[strt - trange[0]:strt, :, :]
            else:
                im[trange[0] - strt:trange[0], :, :] = im1[0:strt, :, :]

            im[trange[0]:(trange[0] + n), :, :] = im1[strt:(strt + n), :, :]

            if strt + n + im.shape[0] - trange[1] <= im1.shape[0]:
                im[trange[1]:im.shape[0], :, :] = im1[strt + n:strt + n + im.shape[0] - trange[1], :, :]
            else:
                im[trange[1]:trange[1] + im1.shape[0] - strt - n, :, :] = im1[strt + n:im1.shape[0], :, :]

            im = np.swapaxes(im, 0, 1)
            im = np.swapaxes(im, 1, 2)

        if im.shape[2] > sz[2]:
            im = im[:, :, 0:sz[2]]
        elif im.shape[2] < sz[2]:
            im_ = np.zeros(sz, dtype=im.dtype)
            im_[:, :, 0:im.shape[2]] = im
            im = im_

        im = im / mult
        return im

    def copy_files_from_storage(self):
        logmess(self.reg_dir, 'Copying files from ' + self.reg_files_storage)

        files = os.listdir(self.reg_files_storage + '/')
        for file in files:
            if file.endswith('.txt'):
                src = os.path.join(self.reg_files_storage, file)
                dst = os.path.join(self.reg_dir, file)
                shutil.copyfile(src, dst)
