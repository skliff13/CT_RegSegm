import os
import json
import shutil
import traceback
import pandas as pd
import numpy as np
import nibabel as nb
from scipy.spatial import distance_matrix
from scipy.ndimage import gaussian_filter
import regsegm_utils as reg
from regsegm_logging import logmess


class CtRegSegmentor():
    """Creates an instance of CtRegSegmentor class which can be used for segmentation of lung regions in 3D Computed
    Tomography (CT) images of chest.

    # Arguments
        config_file_path: Path to JSON configuration file. Defaults to 'config.json'
        resized_data_dir: Path to directory with the resized image data (id*_img.npz files). Defaults to 'resized_data'
        reg_dir: Path to the directory which will be created for storing the temporary files.
            Defaults to 'registration_py'
        reg_files_storage: Path to directory with files storing the 'elastix' parameters. Defaults to 'regFiles'
        lungs_proj_file_path: Path to the file with projections information. Defaults to 'lungproj_xyzb_130_py.txt'.

    # Examples
    `.process_file('test_data/test_image.nii.gz')`
    `.process_dir('test_data/dir_with_images')`

    """

    def __init__(self, config_file_path='config.json',
                 resized_data_dir='resized_data',
                 reg_dir='registration_py',
                 reg_files_storage='regFiles',
                 lungs_proj_file_path='lungproj_xyzb_130_py.txt'):

        self.config_file_path = config_file_path
        self.resized_data_dir = resized_data_dir
        self.reg_dir = reg_dir
        self.reg_files_storage = reg_files_storage
        self.lungs_proj_file_path = lungs_proj_file_path

    def process_file(self, file_path):
        """Reads a Nifti image, performs segmentation of lungs and saves the result into '*_regsegm.nii.gz' file.

        # Arguments
            file_path: Path to the input file

        """

        os.makedirs(self.reg_dir, exist_ok=True)

        logmess(self.reg_dir)

        self._copy_files_from_storage()

        with open(self.config_file_path, 'r') as f:
            config = json.load(f)

        num_nearest = config['num_nearest']
        resz = config['slice_resize']
        pos_val = config['positive_value']

        logmess(self.reg_dir, 'Reading lung projections from ' + self.lungs_proj_file_path)
        df = pd.read_csv(self.lungs_proj_file_path, header=None)
        data = df.get_values()
        xyz_bounds = data[:, 300:306]
        lung_projs = data[:, 0:300]

        self._log('Reading 3D image from ' + file_path)
        im, voxel_dimensions, affine, shape0 = reg.adv_analyze_nii_read(file_path)

        self._log('Coarse extraction of lungs')
        lungs = reg.catch_lungs(im, voxel_dimensions)
        self._log('Calculating lung projections')
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
            self._log('Similar image #%i: Reading image from %s' % (j + 1, path))
            data = np.load(path)
            moving = data[data.files[0]]
            moving = self._shift3(moving, fixed.shape, bounds_moving // resz, bounds // resz).astype(np.uint8)

            path = os.path.join(self.resized_data_dir, 'id%03i_msk.npz' % ids[j])
            self._log('Similar image #%i: Reading mask from %s' % (j + 1, path))
            data = np.load(path)
            mask = data[data.files[0]]
            mask = self._shift3(mask, fixed.shape, bounds_moving // resz, bounds // resz).astype(np.uint8)

            self._log('Similar image #%i: Registration' % (j + 1))
            moved_mask = reg.register3d(moving, fixed, mask, self.reg_dir)

            mean_mask += moved_mask.astype(np.float32) / pos_val / num_nearest

        self._log('Resizing procedures')
        mean_mask = mean_mask[:, ::-1, :]
        mean_mask = np.swapaxes(mean_mask, 0, 1)

        mean_mask = reg.imresize(mean_mask, shape0, order=1)
        if config['smooth_sigma'] > 0:
            z_sigma = config['smooth_sigma'] * voxel_dimensions[2] / voxel_dimensions[1]
            mean_mask = gaussian_filter(mean_mask, (1, 1, z_sigma))
        mean_mask = np.array(mean_mask > 0.5)
        mean_mask = mean_mask.astype(np.int16)
        affine = np.abs(affine) * np.eye(4, 4)
        affine[1, 1] = -affine[1, 1]
        affine[0, 0] = -affine[0, 0]
        nii = nb.Nifti1Image(mean_mask, affine)

        fno = file_path[:-7] + '_regsegm_py.nii.gz'
        self._log('Saving result to ' + fno)
        nb.save(nii, fno)

        return 0

    def process_dir(self, dir_path):
        """Reads Nifti (*.nii.gz) images from the specified directory, performs segmentation of lungs and saves the results
        into '*_regsegm.nii.gz' files.

            # Arguments
                dir_path: Path to the directory with Nifti images.

        """

        file_ending = '.nii.gz'
        files = os.listdir(dir_path)

        skipped = 0
        processed = 0
        failed = 0
        for file in files:
            file_path = os.path.join(dir_path, file)
            out_path = os.path.join(dir_path, file[:-7] + '_regsegm_py.nii.gz')

            if file.endswith(file_ending) and '_regsegm' not in file:
                if os.path.isfile(out_path):
                    print('File ' + out_path + ' already exists. Skipping.')
                    skipped += 1
                else:
                    print('Processing "' + file + '"')
                    try:
                        self.process_file(file_path)
                        processed += 1
                    except:
                        print('*** FAILED to process files "' + file + '"')
                        failed += 1
                        traceback.print_exc()

        print('Finished processing images in ' + dir_path)
        print('Skipped: %i\nProcessed: %i\nFailed: %i' % (skipped, processed, failed))

    def _log(self, message=None):
        if message is None:
            logmess(self.reg_dir)
        else:
            logmess(self.reg_dir, message)

    @staticmethod
    def _shift3(im, sz, xyzsrc, xyztrg):
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

    def _copy_files_from_storage(self):
        logmess(self.reg_dir, 'Copying files from ' + self.reg_files_storage)

        files = os.listdir(self.reg_files_storage + '/')
        for file in files:
            if file.endswith('.txt'):
                src = os.path.join(self.reg_files_storage, file)
                dst = os.path.join(self.reg_dir, file)
                shutil.copyfile(src, dst)


if __name__ == '__main__':
    print('Testing CtRegSegmentor')
    CtRegSegmentor().process_file('test_data/test_image.nii.gz')
