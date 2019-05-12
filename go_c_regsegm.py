import os
import json
import shutil
import pandas as pd
import numpy as np
import nibabel as nb
from scipy.spatial import distance_matrix
import regsegm_utils as reg
from regsegm_logging import logmess


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
            im[trange[1]:trange[1]+ im1.shape[0] - strt - n, :, :] = im1[strt + n:im1.shape[0], :, :]

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


def makeUint8(im):
    im = im * 0.17
    im[im < 0] = 0
    im[im > 255] = 255
    im = im.astype(np.uint8)

    return im


def ct_reg_segm(file_path, lungs_proj_filename, resized_data_dir, reg_dir, reg_files_storage):
    os.makedirs(reg_dir, exist_ok=True)

    logmess(reg_dir)

    copy_files_from_storage(reg_dir, reg_files_storage)

    with open('config.json', 'r') as f:
        config = json.load(f)

    nsim = config['num_nearest']
    rs = config['slice_resize']
    posval = config['200']

    logmess(reg_dir, 'Reading lung projections from ' + lungs_proj_filename)
    df = pd.read_csv(lungs_proj_filename, header=None)
    lprojs = df.get_values()
    xyzbs = lprojs[:, 300:306]
    lprojs = lprojs[:, 0:300]

    logmess(reg_dir, 'Reading 3D image from ' + file_path)
    im, pxdim, affine = reg.advAnalyzeNiiRead(file_path)

    logmess(reg_dir, 'Coarse extraction of lungs')
    lng = reg.catchLungs(im, pxdim)
    logmess(reg_dir, 'Calculating lung projections')
    proj, xyzb = reg.lungproj(lng, pxdim)

    ds = distance_matrix(proj, lprojs).flatten()
    idx = np.argsort(ds)
    if ds[idx[0]] < 0.001:
        ids = idx[1:nsim + 1] + 1
    else:
        ids = idx[0:nsim] + 1

    fixed = makeUint8(im[::rs, ::rs, :]).copy()
    mean_mask = (fixed * 0).astype(np.float32)
    for j in range(nsim):
        xyzb1 = xyzbs[ids[j] - 1, :]

        path = os.path.join(resized_data_dir, 'id%03i_img.npz' % ids[j])
        logmess(reg_dir, 'Similar image #%i: Reading image from %s' % (j + 1, path))
        data = np.load(path)
        moving = data[data.files[0]]
        moving = shift3(moving, fixed.shape, xyzb1 // rs, xyzb // rs).astype(np.uint8)

        path = os.path.join(resized_data_dir, 'id%03i_msk.npz' % ids[j])
        logmess(reg_dir, 'Similar image #%i: Reading mask from %s' % (j + 1, path))
        data = np.load(path)
        mask = data[data.files[0]]
        mask = shift3(mask, fixed.shape, xyzb1 // rs, xyzb // rs).astype(np.uint8)

        logmess(reg_dir, 'Similar image #%i: Registration' % (j + 1))
        moved_mask = reg.register3d(moving, fixed, mask, reg_dir)

        mean_mask += moved_mask.astype(np.float32) / posval / nsim

    logmess(reg_dir, 'Resizing procedures')
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
    logmess(reg_dir, 'Saving result to ' + fno)
    nb.save(nii, fno)

    return 0


def copy_files_from_storage(reg_dir, reg_files_storage):
    logmess(reg_dir, 'Copying files from ' + reg_files_storage)
    files = os.listdir(reg_files_storage + '/')
    for file in files:
        if file.endswith('.txt'):
            src = reg_files_storage + '/' + file
            dst = reg_dir + '/' + file
            shutil.copyfile(src, dst)


def process_dir(target_dir):
    resized_data_dir = 'resized_data'
    reg_dir = 'registration_py'
    reg_files_storage = 'regFiles/'

    lungs_proj_filename = 'lungproj_xyzb_130_py.txt'
    file_ending = '.nii.gz'
    files = os.listdir(target_dir)

    for file in files:
        file_path = os.path.join(target_dir, file)
        out_path = os.path.join(target_dir, file[:-7] + '_regsegm_py.nii.gz')
        if file.endswith(file_ending) and (not '_regsegm' in file) and (not os.path.isfile(out_path)):
            print('Processing "' + file + '"')

            ct_reg_segm(file_path, lungs_proj_filename, resized_data_dir, reg_dir, reg_files_storage)

    print('FINISHED')


if __name__ == '__main__':
    dir_with_nii = 'test_images'
    process_dir(dir_with_nii)
