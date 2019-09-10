import os
import numpy as np
import nibabel as nb
from skimage import measure, transform
from scipy.ndimage.morphology import binary_dilation

from regsegm_logging import logmess


def imresize(m, new_shape, order=1, mode='constant'):
    data_type = m.dtype

    multiplier = np.max(np.abs(m)) * 2
    m = m.astype(np.float32) / multiplier
    m = transform.resize(m, new_shape, order=order, mode=mode)
    m = m * multiplier

    return m.astype(dtype=data_type)


def write_mhd(filename, im, data_type):
    with open(filename + '.mhd', 'w') as f:
        f.write('ObjectType = Image\nNDims = %i\nBinaryData = True\n' % len(im.shape))

        s = ' '.join([str(i) for i in im.shape])
        f.write('BinaryDataByteOrderMSB = False\nDimSize = %s\n' % s)

        fn = filename.split('/')[-1] + '.raw'
        f.write('ElementType = MET_%s\nElementDataFile = %s' % (data_type.upper(), fn))

    im = np.swapaxes(im, 0, 2).copy()
    im.tofile(filename + '.raw')


def read_mhd_simple(filename, shape):
    arr = np.fromfile(filename + '.raw', dtype=np.uint8)
    arr = np.reshape(arr, shape[::-1])
    im = np.swapaxes(arr, 0, 2).copy()
    return im


def change_transform_parameters_file(oldfile, newfile):
    with open(oldfile, 'rt') as f1:
        lines = f1.readlines()

    toFind = '(FinalBSplineInterpolationOrder'
    with open(newfile, 'wt') as f2:
        for line in lines:
            if toFind in line:
                line = toFind + ' 0)\n'
            f2.write(line)


def register3d(mov, fxd, moving_mask, out_dir):
    write_mhd(out_dir + '/moving', mov, 'char')
    write_mhd(out_dir + '/fixed', fxd, 'char')

    cmd = 'elastix -f {0}/fixed.mhd -m {0}/moving.mhd -out {0} -p {0}/parameters_BSpline.txt'.format(out_dir)
    logmess(out_dir, cmd)
    os.system(cmd)

    change_transform_parameters_file(out_dir + '/TransformParameters.0.txt', out_dir + '/FinalTransformParameters.txt')

    write_mhd(out_dir + '/mask_moving', moving_mask, 'char')
    cmd = 'transformix -in {0}/mask_moving.mhd -out {0} -tp {0}/FinalTransformParameters.txt'.format(out_dir)
    logmess(out_dir, cmd)
    os.system(cmd)

    moved_mask = read_mhd_simple(out_dir + '/result', mov.shape)

    return moved_mask


def adv_analyze_nii_read(fn):
    im = nb.load(fn)
    afn = im.affine
    im = im.get_data() + 1024
    im = np.swapaxes(im, 0, 1)
    im = im[:, ::-1, :]
    voxel_dimensions = np.abs(np.diag(afn[:3, :3]))

    if voxel_dimensions[2] < 1.5:
        im = im[:, :, ::2].copy()
        voxel_dimensions[2] *= 2
    elif voxel_dimensions[2] > 3:
        new_size = (im.shape[0], im.shape[1], im.shape[2] * 2)
        im = imresize(im, new_size, order=0)
        voxel_dimensions[2] /= 2

    return im, voxel_dimensions, afn


def catch_lungs(im3, voxel_dimensions):
    d3 = int(round(2.5 / voxel_dimensions[2]))
    sml = im3[::4, ::4, ::d3]
    sbw = sml > 700
    se = np.ones((3, 3, 3), dtype=bool)
    sbw = binary_dilation(sbw, se)

    sbw = np.invert(sbw).astype(int)
    lbl = measure.label(sbw)
    num_labels = np.max(lbl)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = lbl.shape
                l = lbl[i * (s[0] - 1), j * (s[1] - 1), k * (s[2] - 1)]
                lbl[lbl == l] = 0

    for i in range(num_labels):
        if np.sum(lbl == i) < 100:
            lbl[lbl == i] = 0

    lung = lbl[::2, ::2, ::2] > 0
    return lung


def trim_projection(projection):
    cumsum = np.cumsum(projection.astype(np.float) / np.sum(projection))

    d = 3
    i1 = np.where(cumsum > 0.01)[0][0] - d
    i2 = np.where(cumsum < 0.99)[0][-1] + d
    bounds = (max((0, i1)), min((len(projection), i2 + 2)))
    projection = projection[bounds[0]:bounds[1]]

    projection = np.asarray([projection])
    projection = imresize(projection, (1, 100))
    projection = projection.astype(np.float32) / projection.sum()

    return projection, np.asarray(bounds)


def calculate_lung_projections(lung, voxel_dimensions):
    x_proj = np.sum(lung, axis=(0, 2)).flatten()
    y_proj = np.sum(lung, axis=(1, 2)).flatten()
    z_proj = np.sum(lung, axis=(0, 1)).flatten()

    x_proj, xb = trim_projection(x_proj)
    y_proj, yb = trim_projection(y_proj)
    z_proj, zb = trim_projection(z_proj)

    projections = np.append(x_proj, (y_proj, z_proj))
    projections[projections < 0] = 0
    d3 = int(round(2.5 / voxel_dimensions[2]))
    mlt = (8, 8, (2 * d3))

    xyz_bounds = np.append(xb * mlt[0], (yb * mlt[1], zb * mlt[2]))

    projections = np.asarray([projections])
    return projections, xyz_bounds


def make_uint8(im):
    im = im * 0.17
    im[im < 0] = 0
    im[im > 255] = 255
    im = im.astype(np.uint8)

    return im
