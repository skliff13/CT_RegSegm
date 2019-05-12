import os
import numpy as np
import nibabel as nb
from skimage import measure, transform
from scipy.ndimage.morphology import binary_dilation

from regsegm_logging import logmess


def imresize(m, newshape, order=1, mode='constant'):
    dtype = m.dtype

    mult = np.max(np.abs(m)) * 2
    m = m.astype(np.float32) / mult
    m = transform.resize(m, newshape, order=order, mode=mode)
    m = m * mult

    return m.astype(dtype=dtype)


def writeMHD(filename, im, datatype):
    with open(filename + '.mhd', 'w') as f:
        f.write('ObjectType = Image\nNDims = %i\nBinaryData = True\n' % len(im.shape))

        s = ' '.join([str(i) for i in im.shape])
        f.write('BinaryDataByteOrderMSB = False\nDimSize = %s\n' % s)

        fn = filename.split('/')[-1] + '.raw'
        f.write('ElementType = MET_%s\nElementDataFile = %s' % (datatype.upper(), fn))

    im = np.swapaxes(im, 0, 2).copy()
    im.tofile(filename + '.raw')


def readMHD_simple(filename, shape):
    arr = np.fromfile(filename + '.raw', dtype=np.uint8)
    arr = np.reshape(arr, shape[::-1])
    im = np.swapaxes(arr, 0, 2).copy()
    return im


def changeTransformParametersFile(oldfile, newfile):
    with open(oldfile, 'rt') as f1:
        lines = f1.readlines()

    toFind = '(FinalBSplineInterpolationOrder'
    with open(newfile, 'wt') as f2:
        for line in lines:
            if toFind in line:
                line = toFind + ' 0)\n'
            f2.write(line)


def register3d(mov, fxd, movmsk, outDir):
    writeMHD(outDir + '/moving', mov, 'char')
    writeMHD(outDir + '/fixed', fxd, 'char')

    cmd = 'elastix -f {0}/fixed.mhd -m {0}/moving.mhd -out {0} -p {0}/parameters_BSpline.txt'.format(outDir)
    logmess(outDir, cmd)
    os.system(cmd)

    changeTransformParametersFile(outDir + '/TransformParameters.0.txt', outDir + '/FinalTransformParameters.txt')

    writeMHD(outDir + '/mask_moving', movmsk, 'char')
    cmd = 'transformix -in {0}/mask_moving.mhd -out {0} -tp {0}/FinalTransformParameters.txt'.format(outDir)
    logmess(outDir, cmd)
    os.system(cmd)

    movedmsk = readMHD_simple(outDir + '/result', mov.shape)

    return movedmsk


def advAnalyzeNiiRead(fn):
    im = nb.load(fn)
    afn = im.affine
    im = im.get_data() + 1024
    im = np.swapaxes(im, 0, 1)
    im = im[:, ::-1, :]
    pxdim = np.abs(np.diag(afn[:3, :3]))

    if pxdim[2] < 1.5:
        im = im[:, :, ::2].copy()
        pxdim[2] *= 2
    elif pxdim[2] > 3:
        newsz = (im.shape[0], im.shape[1], im.shape[2] * 2)
        im = imresize(im, newsz, order=0)
        pxdim[2] /= 2

    return im, pxdim, afn


def catchLungs(im3, pxdim):
    d3 = int(round(2.5 / pxdim[2]))
    sml = im3[::4, ::4, ::d3]
    sbw = sml > 700
    se = np.ones((3, 3, 3), dtype=bool)
    sbw = binary_dilation(sbw, se)

    sbw = np.invert(sbw).astype(int)
    lbl = measure.label(sbw)
    nlbl = np.max(lbl)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = lbl.shape
                l = lbl[i * (s[0] - 1), j * (s[1] - 1), k * (s[2] - 1)]
                lbl[lbl == l] = 0

    for i in range(nlbl):
        if np.sum(lbl == i) < 100:
            lbl[lbl == i] = 0

    lung = lbl[::2, ::2, ::2] > 0
    return lung


def trimProjection(proj):
    cumsum = np.cumsum(proj.astype(np.float) / np.sum(proj))

    d = 3
    i1 = np.where(cumsum > 0.01)[0][0] - d
    i2 = np.where(cumsum < 0.99)[0][-1] + d
    bounds = (max((0, i1)), min((len(proj), i2 + 2)))
    proj = proj[bounds[0]:bounds[1]]

    proj = np.asarray([proj])
    proj = imresize(proj, (1, 100))
    proj = proj.astype(np.float32) / proj.sum()

    return proj, np.asarray(bounds)


def lungproj(lung, pxdim):
    xproj = np.sum(lung, axis=(0, 2)).flatten()
    yproj = np.sum(lung, axis=(1, 2)).flatten()
    zproj = np.sum(lung, axis=(0, 1)).flatten()

    xproj, xb = trimProjection(xproj)
    yproj, yb = trimProjection(yproj)
    zproj, zb = trimProjection(zproj)

    proj = np.append(xproj, (yproj, zproj))
    proj[proj < 0] = 0
    d3 = int(round(2.5 / pxdim[2]))
    mlt = (8, 8, (2 * d3))

    xyzb = np.append(xb * mlt[0], (yb * mlt[1], zb * mlt[2]))

    proj = np.asarray([proj])
    return proj, xyzb


def makeUint8(im):
    im = im * 0.17
    im[im < 0] = 0
    im[im > 255] = 255
    im = im.astype(np.uint8)

    return im
