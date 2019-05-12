import os
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


def ctregsegm(filename, lunprojFilename, initDir, maskDir, outDir, regFilesStorage):

    if not os.path.isdir(outDir):
        os.mkdir(outDir)

    logmess(outDir)

    logmess(outDir, 'Copying files from ' + regFilesStorage)
    files = os.listdir(regFilesStorage + '/')
    for file in files:
        if file.endswith('.txt'):
            src = regFilesStorage + '/' + file
            dst = outDir + '/' + file
            shutil.copyfile(src, dst)

    nsim = 5
    rs = 4
    # shift = 2

    logmess(outDir, 'Reading lung projections from ' + lunprojFilename)
    df = pd.read_csv(lunprojFilename, header=None)
    lprojs = df.get_values()
    xyzbs = lprojs[:, 300:306]
    lprojs = lprojs[:, 0:300]

    logmess(outDir, 'Reading 3D image from ' + filename)
    im, pxdim, affine = reg.advAnalyzeNiiRead(filename)

    logmess(outDir, 'Coarse extraction of lungs')
    lng = reg.catchLungs(im, pxdim)
    logmess(outDir, 'Calculating lung projections')
    proj, xyzb = reg.lungproj(lng, pxdim)

    ds = distance_matrix(proj, lprojs).flatten()
    idx = np.argsort(ds)
    if ds[idx[0]] < 0.001:
        ids = idx[1:nsim + 1] + 1
    else:
        ids = idx[0:nsim] + 1

    posval = 200
    imr = makeUint8(im[::rs, ::rs, :]).copy()
    meanmsk = (imr * 0).astype(np.float32)
    for j in range(nsim):
        fn = '%sid%03i.nii.gz' % (initDir, ids[j])
        logmess(outDir, 'Similar image #%i: Reading image from %s' % (j + 1, fn))
        mov = reg.advAnalyzeNiiRead(fn)[0]
        mov = makeUint8(shift3(mov, im.shape, xyzbs[ids[j] - 1, :], xyzb))

        fn = '%sid%03i_resegm2.nii.gz' % (maskDir, ids[j])
        logmess(outDir, 'Similar image #%i: Reading mask from %s' % (j + 1, fn))
        msk = reg.advAnalyzeNiiRead(fn)[0]
        msk = makeUint8(shift3(msk, im.shape, xyzbs[ids[j] - 1, :], xyzb))
        msk[msk > 0] = posval
        msk[msk == 0] = 0

        mov = mov[::rs, ::rs, :]
        msk = msk[::rs, ::rs, :]

        logmess(outDir, 'Similar image #%i: Registration' % (j + 1))
        movedmsk = reg.register3d(mov, imr, msk, outDir)

        meanmsk = meanmsk + movedmsk.astype(np.float32) / posval / nsim

    logmess(outDir, 'Resizing procedures')
    meanmsk[meanmsk < 0.5] = 0
    meanmsk[meanmsk >= 0.5] = 1
    # meanmsk = meanmsk[::-1, :, :]
    meanmsk = meanmsk[:, ::-1, :]
    meanmsk = np.swapaxes(meanmsk, 0, 1)

    meanmsk = reg.imresize(meanmsk, im.shape, order=0)
    meanmsk = meanmsk.astype(np.int16)
    affine = np.abs(affine) * np.eye(4, 4)
    affine[1, 1] = -affine[1, 1]
    affine[0, 0] = -affine[0, 0]
    nii = nb.Nifti1Image(meanmsk, affine)

    fno = filename[:-7] + '_regsegm_py.nii.gz'
    logmess(outDir, 'Saving result to ' + fno)
    nb.save(nii, fno)

    return 0


def process_dir():
    dr = r'd:\DATA\ImageCLEF\2019\downloaded\clef2019\TrainingSet\test'

    dr0 = 'd:/DATA/CRDF/NII/initial/'
    drr = 'd:/DATA/CRDF/NII/resegm2/'
    outdir = 'registration_py'
    regFilesStorage = 'regFiles/'

    lunprojfn = 'lungproj_xyzb_130_py.txt'
    fileEnding = '.nii.gz'
    files = os.listdir(dr)

    for fn in files:
        path = dr + '/' + fn
        outpath = dr + '/' + fn[:-7] + '_regsegm_py.nii.gz'
        if fn.endswith(fileEnding) and (not '_regsegm' in fn) and (not os.path.isfile(outpath)):
            print('Processing "' + fn + '"')

            ctregsegm(path, lunprojfn, dr0, drr, outdir, regFilesStorage)

    print('FINISHED')


if __name__ == '__main__':
    process_dir()
