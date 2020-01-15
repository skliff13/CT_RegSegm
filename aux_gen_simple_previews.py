import os
import nibabel as nb
import numpy as np
from glob import glob
from skimage import io
from skimage.transform import resize


def gen_frontal_slice_previews(dir_with_nifti):
    ext = '.nii.gz'

    out_dir = os.path.join(dir_with_nifti, 'previews')
    os.makedirs(out_dir, exist_ok=True)

    paths = glob(os.path.join(dir_with_nifti, '*' + ext))
    for path in paths:
        nii = nb.load(path)
        im3 = nii.get_data()

        im = im3[:, 256, :].astype(float)
        im = (im - im.min()) / (im.max() - im.min())
        im = resize(im, (256, 256))
        im = im.T[::-1]

        name = os.path.split(path)[1][:-len(ext)]
        out_path = os.path.join(out_dir, name + '.png')
        print(out_path)
        io.imsave(out_path, (im * 255).astype(np.uint8))


if __name__ == '__main__':
    gen_frontal_slice_previews('path/to/dir/with/nifti')
