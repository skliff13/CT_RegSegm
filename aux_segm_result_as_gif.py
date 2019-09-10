import imageio
import numpy as np
from skimage import transform

import regsegm_utils as reg
from ct_reg_segmentor import CtRegSegmentor


if __name__ == '__main__':
    rs = CtRegSegmentor()

    rs.process_dir('test_data/dir_with_images')

    img_path = 'test_data/dir_with_images/image1.nii.gz'
    out_path = 'result1.gif'
    msk_path = img_path.replace('.nii.gz', '_regsegm_py.nii.gz')

    img = reg.adv_analyze_nii_read(img_path)[0]
    img[img < 0] = 0
    msk = reg.adv_analyze_nii_read(msk_path)[0]

    ground = msk.min()

    images = []
    for k in range(10, img.shape[2] - 10, 2):
        if k % 10 == 0:
            print('%i / %i' % (k, img.shape[2]))

        slice = img[:, :, k:k + 1] / 1500

        mask_slice = msk[:, :, k:k + 1] == ground

        shadowed = slice * mask_slice

        rgb = np.concatenate((slice, slice, shadowed), axis=2)
        rgb = transform.resize(rgb, (256, 256))
        rgb = np.flipud(rgb)

        rgb += 0.1
        rgb[rgb > 1] = 1

        images.append((rgb * 255).astype(np.uint8))

    print('Writing gif to ' + out_path)
    imageio.mimsave(out_path, images, duration=1./20)
