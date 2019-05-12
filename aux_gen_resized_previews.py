import os
import numpy as np
from skimage import io, transform


def main():
    data_dir = 'resized_data'

    for i in range(1, 131):
        for postfix in ['img', 'msk']:
            path = os.path.join(data_dir, 'id%03i_%s.npz' % (i, postfix))

            data = np.load(path)
            img = data[data.files[0]]

            shape2d = (img.shape[0], img.shape[1])
            im1 = img[:, :, img.shape[2] // 2]
            im1 = transform.resize(im1, shape2d)
            im2 = img[:, img.shape[1] // 2, :]
            im2 = transform.resize(im2, shape2d)
            im3 = img[img.shape[0] // 2, :, :]
            im3 = transform.resize(im3, shape2d)

            im = np.concatenate([im1, np.transpose(im2), np.transpose(im3)], axis=1)
            im = np.flipud(im)

            out_path = path[:-4] + '_preview.png'
            print('Saving preview to ' + out_path)
            io.imsave(out_path, im)


if __name__ == '__main__':
    main()
