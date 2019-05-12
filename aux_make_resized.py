import os
import json
import numpy as np

import regsegm_utils as reg


def main():
    with open('config.json', 'r') as f:
        config = json.load(f)

    dir_initial = '/path/to/original/images'
    dir_segmented = '/path/to/segmented/images'

    out_dir = 'resized_data'

    os.makedirs(out_dir, exist_ok=True)

    rs = config['slice_resize']
    pos_val = config['positive_value']

    for i in range(1, 131):
        img_path = os.path.join(dir_initial, 'id%03i.nii.gz' % i)
        msk_path = os.path.join(dir_segmented, 'id%03i_resegm2.nii.gz' % i)

        print('Reading ' + img_path)
        img = reg.advAnalyzeNiiRead(img_path)[0]
        img = reg.makeUint8(img)
        img = img[::rs, ::rs, :]

        out_path = os.path.join(out_dir, 'id%03i_img.npz' % i)
        print('Saving resized image to ' + out_path)
        np.savez_compressed(out_path, img)

        print('Reading ' + msk_path)
        msk = reg.advAnalyzeNiiRead(msk_path)[0]
        msk = reg.makeUint8(msk)
        msk[msk > 0] = pos_val
        msk[msk == 0] = 0
        msk = msk[::rs, ::rs, :]

        out_path = os.path.join(out_dir, 'id%03i_msk.npz' % i)
        print('Saving resized mask to ' + out_path)
        np.savez_compressed(out_path, msk)


if __name__ == '__main__':
    main()
