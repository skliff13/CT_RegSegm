import os
import sys
from ct_reg_segmentor import CtRegSegmentor


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('\nUsage:\tpython3 %s path/to/image/or/directory\n\n' % os.path.split(__file__)[-1])
        exit(1)

    rs = CtRegSegmentor()

    path = sys.argv[1]

    if os.path.isdir(path):
        print('Processing directory ' + path + '\n')

        rs.process_dir(path)
    elif os.path.isfile(path):
        if path.lower().endswith('.nii.gz'):
            print('Processing file ' + path + '\n')

            rs.process_file('test_data/test_image.nii.gz')
        else:
            print('\nOnly .NII.GZ files are supported\n')
