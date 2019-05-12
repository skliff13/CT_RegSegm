import pandas as pd
import numpy as np
import regsegm_utils as reg


def save_lungproj(filename):
    im, pxdim, affine = reg.advAnalyzeNiiRead(filename)

    lng = reg.catchLungs(im, pxdim)
    proj, xyzb = reg.lungproj(lng, pxdim)

    result = np.append(proj, xyzb)
    result = np.asarray([result])

    fno = filename[:-7] + '_lungproj_py.txt'
    df = pd.DataFrame(result)
    df.to_csv(fno, header=False, index=False)

    return 0


def main():
    dr = '/path/to/NII/initial/'

    for i in range(130):
        fn = 'id%03i.nii.gz' % (i + 1)
        path = dr + '/' + fn

        print('Processing "' + fn + '"')
        save_lungproj(path)

    print('FINISHED')


if __name__ == '__main__':
    main()
