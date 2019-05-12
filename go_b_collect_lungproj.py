import pandas as pd
import numpy as np


def main():
    dr = '/path/to/original/images/'

    tbl = None
    for i in range(130):
        fn = 'id%03i_lungproj_py.txt' % (i + 1)
        path = dr + '/' + fn

        print('Reading "' + fn + '"')
        row = pd.read_csv(path, header=None).get_values()

        if tbl is None:
            tbl = row
        else:
            tbl = np.append(tbl, row, axis=0)

    fno = 'lungproj_xyzb_130_py.txt'
    print('Saving table to "' + fno + '"')
    df = pd.DataFrame(tbl)
    df.to_csv(fno, header=False, index=False)

    print('FINISHED')


if __name__ == '__main__':
    main()
