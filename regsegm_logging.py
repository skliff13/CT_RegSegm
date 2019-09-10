import time
import datetime


def logmess(out_dir, str=None):
    logfile = 'ctregsegm_py.log'

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

    if str is None:
        with open(out_dir + '/' + logfile, 'w') as f:
            f.write('\n==================================================\n')
        str = 'CT_RegSegm v1.0 started'

    str = str.replace('\\', '/')

    with open(out_dir + '/' + logfile, 'a') as f:
        print(str)
        f.write(st + ' -- ' + str + '\n')
