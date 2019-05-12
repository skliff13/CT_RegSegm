
import time
import datetime


def logmess(outDir, str=None):
    logfile = 'ctregsegm_py.log'

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

    if str is None:
        with open(outDir + '/' + logfile, 'w') as f:
            f.write('\n==================================================\n')
        str = 'CTRegSegm v0.1 started'

    str = str.replace('\\', '/')

    with open(outDir + '/' + logfile, 'a') as f:
        f.write(st + ' -- ' + str + '\n')
