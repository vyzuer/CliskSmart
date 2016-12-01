import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import numpy as np


def compute(db_path, out_dir, alpha=0.002, beta=0.01, gamma=0.01, clean=True):
    out_file = out_dir + 'aesthetic.scores'

    if not clean and os.path.isfile(out_file):
        print '\nAesthetics score already computed.\nPass clean=True flag for a fresh compilation.\n'
        return
    else:
        try:
            os.remove(out_file)
        except OSError:
            pass

    f_db_details = db_path + 'photo.info'

    data = np.loadtxt(f_db_details, dtype='string')

    num_images, dim = data.shape

    v = np.asarray(map(int, data[:, 5]))
    f = np.asarray(map(int, data[:, 6]))
    c = np.asarray(map(int, data[:, 7]))

    a_score = 1 - np.exp(-(alpha*v + beta*f + gamma*c))

    np.savetxt(out_file, a_score, fmt='%.6f')

    x = a_score
    hist, bins = np.histogram(x, 100)
    plt.axis([0, 1, 0, 600])
    width = 0.8 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.title('Aesthetic Score distribution')
    f_name = out_dir + 'a_score.png'
    plt.savefig(f_name)
    plt.close()


if __name__ == '__main__':
    
    if len(sys.argv) != 3:
        print "usage: ", __file__, "dataset_path dump_path"
        sys.exit(0)

    db_path = sys.argv[1]
    out_dir = sys.argv[2]

    compute(db_path, out_dir)

