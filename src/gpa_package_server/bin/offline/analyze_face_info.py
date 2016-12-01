import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

import _mypath
import preprocess.utils as my_utils

def analyze(dataset_path, master_dump_path, clean=True):

    assert os.path.exists(dataset_path)
    assert os.path.exists(master_dump_path)

    dump_path = master_dump_path + 'data_analysis/face/'

    # check if compilation required
    if not clean and my_utils.dataset_valid(dump_path):
        print '\nFace Analysis Database is up-to-date.\nPass clean=True for fresh compilation\n'
        return
    else:
        my_utils.invalidate_dataset(dump_path)

    if not os.path.exists(dump_path):
        os.makedirs(dump_path)

    f_db_details = master_dump_path + 'face_info/type_info.list'

    data = np.loadtxt(f_db_details)

    num_images, dim = data.shape

    out_file = dump_path + 'faces.analysis'
    try:
        os.remove(out_file)
    except OSError:
        pass
    
    fp_out_file = open(out_file, 'a')

    fp_out_file.write('Dataset Size : %d\n' % (num_images))

    tot_faces = np.sum(map(int, data[:, 6]))
    min_faces = np.min(map(int, data[:, 6]))
    max_faces = np.max(map(int, data[:, 6]))
    avg_faces = np.mean(map(int, data[:, 6]))
    fp_out_file.write('Total Faces : %d\tMin Faces : %d\tMax Faces : %d\tMean Faces : %f\n' % (tot_faces, min_faces, max_faces, avg_faces))

    fp_out_file.close()
    
    # plot the histograms for social media data
    x = [int(x) for x in data[:, 6]]
    bins = int(np.max(x) - np.min(x))
    hist = plt.hist(x, bins=bins, color='blue')
    plt.title('Number of Faces Distribution in Photos')

    f_name = dump_path + 'faces_dist.png'
    plt.savefig(f_name)
    plt.close()

    my_utils.validate_dataset(dump_path)

    # plot the position centers
    plt.scatter(data[:,0], 1-data[:,1])
    plt.title('Distribution of position for group centers')
    plt.xticks(())
    plt.yticks(())

    f_name = dump_path + 'faces_position.png'
    plt.savefig(f_name)
    plt.close()


if __name__ == '__main__':
    
    if len(sys.argv) != 3:
        print "usage: ", __file__, "dataset_path dump_path"
        sys.exit(0)

    db_path = sys.argv[1]
    out_dir = sys.argv[2]

    analyze(db_path, out_dir)

