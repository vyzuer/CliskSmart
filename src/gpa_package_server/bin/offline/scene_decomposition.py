import sys
import time
import os
import glob
import numpy as np
from skimage import io

import shutil
import _mypath

import preprocess.image_graph as igraph
import preprocess.utils as my_utils
import scene_props.scene_props as sprops

fp_essd = None # edge scene structure descriptor
fp_sssd = None # saliency scene structure descriptor
fp_emap = None # edge map
fp_smap = None # saliency map

def _open_data_files(dump_path):
    
    global fp_essd, fp_sssd, fp_emap, fp_smap

    # dump file for descriptor
    essd_list = dump_path + 'essd.list'
    fp_essd = open(essd_list, 'w')

    sssd_list = dump_path + 'sssd.list'
    fp_sssd = open(sssd_list, 'w')

    emap_list = dump_path + 'emap.list'
    fp_emap = open(emap_list, 'w')

    smap_list = dump_path + 'smap.list'
    fp_smap = open(smap_list, 'w')


def _close_data_files():
    global fp_essd, fp_sssd, fp_emap, fp_smap

    fp_essd.close()
    fp_sssd.close()
    fp_emap.close()
    fp_smap.close()

def process_scene(img_src, grid_size, dump_path):

    plot_dumps = dump_path + 'plot_maps/'
    if not os.path.exists(plot_dumps):
        os.makedirs(plot_dumps)

    sp_obj = sprops.scene_props(img_src, grid_size=grid_size, visualise=False, dump_path = plot_dumps)

    # features for edges
    e_descriptor, edge_map, s_descriptor, s_map = sp_obj.get_scene_features()

    # dump the features
    np.savetxt(fp_essd, np.atleast_2d(e_descriptor), fmt='%0.12f')
    np.savetxt(fp_emap, np.atleast_2d(edge_map), fmt='%0.12f')
    np.savetxt(fp_sssd, np.atleast_2d(s_descriptor), fmt='%0.12f')
    np.savetxt(fp_smap, np.atleast_2d(s_map), fmt='%0.12f')


def process_dataset(dataset_path, dump_path, clean=False):

    dump_path += 'scene_descriptors/'

    # check if compilation required
    if not clean and my_utils.dataset_valid(dump_path):
        print '\nScene Structure Descriptor Database is up-to-date.\nPass clean=True for fresh compilation\n'
        return
    else:
        my_utils.invalidate_dataset(dump_path)

    if not os.path.exists(dump_path):
        os.makedirs(dump_path)

    # browsing the directory
    image_dir = dataset_path + "ImageDB/"
    
    image_list = dataset_path + "photo.info"
    fp_image_list = open(image_list, 'r')

    grid_size = (60, 80)
    _open_data_files(dump_path)

    for img_details in fp_image_list:
        img_name = img_details.split()[0]
        print img_name

        img_src = image_dir + img_name

        timer = time.time()

        process_scene(img_src, grid_size=grid_size, dump_path=dump_path)
        
        print "Total run time = ", time.time() - timer, '\n'

    fp_image_list.close()

    _close_data_files()

    my_utils.validate_dataset(dump_path)

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print "Usage : dataset_path dump_path"
        sys.exit(0)

    dataset_path = sys.argv[1]
    dump_path = sys.argv[2]

    process_dataset(dataset_path, dump_path, clean=True)


