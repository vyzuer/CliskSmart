import sys, os, time
import numpy as np
import scipy
import matplotlib.pyplot as plt
from skimage import io
import cv2

import _mypath
import preprocess.pixel_map as pmap
import preprocess.face_detection as fdetect

_DEBUG = False

def _plot_pixel_map(pxl_map, fd, img_src):
    img = io.imread(img_src)
    faces, frames = fdetect.detect_face(img, visualise=False)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    plt.subplot(1, 3, 1)
    plt.imshow(pxl_map, interpolation='nearest')

    plt.subplot(1, 3, 2)
    plt.imshow(fd, interpolation='nearest')

    plt.subplot(1, 3, 3)
    plt.imshow(img)

    plt.show()
    plt.close()


def _find_min_max(n_faces, fi_data, j):
    x_min, x_max, y_min, y_max = np.inf, 0.0, np.inf, 0.0

    for k in range(n_faces):
        x_pos, y_pos, x_size, y_size = fi_data[j+k,0], fi_data[j+k,1], fi_data[j+k,2], fi_data[j+k, 3]

        if x_min > x_pos:
            x_min = x_pos
        if x_max < x_pos+x_size:
            x_max = x_pos+x_size

        if y_min > y_pos:
            y_min = y_pos
        if y_max < y_pos+y_size:
            y_max = y_pos+y_size
    
    return x_min, x_max, y_min, y_max


def process(dataset_path, dump_path):

    assert os.path.exists(dataset_path)
    assert os.path.exists(dump_path)

    pmap_dump_path = dump_path + 'formation/'
    pmap_file = pmap_dump_path + 'pixel_map.list'
    if not os.path.exists(pmap_dump_path):
        os.makedirs(pmap_dump_path)

    fp = open(pmap_file, 'w')

    image_dir = dataset_path + "ImageDB/"
    image_list = dataset_path + "photo.info"
    img_list = np.loadtxt(image_list, dtype='string')

    face_dump_path = dump_path + 'face_info/'

    # dump file for faces
    fi_list = face_dump_path + 'face_info.list'
    # face count
    fc_list = face_dump_path + 'face_count.list'

    fi_data = np.loadtxt(fi_list)
    fc_data = np.loadtxt(fc_list)

    n_samples, ndim = fc_data.shape

    j = 0
    # iterate over each image to generate a pixel map
    for i in range(n_samples):
        n_faces = int(fc_data[i,0])
        img_h, img_w = fc_data[i,1], fc_data[i,2]

        # find the min/max x and y position for face frame
        x_min, x_max, y_min, y_max = _find_min_max(n_faces, fi_data, j)

        # process in matrix
        data = fi_data[j:j+n_faces,:]

        # subtract the min positions for normalization
        data -= [x_min, y_min, 0, 0]
        fw = x_max - x_min
        fh = y_max - y_min

        data /= [fw, fh, fw, fh]

        j += n_faces
        
        pxl_map = pmap.get_pixel_map(n_faces, data[:,0:2], data[:,2:4], aspect_ratio=(3,4), map_scale=100)

        # extract feature from this map
        fd = pmap.extract_descriptor(pxl_map, cell_size=(10,10))
       
        np.savetxt(fp, np.atleast_2d(fd), fmt='%.10f')

        if _DEBUG:
            # face detection and plot
            img_name = img_list[i,0]
            img_src = image_dir + img_name

            _plot_pixel_map(pxl_map, fd, img_src)


    fp.close()


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print "Usage : dataset_path dump_path"
        sys.exit(0)

    dataset_path = sys.argv[1]
    dump_path = sys.argv[2]

    process(dataset_path, dump_path)


