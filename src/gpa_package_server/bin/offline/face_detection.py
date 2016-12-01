import sys
import time
import os
import glob
import numpy as np
from skimage import io

import shutil
import _mypath

import preprocess.face_detection as fdetect
import preprocess.utils as my_utils

def dump_type_features(faces, num_faces, img_h, img_w, fp_type):

    x_min, y_min, x_max, y_max = np.inf, np.inf, 0, 0

    min_size, max_size, avg_size = np.inf, 0, 0
    img_size = img_h*img_w

    # arrays to store the face centers
    face_positions = np.zeros(shape=(num_faces,2))
    face_weights = np.zeros(num_faces)

    for i in range(num_faces):
        x,y,w,h = faces[i]

        if x_min > x:
            x_min = x
        if y_min > y:
            y_min = y
        if x_max < x+w:
            x_max = x+w
        if y_max < y+h:
            y_max = y+h

        f_size = w*h
        if min_size > f_size:
            min_size = f_size
        if max_size < f_size:
            max_size = f_size

        avg_size += f_size

        # store face center
        face_positions[i][0] = x+w/2.0
        face_positions[i][1] = y+h/2.0

        face_weights[i] = w*h

    # find the weighted mean of face positions
    pos_mean = (0,0)
    if num_faces > 0:
        pos_mean = np.dot(face_weights, face_positions)/np.sum(face_weights)

    pos_w = x_max-x_min
    pos_h = y_max-y_min

    size = 1.0*pos_w*pos_h/img_size

    x_pos = (x_min+pos_w/2.0)/img_w
    y_pos = (y_min+pos_h/2.0)/img_h

    shape = 1.0*pos_h/pos_w

    min_size = 1.0*min_size/img_size
    max_size = 1.0*max_size/img_size
    if num_faces > 0:
        avg_size = 1.0*avg_size/(img_size*num_faces)

    np.savetxt(fp_type, np.atleast_2d([pos_mean[0]/img_w, pos_mean[1]/img_h, size, x_pos, y_pos, shape, num_faces, max_size, min_size, avg_size]), fmt='%.10f')

def process_image(img_src, dump_path, fp_cnt, fp_info, fp_type):

    img = io.imread(img_src)

    faces, frames = fdetect.detect_face(img, visualise=False)

    num_faces = len(faces)

    img_h, img_w, img_d = img.shape

    fp_cnt.write('%d %d %d\n' % (num_faces, img_h, img_w))

    for i in faces:
        np.savetxt(fp_info, np.atleast_2d(i), fmt='%d')

    dump_type_features(faces, num_faces, img_h, img_w, fp_type)

def process_dataset(dataset_path, dump_path, clean=True):

    dump_path += 'face_info/'

    # check if compilation required
    if not clean and my_utils.dataset_valid(dump_path):
        print '\nFace Info Database is up-to-date.\nPass clean=True for fresh compilation\n'
        return
    else:
        my_utils.invalidate_dataset(dump_path)

    if not os.path.exists(dump_path):
        os.makedirs(dump_path)

    # browsing the directory
    image_dir = dataset_path + "ImageDB/"
    
    image_list = dataset_path + "photo.info"
    fp_image_list = open(image_list, 'r')

    f_face_count = dump_path + 'face_count.list'
    f_face_info = dump_path + 'face_info.list'
    f_type_info = dump_path + 'type_info.list'

    fp_face_cnt = open(f_face_count, 'w')
    fp_face_info = open(f_face_info, 'w')
    fp_type_info = open(f_type_info, 'w')

    print 'Extracting face features...\n'
    timer = time.time()

    for img_details in fp_image_list:
        img_name = img_details.split()[0]

        img_src = image_dir + img_name

        process_image(img_src, dump_path, fp_face_cnt, fp_face_info, fp_type_info)
        
    print "Total run time = ", time.time() - timer, '\n'

    fp_image_list.close()
    fp_face_cnt.close()
    fp_face_info.close()

    my_utils.validate_dataset(dump_path)


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print "Usage : dataset_path dump_path"
        sys.exit(0)

    dataset_path = sys.argv[1]
    dump_path = sys.argv[2]

    process_dataset(dataset_path, dump_path)


