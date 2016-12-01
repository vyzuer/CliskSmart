import sys
import time
import os
import glob
import numpy as np
from skimage import io

from sklearn.externals import joblib

import shutil
import _mypath

import preprocess.image_graph as igraph
import preprocess.utils as my_utils
import scene_props.scene_props as sprops
import scene_identification as s_id
import optimize_position as opt_pos

import main_server.global_variables as gv


def predict_position_and_size(scene_id, num_people, gp_model_path):

    """
        Given a scene type and number of people predict the preferable position
    """

    model_dump = gp_model_path + 'position_modeling/' + str(scene_id) + '/models/'
    gmm_model_path = model_dump + 'gmm/gmm.pkl'
    scaler_model_path = model_dump + 'scaler/scaler.pkl'

    gmm_model = joblib.load(gmm_model_path)
    scaler = joblib.load(scaler_model_path)

    x_min = 0
    y_min = 0
    x_max = 1
    y_max = 1
    size_min = 0.000
    size_max = 0.150

    x_dim, y_dim, z_dim = 80, 60, 30
    x, y, z = np.mgrid[x_min:x_max:x_dim*1j, y_min:y_max:y_dim*1j, size_min:size_max:z_dim*1j]
    
    positions = np.vstack([x.ravel(), y.ravel(), num_people*np.ones(x_dim*y_dim*z_dim), z.ravel()])

    positions = scaler.transform(zip(*positions))

    prob_score, response = gmm_model.score_samples(positions)
    prob_score = np.exp(prob_score)

    scores = np.reshape(prob_score, x.shape)

    # find the highest probability position
    idx = scores.argmax()
    pos = np.unravel_index(idx, scores.shape)

    m_position = (1.0*pos[0]/x_dim, 1.0*pos[1]/y_dim)
    m_size = size_max*pos[2]/z_dim

    return m_position, m_size


def gen_recommendation(img_src, num_people, color_data):
    rec_pos = np.zeros(shape=(num_people, 4), dtype=np.int)

    num_people, rec_pos, color_data = pos_rec(img_src, gv.gp_dump_path_, gv.gp_model_path_, visualise=False, server=True)

    return num_people, rec_pos, color_data


def pos_rec(img, img_src, dump_path, gp_model_path, visualise=True, server=False):
    
    # process the image and identify the scene category
    plot_dumps = dump_path + '/plots/'
    if visualise and not os.path.exists(plot_dumps):
        os.makedirs(plot_dumps)

    scene_obj = s_id.scene_info(img, img_src, gp_model_path, visualise=visualise, dump_path=plot_dumps)

    scene_id = scene_obj.predict_scene_id()
    print 'Predicted scene category : ', scene_id[0]

    num_people = scene_obj.get_num_people()
    if num_people < 1:
        print 'No face detected, generating random number...'
        # num_people = np.random.random_integers(2,6)
        num_people = 1

    # num_people = 1

    print 'Number of people: ', num_people

    # predict the recommended position and face size on the frame
    m_position, m_size = predict_position_and_size(scene_id[0], num_people, gp_model_path)

    print 'Recommended Position : ', m_position
    print 'Recommended Size : ', m_size

    # extract salient objects from a scene
    s_objects, p_objects, psuedo_objs = scene_obj.get_salient_objects()

    # people object list to be used in graph
    # it will have different size and position as p_objects
    # also for no people it will create new list
    p_obj_graph, psuedo_objs = scene_obj.get_pobj_for_graph(num_people, m_position, m_size)

    # generate position recommendation
    rec_pos, rec_color = opt_pos.gen_pos_rec(s_objects, p_obj_graph, psuedo_objs, img_src, visualize=visualise, dump_path=plot_dumps)

    return num_people, rec_pos, rec_color


if __name__ == '__main__':

    if len(sys.argv) != 4:
        print "Usage : dataset_path dump_path gp_model_path"
        sys.exit(0)

    dataset_path = sys.argv[1]
    dump_path = sys.argv[2]
    gp_model_path = sys.argv[3]

    pos_rec(dataset_path, dump_path, gp_model_path)


