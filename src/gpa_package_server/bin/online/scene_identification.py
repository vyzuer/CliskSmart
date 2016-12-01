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


class scene_info:
    def __init__(self, img, img_src, gp_model_path, grid_size=(60,80), b_pca=True, b_scaler=True, visualise=False, dump_path=None, n_iter=20):

        # initialize class memebers
        self.image = img
        self.img_src = img_src
        self.gp_model_path = gp_model_path
        self.grid_size = grid_size
        self.b_pca = b_pca
        self.b_scaler = b_scaler
        self.visualise = visualise
        self.dump_path = dump_path

        self.sp_obj = sprops.scene_props(self.image, self.img_src, grid_size=self.grid_size, visualise=self.visualise, dump_path = self.dump_path, max_iter_slic=n_iter)

    def predict_scene_id(self):

        # features for edges
        e_descriptor, edge_map, s_descriptor, s_map = self.sp_obj.get_scene_features()

        scene_descriptor = np.hstack([e_descriptor[0:244], s_descriptor]).reshape(1,-1)

        sc_model_path = self.gp_model_path + 'scene_categories/cluster_model/'

        km_model_path = sc_model_path + 'cluster.pkl'
        pca_model_path = sc_model_path + 'pca.pkl'
        scaler_model_path = sc_model_path + 'scaler.pkl'

        km_model = joblib.load(km_model_path)
        pca_model = joblib.load(pca_model_path)
        scaler_model = joblib.load(scaler_model_path)

        # preprocess features
        scene_descriptor = scaler_model.transform(scene_descriptor)
        scene_descriptor = pca_model.transform(scene_descriptor)
        
        scene_id = km_model.predict(scene_descriptor)

        return scene_id

    def get_num_people(self):
        return self.sp_obj.num_faces


    def get_salient_objects(self):

        s_objects = self.sp_obj.get_salient_objects()

        return s_objects

    def get_pobj_for_graph(self, num_people, m_position, m_size):

        s_objects = self.sp_obj.get_pobj_for_graph(num_people, m_position, m_size)

        return s_objects

