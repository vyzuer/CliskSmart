from skimage import io
from skimage.segmentation import mark_boundaries
from skimage.segmentation import relabel_sequential
from skimage import img_as_float
from skimage.color import rgb2gray, rgb2hsv, rgb2lab
from matplotlib import pyplot as plt
import numpy as np
import string
from scipy import ndimage
import os
import time
import cv2
from skimage.feature import hog
from skimage import measure
from sklearn.externals import joblib
from skimage import transform as tf
from skimage import color

import _mypath

import img_proc.segmentation as seg
import img_proc.saliency as saliency
import img_proc.cython_utils as cutils
import preprocess.image_graph as img_graph
import preprocess.face_detection as fdetect

_MIN_OBJ_SOLIDITY = 0.050 # atleast 20%
_NI_OBJS = 50 # maximum number of salient objects to consider
_NUM_COLORS = 5 # maximum number of colors to consider for clothing

# the coolest hue is considered at 225 degree which is equivalent to 0.625 when scaled between 0-1
_COOLEST_HUE = 0.5
# _COOLEST_HUE = 0.625

_CE_WEIGHTS = [5.0,8.0,5.0,2.0,2.0]  # hue, saturation, value, size, contrast

class scene_object:
    def __init__(self, id, size=0.1, color=(0.5,0.5,0.5), position=(0.5,0.5), c_energy=0.0, saliency=0.0, gid=-1, psuedo=0):
        self.id = id
        self.size = size
        self.color = color
        self.pos = position
        self.color_energy = c_energy
        self.saliency = saliency
        self.gid = gid
        self.psuedo = psuedo


class scene_props:
    def __init__(self, image, image_src='/tmp/temp/sample.jpeg', min_saliency=0.000, max_iter_slic=200, offline=True, grid_size=(60,80), visualise=False, dump_path=None):
        self.timer = time.time()
        self.image_src = image_src
        self.min_saliency = 0.0000
        self.grid_size = grid_size
        self.offline_mode = offline
        self.visualise = visualise
        self.dump_path = dump_path

        self.image = image
        print "Image source : ", self.image_src

        self.__set_timer("segmentation...")
        self.segment_object = seg.SuperPixelSegmentation(self.image, max_iter=max_iter_slic)
        self.slic_map = self.segment_object.getSlicSegmentationMap()
        # self.mean_color = self.segment_object.get_slic_segments_color()
        self.__print_timer("segmentation")

        self.__set_timer("saliency...")
        saliency_object = saliency.Saliency(self.image, 3)
        self.saliency_map = saliency_object.getSaliencyMap()
        self.__print_timer("saliency")

        self.__set_timer("face detection...")
        padding = self.image.shape[1]/self.grid_size[1]
        self.faces, self.frames = fdetect.detect_face(self.image, padding=padding, visualise=False)
        self.num_faces = len(self.faces)
        self.__print_timer("face detection")

        self.segment_map = self.segment_object.getSegmentationMap()
        self.mean_color = self.segment_object.get_segments_color()

        self.img_h, self.img_w = self.segment_map.shape
        self.img_size = self.img_h*self.img_w


    def get_random_color(self, i):
        rgb_color = np.random.rand(3)
        if i==0:
            rgb_color = np.array([0.0,0.4,0.65098])  # blue
            rgb_color = np.array([0.6627451,0.08627451,0.2])  # off white
            rgb_color = np.array([0.62745,0.05882,0.08627])  # red
            rgb_color = np.array([0.0,0.52156,0.262745]) # dark green
            rgb_color = np.array([0.73333333,  0.03137255,  0.1058823]) # bright red
            rgb_color = np.array([0.6627451,0.08627451,0.2])  # red
            rgb_color = np.array([0.87843137,  0.83529412,  0.7254902]) # white shade
            rgb_color = np.array([0.0,0.52156,0.262745]) # dark green
            rgb_color = np.array([0.86274,0.86274,0.854901])  # off white
            rgb_color = np.array([0.0,  0.0,  0.98823529]) # blue
            rgb_color = np.array([0.0,0.0,0.0]) # black
            rgb_color = np.array([0.0,0.52156,0.262745]) # dark green
            rgb_color = np.array([0.62745,0.05882,0.08627])  # red
        elif i == 1:
            # rgb_color = np.array([0.57647,0.71764,0.43529])  # light green
            rgb_color = np.array([0.62745,0.05882,0.08627])  # red
            rgb_color = np.array([0.00784314,  0.67843137,  0.78823529]) # blue
            rgb_color = np.array([0.11372549,0.188235294,0.10980392]) # very dark green
            rgb_color = np.array([0.64705882,  0.45882353,  0.27058824]) # light brown
            rgb_color = np.array([0.86274,0.86274,0.854901])  # off white
            rgb_color = np.array([0.0,0.0,0.0]) # black
            rgb_color = np.array([0.0,0.52156,0.262745]) # dark green
        elif i==2:
            rgb_color = np.array([0.0,0.0,0.0]) # black
            rgb_color = np.array([0.403921,0.22352941,  0.16078431]) # brown
            rgb_color = np.array([0.0,0.0,0.0]) # black
            rgb_color = np.array([0.73333333,  0.03137255,  0.1058823]) # bright red
            rgb_color = np.array([0.0,0.52156,0.262745]) # dark green
            rgb_color = np.array([0.85882353,  0.24705882,  0.50196078]) # pink
            rgb_color = np.array([0.403921,0.22352941,  0.16078431]) # brown
            rgb_color = np.array([0.62745,0.05882,0.08627])  # red
            rgb_color = np.array([0.403921,0.22352941,  0.16078431]) # brown
        elif i==3:
            # rgb_color = np.array([0.96862,0.78431,0.03137]) # yellow
            rgb_color = np.array([0.0,0.0,0.0]) # black
            rgb_color = np.array([0.0,0.52156,0.262745]) # dark green
            rgb_color = np.array([0.73333333,  0.03137255,  0.1058823]) # bright red
            rgb_color = np.array([0.0,0.52156,0.262745]) # dark green
            rgb_color = np.array([0.62745,0.05882,0.08627])  # red
            rgb_color = np.array([0.0,0.0,0.0]) # black
        elif i==4:
            # rgb_color = np.array([0.96862,0.78431,0.03137]) # yellow
            rgb_color = np.array([0.0,0.52156,0.262745]) # dark green
            rgb_color = np.array([0.0,0.52156,0.262745]) # dark green
            rgb_color = np.array([0.0,0.0,0.0]) # black
            rgb_color = np.array([0.96862,0.78431,0.03137]) # yellow
        else:
            # rgb_color = np.array([0.96862,0.78431,0.03137]) # yellow
            rgb_color = np.array([0.73333333,  0.03137255,  0.1058823]) # bright red
            rgb_color = np.array([0.0,0.52156,0.262745]) # dark green
            rgb_color = np.array([0.86274,0.86274,0.854901])  # off white

        hsv_color = color.rgb2hsv(np.reshape(rgb_color, (1,1,3)))

        return hsv_color[0,0], rgb_color

    def get_pobj_for_graph(self, num_people, m_position, m_size):
        """
            This function creates a new list of people objects to
            be used in the graph model. For case where no people are 
            present this will create a new dummy list to be used
        """

        p_obj_fg = []
        psuedo_obj = []

        if self.p_objs.size:
            ratio = 4.*m_size/self.p_segment_size.mean()
            mean_pos = self.p_segment_pos.mean(axis=0)
            x_disp = m_position[1] + 3.*np.sqrt(m_size)- mean_pos[0]
            y_disp = m_position[0] - mean_pos[1]
            
            for obj in self.p_objs:
                # update the size of segment
                obj.size *= ratio
                # update the position
                obj.pos += [x_disp, y_disp]

            for obj in self.psuedo_objs:
                # update the size of segment
                obj.size *= ratio
                # update the position
                obj.pos += [x_disp, y_disp]

            p_obj_fg = self.p_objs
            psuedo_obj = self.psuedo_objs
        else:
            # iterate over the number of faces and create 
            # a graph node for each
            _id = self.num_segments
            for i in range(num_people):
                size = 4*m_size
                pos = (m_position[1] + 3.*np.sqrt(m_size), m_position[0])
                saliency = 1.0

                hsv_color, rgb_color = self.get_random_color(i)

                # compute hue energy
                hue_energy = self.__compute_hue_energy(hsv_color[0])

                saturation_energy = hsv_color[1]
                brightness_energy = hsv_color[2]

                # compute relative size of the object
                size_energy = size

                # compute contrast of the salient object
                contrast_energy = 0.3

                energy_ = [hue_energy, saturation_energy, brightness_energy, size_energy, contrast_energy]
                color_energy = np.dot(energy_, _CE_WEIGHTS)/np.sum(_CE_WEIGHTS)

                g_obj = scene_object(id=_id, \
                                size=size, \
                                color=rgb_color, \
                                position=pos, \
                                c_energy=color_energy, \
                                saliency=saliency, \
                                gid=i)

                p_obj_fg.append(g_obj)

                i_obj = scene_object(id=100+i, \
                                    size=size, \
                                    position=pos, \
                                    color=rgb_color, \
                                    psuedo=1, \
                                    gid=i)
                psuedo_obj.append(i_obj)
                _id += 1
                

        return np.asarray(p_obj_fg), np.asarray(psuedo_obj)


    def get_salient_objects(self):
        """ this will be called in the online phase. find all the salient objects in the image 
            and also compute their color energy

        """
        self.__set_timer("saliency detection of objects...")
        self.saliency_list, self.salient_objects, self.pixel_count, self.segment_map2 = cutils.detect_saliency_of_segments(self.segment_map.astype(np.intp), self.saliency_map, self.min_saliency)
        self.num_segments = len(self.salient_objects)
        self.__print_timer("saliency detection of objects")

        # compute color energy for all the salient objects
        self.__set_timer("color energy...")
        self.color_energy, self.obj_size_list, \
            self.obj_pos_list, self.obj_color_list = self.__find_obj_props()
        self.__print_timer("color energy")

        # find important objects in the scene
        self.imp_objs, self.p_objs, self.psuedo_objs = self.find_imp_objects()

        if self.visualise:
            # just for testing
            self.color_energy_map, self.graph_nodes = self.__compute_ce_map()
            self.graph_image = self.construct_image()

            self.plot_energy_map(self.dump_path)

        return self.imp_objs, self.p_objs, self.psuedo_objs

    def _mark_ovelapping_segments(self):
        seg_overlap = []

        # image dimensions
        img_h, img_w = self.segment_map2.shape

        p_segment = np.zeros(self.num_segments, dtype=int)
        self.p_segment_size = np.zeros(self.num_faces)
        self.p_segment_pos = np.zeros(shape=(self.num_faces,2))
        self.p_segment_gid = np.zeros(self.num_segments, dtype=int)
        self.p_segment_colors = np.zeros(shape=(self.num_faces,3))

        # frames have different x and y axis
        for x0,y0,x1,y1 in self.frames:
            block = self.segment_map2[y0:y1, x0:x1]

            seg_overlap.extend(np.unique(block))

        i = 0
        avg_face_size = 0.0
        mean_pos = [0.5, 0.5]
        for x,y,w,h in self.faces:

            # for each person find the top 5 segments based on size
            head_size = w*h

            x0 = y + h + h/2
            x1 = x0 + 5*h
            y0 = x - w/2
            y1 = y0 + 2*w

            # check the bounds
            if x0 > img_h:
                x0 = img_h-1
            if y0 < 0:
                y0 = 0
            if x1 > img_h:
                x1 = img_h
            if y1 > img_w:
                y1 = img_w

            # x01 = y + h
            # x11 = x01 + 8*h
            # y01 = x - w
            # y11 = y01 + 3*w

            # # check the bounds
            # if x01 > img_h:
            #     x01 = img_h-1
            # if y01 < 0:
            #     y01 = 0
            # if x11 > img_h:
            #     x11 = img_h
            # if y11 > img_w:
            #     y11 = img_w
            up_body = self.segment_map2[x0:x1, y0:y1]
            # outer_frame = self.segment_map2[x01:x11, y01:y11]

            counts = np.bincount(np.asarray(up_body).reshape(-1))
            # o_counts = np.bincount(np.asarray(outer_frame).reshape(-1))
            # ignoring segment id 0 and adding 1 to adjust that
            seg_ids = np.argsort(counts[1:])[::-1] + 1

            n_colors = min(_NUM_COLORS, seg_ids.size)

            # iterate through all the colors and mark them 
            for seg_id in seg_ids[:n_colors]:
                if counts[seg_id] > 0.5*self.pixel_count[seg_id]:
                    p_segment[seg_id] = 1
                    self.p_segment_gid[seg_id] = i
                    if p_segment[seg_ids[0]] == 0:
                        seg_ids[0] = seg_id
                elif n_colors < seg_ids.size and counts[seg_id] > 0:
                    n_colors += 1
                    

            # store the size and position for later use
            self.p_segment_size[i] = 4.0*head_size/self.img_size
            self.p_segment_pos[i] = 1.0*(y+3*h)/self.img_h, 1.0*(x+w/2)/self.img_w
            _id = seg_ids[0]
            self.p_segment_colors[i] = self.segment_object.get_rgb_mean_color(_id)

            i += 1

        seg_overlap = np.unique(seg_overlap)
    
        n_ol_segs = len(seg_overlap)

        s_overlap = np.zeros(self.num_segments, dtype=int)

        for i in range(n_ol_segs):
            if not p_segment[seg_overlap[i]]:
                s_overlap[seg_overlap[i]] = 1

        return s_overlap, p_segment


    def find_imp_objects(self):

        # in case of faces present we want to preserve all of them as well
        n_objs = np.min([self.num_segments-self.num_faces, _NI_OBJS])
        NI_OBJS = n_objs + self.num_faces

        # list of important objects
        imp_objs = []
        p_objs = []
        psuedo_objs = []

        # use weighted size, saliency and color energy for imp detection
        l_color, l_saliency, l_size = 0.1, 0.5, 0.5 # equal wightage to size, saliency and color 0.5, 0.5, 0.5

        self.imp_list = l_color*self.color_energy/max(self.color_energy) + \
                        l_saliency*self.saliency_list/max(self.saliency_list) + \
                        l_size*self.obj_size_list/max(self.obj_size_list)

        imp_obj_idx = self.imp_list.argsort()[::-1][:NI_OBJS]
        # print imp_obj_idx
        # print self.imp_list[imp_obj_idx]

        for i in range(NI_OBJS):
            _id = imp_obj_idx[i]
            # break if saliency of objects drop to zero
            # if self.saliency_list[_id] == 0.0 and not self.p_segments[_id]:
            #     break

            i_obj = scene_object(id=_id, \
                                size=self.obj_size_list[_id], \
                                color=self.obj_color_list[_id], \
                                position=self.obj_pos_list[_id], \
                                c_energy=self.color_energy[_id], \
                                saliency=self.saliency_list[_id], \
                                gid=self.p_segment_gid[_id])

            if self.p_segments[_id]:
                p_objs.append(i_obj)
            else:
                imp_objs.append(i_obj)

        self.min_imp = self.imp_list[imp_obj_idx[NI_OBJS-1]]

        # create psuedo objects for later use as people
        for i in range(self.num_faces):
            i_obj = scene_object(id=100+i, \
                                size=self.p_segment_size[i], \
                                position=self.p_segment_pos[i], \
                                color=self.p_segment_colors[i], \
                                psuedo=1, \
                                gid=i)
            psuedo_objs.append(i_obj)

        return np.asarray(imp_objs), np.asarray(p_objs), np.asarray(psuedo_objs)


    def get_scene_features(self):
        """ compute the edge and saliency features for the scene
        """

        # get feature set for edges
        edge_descriptor, edge_map = img_graph.image_graph(self.segment_map, self.mean_color, self.frames, grid_size=self.grid_size)

        # get feature set for saliency
        sal_descriptor, ds_sal_map = img_graph.get_saliency_based_descriptor(self.saliency_map, self.frames, grid_size=self.grid_size)

        if self.visualise:
            self.edge_map = img_graph.visualize_descriptor(edge_map)
            self.sal_map = img_graph.visualize_s_descriptor(ds_sal_map)

            self.plot_maps(self.dump_path)
        
        return edge_descriptor, edge_map, sal_descriptor, ds_sal_map


    def construct_image(self):
        img = np.zeros(shape=(self.img_h, self.img_w, 3))

        for h in range(self.img_h):
            for w in range(self.img_w):
                _id = self.segment_map2[h,w]
                img[h,w,:] = self.obj_color_list[_id]

        return img

    def __compute_ce_map(self):
        ce_map = np.zeros(shape=self.segment_map2.shape)
        gn_map = np.zeros(shape=self.segment_map2.shape)

        for i in range(self.num_segments):
            mask = self.segment_map2 == i
            ce_map[mask] = self.color_energy[i]
            if self.imp_list[i] >= self.min_imp:
                gn_map[mask] = self.imp_list[i]

        return ce_map, gn_map
            

    def __compute_hue_energy(self, hue):
        warmness = abs(hue - _COOLEST_HUE)

        # to find mimimum circular distance on the hue wheel
        if warmness > 0.5:
            warmness = 1.0 - warmness

        return 2*warmness

    def __find_contrast_energy(self, hsv, _id, adj_matrix):
        contrast_energy = 0.0

        num_segments = adj_matrix.shape[0]

        # iterate through the neighbors for calculating mean hsv
        hsv_mean = [0.0,0.0,0.0]
        num_negh = 0
        # ignore the background
        for i in range(1,num_segments):
            if adj_matrix[_id, i] == 1:
                hsv_mean += self.segment_object.get_mean_color(i)
                num_negh += 1

        if num_negh > 0:
            hsv_mean = hsv_mean/num_negh

        # use michelson formula for contrast estimation
        l_contrast = abs(hsv[2] - hsv_mean[2])/(hsv[2] + hsv_mean[2])
        h_contrast = abs(hsv[0] - hsv_mean[0])/(hsv[0] + hsv_mean[0])

        contrast_energy = np.mean([l_contrast, h_contrast])

        return contrast_energy

    def __find_obj_props(self):
        # wieghts [hue, saturation, brightness, size, contrast]

        self.__set_timer("adjacency matrix...")
        self.adj_matrix = cutils.compute_adj_matrix(self.segment_map2)
        self.__print_timer("adjacency matrix")
        
        self.__set_timer("region_props...")
        props = measure.regionprops(self.segment_map2)
        self.__print_timer("regionprops")
        
        # find segments which are overlapping with people
        segs_overlap, self.p_segments = self._mark_ovelapping_segments()

        obj_size_list = np.zeros(self.num_segments)
        obj_pos_list = np.zeros(shape=(self.num_segments, 2))
        obj_color_list = np.zeros(shape=(self.num_segments, 3))

        color_energy = np.zeros(self.num_segments)

        for obj in props:
            _id = obj.label
            # print _id

            # remove segments with small size and big frame size, must be borders in the image
            if (obj.solidity < _MIN_OBJ_SOLIDITY or segs_overlap[_id]) and not self.p_segments[_id]:
                self.salient_objects[_id] = 0
                self.saliency_list[_id] = 0
                continue

            hsv_color = self.segment_object.get_mean_color(_id)

            # compute hue energy
            hue_energy = self.__compute_hue_energy(hsv_color[0])

            saturation_energy = hsv_color[1]
            brightness_energy = hsv_color[2]

            # compute relative size of the object
            size_energy = 1.*obj.area/self.img_size

            # compute contrast of the salient object
            contrast_energy = self.__find_contrast_energy(hsv_color, _id, self.adj_matrix)

            energy_ = [hue_energy, saturation_energy, brightness_energy, _NI_OBJS*size_energy, contrast_energy]
            color_energy[_id] = np.dot(energy_, _CE_WEIGHTS)/np.sum(_CE_WEIGHTS)

            obj_color_list[_id] = self.segment_object.get_rgb_mean_color(_id)

            # add color information to psuedo nodes

            # increase the saliency of people objetcs to keep them at the top of list
            if self.p_segments[_id]:
                self.saliency_list[_id] = 1.0

            obj_size_list[_id] = size_energy
            obj_pos_list[_id] = obj.centroid[0]/self.img_h, obj.centroid[1]/self.img_w
            
            if 0:
                min_row, min_col, max_row, max_col = obj.bbox

                segment_img = self.image[min_row:max_row, min_col:max_col,:]
                segment_map = self.segment_map2[min_row:max_row, min_col:max_col]

                segment_copy = np.copy(segment_img)
                
                mask = segment_map != _id

                segment_copy[mask,:] = [255,255,255]

                # testing
                plt.imshow(segment_copy)
                plt.show()
                plt.close('all')

        # max_ce = color_energy.max()

        # if max_ce > 0:
        #     color_energy /= max_ce

        return color_energy, obj_size_list, obj_pos_list, obj_color_list

    def num_of_faces(self):
        return len(self.faces)

    def classify_objects(self, dump_path, vp_model_path, seg_dump=False):

        img_x, img_y = self.segment_map2.shape
        segs = ndimage.find_objects(self.segment_map2)
        # ignore the last segment as it is for faces
        num_segments = len(segs) - 1

        # 0 is for backgroud
        # last segment is for faces
        # num_segments = len(self.saliency_list) - 1

        dir_name = os.path.split(self.image_src)[1]
        dir_name = os.path.splitext(dir_name)[0]

        dir_path = dump_path + "/segments/"

        seg_path = dir_path + dir_name
        if not os.path.exists(seg_path):
            os.makedirs(seg_path)

        model_dump = vp_model_path + "/cluster_model/cluster.pkl"
        cluster_model = joblib.load(model_dump)

        lm_objects_list = []
        saliency_list = []

        j = 1
        for i in xrange(num_segments):

            # for deleted segments
            if segs[i] == None:
                continue

            segment_img = self.image[segs[i]]

            segment_copy = np.copy(segment_img)

            mask = self.segment_map2[segs[i]]
            idx=(mask!=i+1)
            segment_copy[idx] = 255, 255, 255
            
            if (seg_dump == True):
                fig, ax = plt.subplots(1, 2)

                ax[0].axis('off')
                ax[1].axis('off')
                fig.patch.set_visible(False)

                ax[0].imshow(segment_img)
                ax[1].imshow(segment_copy)

                file_name = seg_path + '/' + str(j) + ".png"
                plt.savefig(file_name, dpi=60)
                plt.close('all')
            
            obj_id = self.__predict_lm_object(segment_copy, mask, i+1, cluster_model)

            lm_objects_list.append(obj_id)
            saliency_list.append(self.saliency_list[i+1])

            j += 1

        return np.asarray(lm_objects_list), np.asarray(saliency_list)

    def __predict_lm_object(self, img_segment, seg_block, idx, cluster_model):
        fv = self.__find_segment_features(img_segment, seg_block, idx)

        obj_id = cluster_model.predict(fv.reshape(1,-1))

        return obj_id


    def process_segments(self, dump_path, master_dump=None, seg_dump=False):
        global dump_segs
        if master_dump == None:
            dump_segs = False

        img_x, img_y = self.segment_map2.shape
        segs = ndimage.find_objects(self.segment_map2)
        # ignore the last segment as it is for faces
        num_segments = len(segs) - 1

        # 0 is for backgroud
        # last segment is for faces
        # num_segments = len(self.saliency_list) - 1

        dir_name = os.path.split(self.image_src)[1]
        dir_name = os.path.splitext(dir_name)[0]

        dir_path = dump_path + dir_name
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        seg_path = dir_path + "/segments/"   
        if not os.path.exists(seg_path):
            os.makedirs(seg_path)

        feature_file = dir_path + "/feature.list"
        saliency_file = dir_path + "/saliency.list"
        pos_file = dir_path + "/pos.list"

        if os.path.isfile(feature_file):
            os.unlink(feature_file)
        if os.path.isfile(saliency_file):
            os.unlink(saliency_file)
        if os.path.isfile(pos_file):
            os.unlink(pos_file)

        fp = open(saliency_file, 'w')
        fp1 = open(feature_file, 'w')
        fp2 = open(pos_file, 'w')

        # master dump
        f_segment_features = None
        f_segment_images = None

        fp3 = None
        fp4 = None

        if dump_segs == True:
            f_segment_features = master_dump + '/segments.list'
            f_segment_images = master_dump + '/png.list'

            fp3 = open(f_segment_features, 'a')
            fp4 = open(f_segment_images, 'a')

        j = 1
        for i in xrange(num_segments):

            # for deleted segments
            if segs[i] == None:
                continue

            fp.write("%0.8f\n" % self.saliency_list[i+1])

            segment_img = self.image[segs[i]]

            # find the position
            x_0 =  segs[i][0].start
            x_1 =  segs[i][0].stop
            y_0 = segs[i][1].start
            y_1 = segs[i][1].stop
            x_pos = (x_1 + x_0)/2.0
            y_pos = (y_1 + y_0)/2.0
            # print segs[i]
            # print '{0} {1}'.format(x_pos, y_pos)

            fp2.write("{0:0.8f} {1:0.8f}\n".format(x_pos/img_x, y_pos/img_y))

            segment_copy = np.copy(segment_img)

            mask = self.segment_map2[segs[i]]
            idx=(mask!=i+1)
            segment_copy[idx] = 255, 255, 255
            
            file_name = seg_path + str(j) + ".png"
            if (seg_dump == True):
                fig, ax = plt.subplots(1, 2)

                ax[0].axis('off')
                ax[1].axis('off')
                fig.patch.set_visible(False)

                ax[0].imshow(segment_img)
                ax[1].imshow(segment_copy)

                plt.savefig(file_name, dpi=60)
                plt.close('all')
            
            self.__dump_segment_features(segment_copy, mask, fp1, fp3, i+1)

            if dump_segs == True:
                fp4.write("%s\n" % file_name)

            j += 1

        fp.close()
        fp1.close()
        fp2.close()
        if dump_segs == True:
            fp3.close()
            fp4.close()

    def __find_segment_features(self, segment_copy, seg_block, idx):
        fv = []

        # shape of the segment
        scale_factor = 1.0
        img_height, img_width, n_dim = segment_copy.shape
        max_size = np.max([img_height, img_width])
        fv.extend([scale_factor*img_height/max_size, scale_factor*img_width/max_size])
    
        # block wise shape features
        shapeFeature = self.__xShapeFeatures(seg_block, idx)
        shapeFeature = np.asarray(shapeFeature)
        fv.extend(shapeFeature)

        # self.__set_timer("surf")
        surfHist = self.__xSurfHist(segment_copy)
        surfHist = np.asarray(surfHist)
        cv2.normalize(surfHist, surfHist, 0, 1, cv2.NORM_MINMAX)
        fv.extend(surfHist)
        # self.__print_timer("surf")

        # self.__set_timer("hog")
        hogHist = self.__xHOGHist(segment_copy)
        hogHist = np.asarray(hogHist)
        cv2.normalize(hogHist, hogHist, 0, 1, cv2.NORM_MINMAX)
        fv.extend(hogHist)
        # self.__print_timer("hog")

        # self.__set_timer("rgb")
        rgbHist = self.__xRGBHist(segment_copy, seg_block, idx)
        rgbHist = np.asarray(rgbHist)
        cv2.normalize(rgbHist, rgbHist, 0, 1, cv2.NORM_MINMAX)
        fv.extend(rgbHist)
        # self.__print_timer("rgb")

        fv = np.reshape(fv, -1)

        return fv


    def __dump_segment_features(self, segment_copy, seg_block, fp1, fp2, idx):

        fv = self.__find_segment_features(segment_copy, seg_block, idx)
        np.savetxt(fp1, np.atleast_2d(fv), fmt='%.8f')

        if dump_segs == True:
            np.savetxt(fp2, np.atleast_2d(fv), fmt='%.8f')
        

    def __xShapeFeatures(self, seg_block, idx, num_xblock=12, num_yblock=12):
        shapeFeature = []
        scale = 1.0

        img_height, img_width = seg_block.shape

        x_step = 1.0*img_height/num_xblock
        y_step = 1.0*img_width/num_yblock
        block_size = x_step*y_step

        x, y = 0.0, 0.0

        for i in range(num_xblock):
            y = 0.0
            for j in range(num_yblock):
                x_start = int(x)
                y_start = int(y)
                x_end = int(x+x_step)
                y_end = int(y+y_step)
                img_block = seg_block[x_start:x_end, y_start:y_end]
                pixel_count = img_block.ravel().tolist().count(idx)
                ratio = scale*pixel_count/block_size
                shapeFeature.extend([ratio])
                y += y_step

            x += x_step

        return shapeFeature

    
    def __find_mean_color(self, image_block):
        mean_r = np.mean(image_block[:, :, 0])
        mean_g = np.mean(image_block[:, :, 1])
        mean_b = np.mean(image_block[:, :, 2])

        return [mean_r, mean_g, mean_b]

    def __xRGBHistWrap(self, image, num_xblock=12, num_yblock=12):
        rgbHist = []
        scale = 2

        img_height, img_width, n_dim = image.shape
        image = rgb2lab(image)

        x_step = img_height/num_xblock
        y_step = img_width/num_yblock

        x, y = 0, 0

        for i in range(num_xblock):
            y = 0
            for j in range(num_yblock):
                img_block = image[x:x+x_step, y:y+y_step,:]
                # hist_item = self.__xRGBHist(img_block)
                # hist_item = np.asarray(hist_item)
                # cv2.normalize(hist_item, hist_item, 0, scale, cv2.NORM_MINMAX)
                # rgbHist.extend(hist_item)
                mean_color = self.__find_mean_color(img_block)
                rgbHist.extend(mean_color)
                y += y_step

            x += x_step

        return rgbHist

    
    def __xRGBHist(self, image, seg_block, idx):
        numBins = 256
    
        # seg = np.copy(seg_block).astype(dtype=np.uint8)
        # mask=(seg!=idx)
        # seg[mask] = 0

        bCh, gCh, rCh = cv2.split(image)

        bins = np.arange(numBins).reshape(numBins,1)
        color = [ (255,0,0),(0,255,0),(0,0,255) ]
        
        rgbHist = []
        for item,col in zip([bCh, gCh, rCh],color):
            hist_item = cv2.calcHist([item],[0],None,[numBins],[0,255])
            rgbHist.extend(hist_item)

        return rgbHist

    
    def __xHOGHist(self, image):
    
        nBins = 64

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imageBlur = cv2.GaussianBlur(image, (5,5), 0)
        
        fdescriptor = hog(imageBlur, orientations=nBins, pixels_per_cell=(8, 8),
                                    cells_per_block=(1, 1), visualise=False)
        

        fd = np.reshape(fdescriptor, (-1, nBins))
        fHist = np.sum(fd, axis=0)

        return fHist

    
    def __xSurfHist(self, image):
    
        nBins = 64
        hessian_threshold = 500
        nOctaves = 4
        nOctaveLayers = 2
    
        imageGS = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        surf = cv2.SURF(hessian_threshold, nOctaves, nOctaveLayers, False, True)
        keypoints, descriptors = surf.detectAndCompute(imageGS, None) 
        
        surfHist = np.zeros(nBins)

        if len(keypoints) > 0:
            surfHist = np.sum(descriptors, axis=0)
    
        return surfHist


    def __set_timer(self, mesg=""):
        self.timer = time.time()
        if len(mesg) > 0:
            print "Starting ", mesg, "..."

    def __print_timer(self, mesg=""):
        print mesg, "done. run time = ", time.time() - self.timer
        print


    def __find_saliency_of_segments(self, seg_map, sal_map):
        height, width = seg_map.shape
        num_segments = np.amax(seg_map)+1
        #print num_segments
        saliency_list = np.zeros(shape=(num_segments,2), dtype=(float, int))

        for i in xrange(num_segments):
            saliency_list[i][1] = i

        pixel_count = np.zeros(num_segments)
        for i in xrange(height):
            for j in xrange(width):
                if seg_map[i][j] != 0:
                    seg_id = int(seg_map[i][j])
                    saliency_list[seg_id][0] += sal_map[i][j]
                    pixel_count[seg_id] += 1

        #print sorted(pixel_count)
        #print pixel_count
        
        # for i in xrange(num_segments):
        #     #print pixel_count[i]
        #     #print saliency_list[i][0]
        #     #print saliency_list[i][1]

        #     saliency_list[i][0] = saliency_list[i][0]/(pixel_count[i]+1)

        #print saliency_list
        saliency_list = sorted(saliency_list, key=lambda x: x[0], reverse=True)
        #saliency_list.sort()
        #print saliency_list

        salient_objects = np.zeros(num_segments)
        for i in xrange(num_segments):
            #print saliency_list[i][0]
            #print saliency_list[i][1]
            salient_objects[int(saliency_list[i][1])] = 1
        
        return saliency_list, salient_objects, pixel_count

    def plot_energy_map(self, db_path=None):

        fig, ax = plt.subplots(2, 3)
        # fig.set_size_inches(8, 3, forward=True)
        # plt.subplots_adjust(0.05, 0.05, 0.95, 0.95, 0.05, 0.05)
        
        # fig.patch.set_visible(False)
        ax[0,0].axis('off')
        ax[0,1].axis('off')
        ax[0,2].axis('off')
        ax[1,0].axis('off')
        ax[1,1].axis('off')
        ax[1,2].axis('off')

        ax[0,0].imshow(self.image)
        ax[0,0].set_title("Input Image")

        ax[0,1].imshow(mark_boundaries(self.image, self.slic_map))
        ax[0,1].set_title("SLIC")

        ax[0,2].imshow(self.saliency_map, interpolation='nearest')
        ax[0,2].set_title("Saliency")

        ax[1,0].imshow(self.segment_map2, interpolation='nearest')
        ax[1,0].set_title("Full Segmented Image")

        # ax[1,1].imshow(self.graph_nodes)
        ax[1,1].imshow(self.graph_image)
        ax[1,1].set_title("Important Objects")

        ax[1,2].imshow(self.color_energy_map)
        ax[1,2].set_title("Color Energy")

        spath = db_path + 'color_energy/'
        if not os.path.exists(spath):
            os.makedirs(spath)

        db, img_name = os.path.split(self.image_src)
        file_name = spath + img_name
        plt.savefig(file_name, dpi=500)
        plt.close('all')


    def plot_maps(self, db_path=None):

        fig, ax = plt.subplots(1, 4)
        fig.set_size_inches(8, 3, forward=True)
        plt.subplots_adjust(0.05, 0.05, 0.95, 0.95, 0.05, 0.05)
        
        # fig.patch.set_visible(False)
        ax[0].axis('off')
        ax[1].axis('off')
        ax[2].axis('off')
        ax[3].axis('off')

        ax[0].imshow(mark_boundaries(self.image, self.segment_map))
        ax[0].set_title("Image")

        ax[1].imshow(self.saliency_map, interpolation='nearest')
        ax[1].set_title("Saliency")

        ax[2].imshow(self.edge_map, interpolation='nearest')
        ax[2].set_title("Edge Map")

        ax[3].imshow(self.sal_map, interpolation='nearest')
        ax[3].set_title("Corrected Saliency")

        spath = db_path + 'edge/'
        if not os.path.exists(spath):
            os.makedirs(spath)

        db, img_name = os.path.split(self.image_src)
        file_name = spath + img_name
        plt.savefig(file_name, dpi=500)
        plt.close('all')


