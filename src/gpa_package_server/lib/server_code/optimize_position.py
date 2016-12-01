import sys, os, time
import numpy as np
import scipy
import cv2
from skimage import io

import _mypath
import spring_graph_model as sgm


def get_results(graph, image, img_src, dump_path=None):

    pos_dumps = None
    if dump_path is not None:
        pos_dumps = dump_path + '/pos/'
        if not os.path.exists(pos_dumps):
            os.makedirs(pos_dumps)

    img_h, img_w, img_dim = image.shape
    size_scale = np.sqrt(img_h*img_w)

    rec_positions = []
    rec_color = []

    for node in graph.get_nodes():
        if not node.fix:
            # calculate position of this node
            x0 = node.x_pos*img_w
            y0 = node.y_pos*img_h
            # draw this node
            rgb_color = 255*node.color

            circle_radius = 1.5*node.radius*size_scale
            # opencv its bgr
            fill_color = (rgb_color[2], rgb_color[1], rgb_color[0])

            x_0 = x0 - 1.5*circle_radius
            y_0 = y0 - 3.5*circle_radius

            x_1 = x0 + 1.5*circle_radius
            y_1 = y0 + 5.0*circle_radius

            # boundary check
            if x_0 < 0:
                x_0 = 0
            if y_0 < 0:
                y_0 = 0
            if x_1 > img_w-1:
                x_1 = img_w-1
            if y_1 > img_h-1:
                y_1 = img_h-1

            if dump_path is not None:
                cv2.rectangle(image, (int(x_0), int(y_0)), (int(x_1), int(y_1)), fill_color, 2)

            rec_positions.append([int(x_0), int(y_0), int(x_1), int(y_1)])
            rec_color.append(rgb_color)

    if dump_path is not None:
        img_n = pos_dumps + os.path.split(img_src)[1]
        cv2.imwrite(img_n, image)

    return np.asarray(rec_positions), rec_color

def draw_pos(graph, img_src, dump_path):
    pos_dumps = dump_path + '/pos/'
    if not os.path.exists(pos_dumps):
        os.makedirs(pos_dumps)

    image = cv2.imread(img_src)

    img_h, img_w, img_dim = image.shape
    size_scale = np.sqrt(img_h*img_w)

    rec_positions = []
    rec_size = []
    rec_color = []

    for node in graph.get_nodes():
        if not node.fix:
            # calculate position of this node
            x0 = node.x_pos*img_w
            y0 = node.y_pos*img_h
            # draw this node
            rgb_color = 255*node.color

            circle_radius = 1.5*node.radius*size_scale
            # opencv its bgr
            fill_color = (rgb_color[2], rgb_color[1], rgb_color[0])

            x_0 = x0 - 1.5*circle_radius
            y_0 = y0 - 3.5*circle_radius

            x_1 = x0 + 1.5*circle_radius
            y_1 = y0 + 5.0*circle_radius

            # boundary check
            if x_0 < 0:
                x_0 = 0
            if y_0 < 0:
                y_0 = 0
            if x_1 > img_w-1:
                x_1 = img_w-1
            if y_1 > img_h-1:
                y_1 = img_h-1

            cv2.rectangle(image, (int(x_0), int(y_0)), (int(x_1), int(y_1)), fill_color, 2)

            rec_positions.append([node.x_pos, node.y_pos])
            rec_size.append(node.size)
            rec_color.append(rgb_color)

    img_n = pos_dumps + os.path.split(img_src)[1]

    cv2.imwrite(img_n, image)

    return rec_positions, rec_size, rec_color

def gen_pos_rec(s_objects, p_objects, psuedo_objs, img, img_src, visualize=False, dump_path=None, server=False):

    graph = sgm.create_graph(s_objects, p_objects)

    # update number of people
    graph.num_people = len(psuedo_objs)

    graph = sgm.minimize_energy(graph, psuedo_objs, img_src, visualize=visualize, dump_path=dump_path)

    rec_positions, rec_color = get_results(graph, img, img_src, dump_path)

    return rec_positions, rec_color
