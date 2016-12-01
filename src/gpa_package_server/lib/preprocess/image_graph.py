import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from scipy import ndimage
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from sklearn import preprocessing
from skimage.measure import block_reduce
import sys
import cv2

import _mypath

import face_detection as fdetect
import img_proc.saliency as saliency

_DEBUG = False


def __xSurfHist(image, nBins, mask):
    
    nBins = 64
    hessian_threshold = 500
    nOctaves = 4
    nOctaveLayers = 2
    
    surf = cv2.SURF(hessian_threshold, nOctaves, nOctaveLayers, False, True)
    keypoints, descriptors = surf.detectAndCompute(image, mask=mask) 
    
    surfHist = np.zeros(nBins)

    num_kp = len(keypoints)

    if num_kp > 0:
        surfHist = np.sum(descriptors, axis=0)/num_kp
    
    return surfHist


def _segmentation(img, compactness=15, n_segments=300, sigma=1.0, max_iter=10, enforce_connectivity=False, visualise=False):

    slic_map, segments = slic(img, compactness=compactness, n_segments=n_segments, sigma=sigma, max_iter=max_iter, enforce_connectivity=enforce_connectivity)

    if _DEBUG:

        print("Number of SLIC segments: %d" % len(np.unique(slic_map)))

        if visualise:
            plt.imshow(mark_boundaries(img, slic_map))
            plt.show()
            plt.close('all')
    
    return slic_map, segments[:,3:6]


def _node_overlap(c_x, c_y, n_pixelsx, n_pixelsy, frames):
    overlap = False

    c_x1 = c_x + n_pixelsx
    c_y1 = c_y + n_pixelsy

    # frames have different x and y axis
    for x0,y0,x1,y1 in frames:
        if c_y1 > x0 and c_y < x1 and c_x1 > y0 and c_x < y1 :
            overlap = True

    return overlap

def _slic_to_graph(slic_map, grid_size, frames=None):

    # nodes removed from the graph will have super-pixel-id = -2

    img_height, img_width = slic_map.shape

    n_blocksx, n_blocksy = grid_size
    n_pixelsx = img_height/n_blocksx
    n_pixelsy = img_width/n_blocksy

    n_pixelsx_f = 1.0*img_height/n_blocksx
    n_pixelsy_f = 1.0*img_width/n_blocksy

    # the last dimension store - superpixel_id, valid flag, four color edges, four surf edges
    graph_map = np.zeros(shape=(n_blocksx, n_blocksy, 10))

    c_x, c_y = 0, 0

    prev_id = 1

    for i in range(n_blocksx):
        c_y = 0
        for j in range(n_blocksy):

            # check if this node has to be removed
            if _node_overlap(c_x, c_y, n_pixelsx, n_pixelsy, frames):
                graph_map[i,j,1] = 1

            current_block = slic_map[c_x:c_x+n_pixelsx, c_y:c_y+n_pixelsy]
            counts = np.bincount(np.asarray(current_block).reshape(-1))
            superpixel_id = np.argmax(counts)
            if superpixel_id == 0:
                superpixel_id = prev_id
            else:
                prev_id = superpixel_id

            graph_map[i, j, 0] = superpixel_id

            c_y = int((j+1)*n_pixelsy_f)

        prev_id = graph_map[i, 0, 0]

        c_x = int((i+1)*n_pixelsx_f)
   
    return graph_map

def _edge_dist(mean_color, surf_features, id_1, id_2, b_surf):
    color_dist = 0.0
    surf_dist = 0.0

    for i in range(0,3):
        color_dist += (mean_color[int(id_1),i] - mean_color[int(id_2),i])**2

    if b_surf:
        surf_dist = np.sum((surf_features[int(id_1)]-surf_features[int(id_2)])**2)

    return color_dist, surf_dist


def _compute_edges_full(graph_map, mean_color):

    n_blocksx, n_blocksy, n_dim = graph_map.shape
    n_nodes = n_blocksx*n_blocksy
    graph = np.zeros(shape=(n_nodes, n_nodes))

    for i in range(n_nodes):
        for j in range(n_nodes):
            x_id = i/n_blocksy
            y_id = i%n_blocksy
            sp_id1 = graph_map[x_id,y_id,0]

            x_id = j/n_blocksy
            y_id = j%n_blocksy
            sp_id2 = graph_map[x_id,y_id,0]

            graph[i,j] = graph[j,i] = _edge_dist(mean_color, sp_id1, sp_id2)

    return graph

def _compute_edges(graph_map, mean_color, surf_features, edge_connectivity=8, normalize=False, b_surf=False):

    n_blocksx, n_blocksy, n_dim = graph_map.shape
    n_nodes = n_blocksx*n_blocksy
    # the last dimension is for color distance and surf distance
    graph = np.zeros(shape=(n_nodes, n_nodes, 2))
    descriptor_c = np.zeros(4*n_nodes)
    descriptor_s = np.zeros(4*n_nodes)

    d_idx= 0
    node_id = 0
    for i in range(n_blocksx):
        for j in range(n_blocksy):
            sp_id = graph_map[i,j,0]

            sp_valid = graph_map[i,j,1]
            if sp_valid == 1:
                d_idx += 4
                node_id += 1
                continue

            # right
            if j < n_blocksy-1:
                nxt_sp_id = graph_map[i,j+1,0]
                nxt_node_id = node_id+1
                
                edge_c, edge_s = _edge_dist(mean_color, surf_features, sp_id, nxt_sp_id, b_surf)

                graph[node_id, nxt_node_id, 0] = edge_c
                graph[nxt_node_id, node_id, 0] = edge_c

                descriptor_c[d_idx] = edge_c
                graph_map[i,j,2] = edge_c

                if b_surf:
                    graph[node_id, nxt_node_id, 1] = edge_s
                    graph[nxt_node_id, node_id, 1] = edge_s

                    descriptor_s[d_idx] = edge_s
                    graph_map[i,j,6] = edge_s

            d_idx += 1

            # bottom
            if i < n_blocksx-1:
                nxt_sp_id = graph_map[i+1,j,0]
                nxt_node_id = node_id+n_blocksy

                edge_c, edge_s = _edge_dist(mean_color, surf_features, sp_id, nxt_sp_id, b_surf)

                graph[node_id, nxt_node_id, 0] = edge_c
                graph[nxt_node_id, node_id, 0] = edge_c

                descriptor_c[d_idx] = edge_c
                graph_map[i,j,3] = edge_c

                if b_surf:
                    graph[node_id, nxt_node_id, 1] = edge_s
                    graph[nxt_node_id, node_id, 1] = edge_s

                    descriptor_s[d_idx] = edge_s
                    graph_map[i,j,7] = edge_s

            d_idx += 1

            # if 8 connectivity
            if edge_connectivity == 8:
                # right bottom
                if i < n_blocksx-1 and j < n_blocksy-1:
                    nxt_sp_id = graph_map[i+1,j+1,0]
                    nxt_node_id = node_id+n_blocksy+1

                    edge_c, edge_s = _edge_dist(mean_color, surf_features, sp_id, nxt_sp_id, b_surf)

                    graph[node_id, nxt_node_id, 0] = edge_c
                    graph[nxt_node_id, node_id, 0] = edge_c

                    descriptor_c[d_idx] = edge_c
                    graph_map[i,j,4] = edge_c

                    if b_surf:
                        graph[node_id, nxt_node_id, 1] = edge_s
                        graph[nxt_node_id, node_id, 1] = edge_s

                        descriptor_s[d_idx] = edge_s
                        graph_map[i,j,8] = edge_s

                d_idx += 1

                # left bottom
                if i < n_blocksx-1 and j > 0:
                    nxt_sp_id = graph_map[i+1,j-1,0]
                    nxt_node_id = node_id+n_blocksy-1

                    edge_c, edge_s = _edge_dist(mean_color, surf_features, sp_id, nxt_sp_id, b_surf)

                    graph[node_id, nxt_node_id, 0] = edge_c
                    graph[nxt_node_id, node_id, 0] = edge_c

                    descriptor_c[d_idx] = edge_c
                    graph_map[i,j,5] = edge_c

                    if b_surf:
                        graph[node_id, nxt_node_id, 1] = edge_s
                        graph[nxt_node_id, node_id, 1] = edge_s

                        descriptor_s[d_idx] = edge_s
                        graph_map[i,j,9] = edge_s

                d_idx += 1

            node_id += 1

    if normalize:
        # divide the edges with maximum value
        # color normalization
        max_dist = np.max(graph_map[:,:,2:6])
        if max_dist > 0:
            graph_map[:,:,2:6] = graph_map[:,:,2:6]/max_dist
        # surf normalization
        if b_surf:
            max_dist = np.max(graph_map[:,:,6:10])
            if max_dist > 0:
                graph_map[:,:,6:10] = graph_map[:,:,6:10]/max_dist

        descriptor_c = preprocessing.normalize([descriptor_c], axis=1)[0]
        if b_surf:
            descriptor_s = preprocessing.normalize([descriptor_s], axis=1)[0]

    # mean_edge = np.mean(np.mean(graph_map[:,:,2:6], axis=0), axis=0)
    # 
    # # fill the area covered by people with average values
    # for i in range(n_blocksx):
    #     for j in range(n_blocksy):
    #         sp_valid = graph_map[i,j,1]
    #         if sp_valid == 1:
    #             graph_map[i,j,2:6] = mean_edge

    return graph, descriptor_c, descriptor_s

def visualize_s_descriptor(descriptor, dump_path='', visualise=False):

    n_dim = len(descriptor)
    assert n_dim%12 == 0

    n_scale = int(np.sqrt(n_dim/12))
    x_dim = n_scale*3
    y_dim = n_scale*4

    vis_map = np.reshape(descriptor, (x_dim, y_dim))
            
    if visualise:
        plt.imshow(vis_map, interpolation='nearest')
        plt.show()
        plt.close('all')

    return vis_map

            
def visualize_descriptor(descriptor, dump_path='', visualise=False):

    n_dim = len(descriptor)
    assert n_dim%4 == 0

    n_nodes = n_dim/4
    assert n_nodes%12 == 0

    n_scale = int(np.sqrt(n_nodes/12))
    x_dim = n_scale*3
    y_dim = n_scale*4

    vis_map = np.zeros(shape=(3*x_dim, 3*y_dim))

    x_pos, y_pos = 1, 1
    x_step, y_step = 3, 3

    # the order of the descriptor is right,bottom,right-bottom,left-bottom
    # the order of visualization map is left-top,top,right-top,right,right-bottom,bottom,left-bottom,left
    for i in range(x_dim):
        y_pos = 1
        for j in range(y_dim):

            node_id = i*y_dim + j

            if i > 0 and j > 0:
                idx = (node_id - y_dim - 1)*4 + 2
                vis_map[x_pos-1, y_pos-1] = descriptor[idx]

            if i > 0:
                idx = (node_id - y_dim)*4 + 1
                vis_map[x_pos-1, y_pos] = descriptor[idx]

            if i > 0 and j < y_dim-1:
                idx = (node_id - y_dim + 1)*4 + 3
                vis_map[x_pos-1, y_pos+1] = descriptor[idx]

            if j < y_dim-1:
                idx = node_id*4
                vis_map[x_pos, y_pos+1] = descriptor[idx]

            if i < x_dim-1 and j < y_dim-1:
                idx = node_id*4 + 2
                vis_map[x_pos+1, y_pos+1] = descriptor[idx]

            if i < x_dim-1:
                idx = node_id*4 + 1
                vis_map[x_pos+1, y_pos] = descriptor[idx]

            if i < x_dim-1 and j > 0:
                idx = node_id*4 + 3
                vis_map[x_pos+1, y_pos-1] = descriptor[idx]

            if j > 0:
                idx = (node_id-1)*4
                vis_map[x_pos, y_pos-1] = descriptor[idx]

            vis_map[x_pos, y_pos] = np.mean(vis_map[x_pos-1:x_pos+1, y_pos-1:y_pos+1])

            y_pos += y_step

        x_pos += x_step
            
    if visualise:
        plt.imshow(vis_map, interpolation='nearest')
        plt.show()
        plt.close('all')

    return vis_map
            

def _visualize_graph(graph, graph_map):
    x_dim, y_dim, n_dim = graph_map.shape

    vis_map = np.zeros(shape=(3*x_dim, 3*y_dim))

    # this is the starting position
    x_pos, y_pos = 1, 1
    x_step, y_step = 3, 3
    node_id = 0

    for i in range(x_dim):
        y_pos = 1
        for j in range(y_dim):

            if i > 0 and j > 0:
                vis_map[x_pos-1, y_pos-1] = graph[node_id, node_id-y_dim-1,0]

            if i > 0:
                vis_map[x_pos-1, y_pos] = graph[node_id, node_id-y_dim,0]

            if i > 0 and j < y_dim-1:
                vis_map[x_pos-1, y_pos+1] = graph[node_id, node_id-y_dim+1,0]

            if j < y_dim-1:
                vis_map[x_pos, y_pos+1] = graph[node_id, node_id+1,0]

            if i < x_dim-1 and j < y_dim-1:
                vis_map[x_pos+1, y_pos+1] = graph[node_id, node_id+y_dim+1,0]

            if i < x_dim-1:
                vis_map[x_pos+1, y_pos] = graph[node_id, node_id+y_dim,0]

            if i < x_dim-1 and j > 0:
                vis_map[x_pos+1, y_pos-1] = graph[node_id, node_id+y_dim-1,0]

            if j > 0:
                vis_map[x_pos, y_pos-1] = graph[node_id, node_id-1,0]

            vis_map[x_pos, y_pos] = np.mean(vis_map[x_pos-1:x_pos+1, y_pos-1:y_pos+1])

            node_id += 1

            y_pos += y_step

        x_pos += x_step
            
    plt.imshow(vis_map, interpolation='nearest')
    plt.show()
    plt.close('all')

    return vis_map

def plot_map(map1, map2):

    fig = plt.figure()
    a=fig.add_subplot(1,2,1)
    imgplot = plt.imshow(map1, interpolation='nearest')
    b=fig.add_subplot(1,2,2)
    imgplot = plt.imshow(map2, interpolation='nearest')
    
    plt.show()
    plt.close('all')


def _compute_s_features(s_map, visualise=False):

    nodes_x, nodes_y = s_map.shape

    # size (3x4)
    n_x, n_y = 3, 4
    fs_1 = block_reduce(s_map, block_size=(nodes_x/n_x, nodes_y/n_y), func=np.mean)

    if visualise:
        plot_map(s_map, fs_1)

    # size (6x8)
    n_x, n_y = 6, 8
    fs_2 = block_reduce(s_map, block_size=(nodes_x/n_x, nodes_y/n_y), func=np.mean)

    if visualise:
        plot_map(s_map, fs_2)

    # size (12x16)
    n_x, n_y = 12, 16
    fs_3 = block_reduce(s_map, block_size=(nodes_x/n_x, nodes_y/n_y), func=np.mean)

    if visualise:
        plot_map(s_map, fs_3)

    feature_set = np.hstack([fs_1.ravel()/4, fs_2.ravel()/2, fs_3.ravel()])

    return feature_set

def _extract_features(graph_map, n_x, n_y, lambda_1, lambda_2, normalize=False, b_surf=False):

    nodes_x, nodes_y, n_dim = graph_map.shape
    x_step, y_step = nodes_x/n_x, nodes_y/n_y
    x_step_f, y_step_f = 1.0*nodes_x/n_x, 1.0*nodes_y/n_y

    # both steps should be same
    assert x_step == y_step

    fdc = np.zeros(shape=(n_x, n_y, 4))
    fds = np.zeros(shape=(n_x, n_y, 4))
    x_pos, y_pos = 0,0

    for i in range(n_x):
        y_pos = 0
        x_pos = int(i*x_step_f)
        for j in range(n_y):
            y_pos = int(j*y_step_f)
            block = graph_map[x_pos:x_pos+x_step,y_pos:y_pos+y_step,2:6]
            fdc[i,j,:] = np.sum(np.sum(block, axis=0), axis=0)
            if b_surf:
                block = graph_map[x_pos:x_pos+x_step,y_pos:y_pos+y_step,6:10]
                fds[i,j,:] = np.sum(np.sum(block, axis=0), axis=0)

    if normalize:
        fdc = fdc/(x_step*y_step*1.0)
        if b_surf:
            fds = fds/(x_step*y_step*1.0)

    fd = lambda_1*fdc + lambda_2*fds

    return fd


def _compute_features(graph_map, normalize=True, visualise=False, b_surf=False):

    lambda_1, lambda_2 = 0.5, 0.5
    if b_surf:
        lambda_1, lambda_2 = 1.0, 0.0

    nodes_x, nodes_y, n_dim = graph_map.shape

    # size 4x1
    fsc_1 = np.sum(np.sum(graph_map[:,:,2:6], axis=0), axis=0)
    fss_1 = np.sum(np.sum(graph_map[:,:,6:10], axis=0), axis=0)
    if normalize:
        fsc_1 = fsc_1/(nodes_x*nodes_y*1.0)
        fss_1 = fss_1/(nodes_x*nodes_y*1.0)

    fs_1 = lambda_1*fsc_1 + lambda_2*fss_1

    # size 4x(3x4)
    n_x, n_y = 3, 4
    fs_2 = _extract_features(graph_map, n_x, n_y, lambda_1, lambda_2, normalize=normalize, b_surf=b_surf)

    if visualise:
        visualize_descriptor(fs_2.ravel(), visualise=True)

    # size 4x(6x8)
    n_x, n_y = 6, 8
    fs_3 = _extract_features(graph_map, n_x, n_y, lambda_1, lambda_2, normalize=normalize, b_surf=b_surf)

    if visualise:
        visualize_descriptor(fs_3.ravel(), visualise=True)

    # size 4x(12x16)
    n_x, n_y = 12, 16
    fs_4 = _extract_features(graph_map, n_x, n_y, lambda_1, lambda_2, normalize=normalize, b_surf=b_surf)

    if visualise:
        visualize_descriptor(fs_4.ravel(), visualise=True)

    # size 4x(24x32)
    n_x, n_y = 24, 32
    fs_5 = _extract_features(graph_map, n_x, n_y, lambda_1, lambda_2, normalize=normalize, b_surf=b_surf)

    if visualise:
        visualize_descriptor(fs_5.ravel(), visualise=True)

    feature_set = np.hstack([fs_1.ravel()/4, fs_2.ravel()/2, fs_3.ravel(), fs_4.ravel()])

    return feature_set

def _find_surf_feaures(img, slic_map):

    num_segments = np.amax(slic_map)+1

    n_bins = 64
    surf_features = np.zeros(shape=(num_segments, n_bins))

    image_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(shape=(slic_map.shape), dtype=np.uint8)

    for i in xrange(num_segments):

        mask_id = slic_map == i
        mask[mask_id] = 1

        surf_features[i] = __xSurfHist(image_gs, n_bins, mask)

        mask[mask_id] = 0

    return surf_features

def _downsample_map(smap, frames, grid_size):
    img_height, img_width = smap.shape

    n_blocksx, n_blocksy = grid_size
    n_pixelsx = img_height/n_blocksx
    n_pixelsy = img_width/n_blocksy

    n_pixelsx_f = 1.0*img_height/n_blocksx
    n_pixelsy_f = 1.0*img_width/n_blocksy

    # the last dimension store - saliency value, valid flag
    sal_map = np.zeros(shape=(n_blocksx, n_blocksy, 2))

    c_x, c_y = 0, 0

    for i in range(n_blocksx):
        c_y = 0
        for j in range(n_blocksy):

            # check if this node has to be removed
            if _node_overlap(c_x, c_y, n_pixelsx, n_pixelsy, frames):
                sal_map[i,j,1] = 1
                c_y = int((j+1)*n_pixelsy_f)
                continue

            current_block = smap[c_x:c_x+n_pixelsx, c_y:c_y+n_pixelsy]
            sval = np.mean(current_block)

            sal_map[i, j, 0] = sval

            c_y = int((j+1)*n_pixelsy_f)

        c_x = int((i+1)*n_pixelsx_f)

    # find the mean saliency value and assign it to removed blocks
    # mean_sal = np.mean(sal_map[:,:,0])
    # 
    # for i in range(n_blocksx):
    #     for j in range(n_blocksy):

    #         # check if this node has to be removed
    #         if sal_map[i,j,1] == 1:
    #             sal_map[i,j,0] = mean_sal
   
    return sal_map

def get_saliency_based_descriptor(smap, frames, grid_size=(15,20), visualise=False):

    # down sample the map and also mark invalid blocks for later use
    ds_smap = _downsample_map(smap, frames, grid_size)

    # compute the feature descriptor based on spatial pyramid
    s_descriptor = _compute_s_features(ds_smap[:,:,0], visualise=visualise)

    return s_descriptor, ds_smap[:,:,0].ravel()

def image_graph(slic_map, mean_color, frames, grid_size=(15, 20), visualise=False, b_surf=False, surf_features=None):

    # convert slic to graph
    graph_map = _slic_to_graph(slic_map, grid_size, frames=frames)

    # compute graph edges
    graph, descriptor_c, descriptor_s = _compute_edges(graph_map, mean_color, surf_features, normalize=True, b_surf=b_surf)

    # use the complete descriptor to extract features for clustering
    feature_set = _compute_features(graph_map, normalize=True, visualise=visualise, b_surf=b_surf)

    # visualise the image graph
    if visualise:
        # _visualize_graph(graph, graph_map)
        # visualize_descriptor(feature_set[1012:])
        visualize_descriptor(descriptor_c, visualise=True)
        if b_surf:
            visualize_descriptor(descriptor_s, visualise=True)
        # visualize_descriptor(graph_map[:,:,2:6].ravel(), visualise=True)

    return feature_set, descriptor_c

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print 'usage : ', __file__, 'image_source'
        sys.exit(0)

    _DEBUG = True

    img_src = sys.argv[1]

    img = io.imread(img_src)

    # slic segmentation
    slic_map, mean_color = _segmentation(img, visualise=False)

    saliency_object = saliency.Saliency(img, 3)
    saliency_map = saliency_object.getSaliencyMap()

    # find surf features for superpixels
    b_surf = False
    surf_features = None
    if b_surf:
        surf_features = _find_surf_feaures(img, slic_map)

    # detect faces
    grid_size=(60, 80)
    padding = img.shape[1]/grid_size[1]
    faces, frames = fdetect.detect_face(img, padding=padding, visualise=False)

    feature_set, descriptor = image_graph(slic_map, mean_color, frames, grid_size=grid_size, visualise=True, b_surf=b_surf, surf_features=surf_features)

    sal_descriptor, ds_sal_map = get_saliency_based_descriptor(saliency_map, frames, grid_size=grid_size, visualise=True)
    
    plot_map(saliency_map, ds_sal_map)


