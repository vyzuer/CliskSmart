import itertools
import sys, os, time
import numpy as np
import scipy
from scipy import linalg
import matplotlib.pyplot as plt

import _mypath

import preprocess.gmm_modeling as gmm_mod
import preprocess.utils as my_utils

from sklearn.decomposition import RandomizedPCA
from sklearn import mixture
from sklearn.externals import joblib
from sklearn import preprocessing


def visualize_results(X, model, scaler, dump_dir, bic_scores, n_components_range, b_num_faces=False, b_face_size=False):

    # scale the axis for 4:3 aspect ratio
    x_scale, y_scale = 4, 3

    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)
    
    bic = np.array(bic_scores)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    color_iter = itertools.cycle(['k', 'r', 'g', 'b', 'c', 'm', 'y'])
    bars = []
    
    # Plot the BIC scores
    fig, ax = plt.subplots()
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + .2 * (i - 2)
        bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                      (i + 1) * len(n_components_range)],
                            width=.2, color=color))
    plt.xticks(n_components_range)
    plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    plt.title('BIC score per model')
    xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
        .2 * np.floor(bic.argmin() / len(n_components_range))
    plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
    ax.set_xlabel('Number of components')
    plt.legend([b[0] for b in bars], cv_types)

    plot_path = dump_dir + "/gmm_bic.png"
    plt.savefig(plot_path, dpi=400)
    plt.close()
    
    xmin, xmax = 0, x_scale
    ymin, ymax = 0, y_scale

    x_min = 0
    y_min = 0
    x_max = 1
    y_max = 1

    x, y = np.mgrid[x_min:x_max:400j, y_min:y_max:300j]
    
    positions = None
    if b_face_size and b_num_faces:
        positions = np.vstack([x.ravel(), y.ravel(), 2*np.ones(400*300), 0.1*np.ones(400*300)])
    elif b_num_faces:
        positions = np.vstack([x.ravel(), y.ravel(), 2*np.ones(400*300)])
    else:
        positions = np.vstack([x.ravel(), y.ravel()])

    if scaler is not None:
        positions = scaler.transform(zip(*positions))
    else:
        positions = zip(*positions)

    prob_score, response = model.score_samples(positions)
    f = np.exp(prob_score)
    f = np.reshape(f, x.shape)

    plt.subplot(311)
    
    # plt.xlim(x_min,x_max)
    # plt.ylim(y_min,y_max)
    plt.xticks(())
    plt.yticks(())
    plt.imshow(np.flipud(np.rot90(f)), cmap=plt.cm.ocean_r, extent=[xmin, xmax, ymin, ymax])
    # plt.colorbar()

    # Plot the winner
    plt.subplot(312, aspect='equal')
    Y_ = model.predict(X)

    if scaler is not None:
        X = scaler.inverse_transform(X)

    # scaling
    X[:,0] *= x_scale
    X[:,1] *= y_scale

    for i, (mean, color) in enumerate(zip(model.means_,
                                                 color_iter)):
        plt.scatter(X[Y_ == i, 0], y_scale-X[Y_ == i, 1], .8, color=color)
    
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xticks(())
    plt.yticks(())

    plt.subplot(313, aspect='equal')
    plt.xlim(0, x_scale)
    plt.ylim(0, y_scale)

    plt.xticks(())
    plt.yticks(())
    plt.scatter(X[:,0], y_scale-X[:,1], 0.6)

    plot_path = dump_dir + "/gmm.png"
    plt.savefig(plot_path, dpi=400)
    plt.close()

    # return the gaussian map
    return np.flipud(np.rot90(f))


def pre_process(data):

    scaler_model = preprocessing.StandardScaler()
    data = scaler_model.fit_transform(data)

    return data, scaler_model


def _modeling(data, dump_path, b_scale=True, n_components_range=range(1,10), n_iter=1000,  b_visualize=False, b_dump_model=True, b_num_faces=False, b_face_size=False):

    model = None
    scaler_model = None
    gmm_map = None

    if b_scale:
        data, scaler_model = pre_process(data)

    model, bic_scores = gmm_mod.gmm(data, n_components_range, n_iter=n_iter)

    if b_visualize:
        gmm_map = visualize_results(data, model, scaler_model, dump_path, bic_scores, n_components_range, b_num_faces=b_num_faces, b_face_size=b_face_size)

    # dump 
    timer = time.time()
    print 'Dumping...'

    if b_dump_model:
        dump_model(dump_path, model, scaler_model)
    
    print 'Dumping done.'
    print 'Total running time ', time.time() - timer

    return gmm_map


def dump_model(dump_path, model, scaler):
    dump_dir = dump_path + 'models/'

    # dump the gmm model
    model_dump = dump_dir + "/gmm/"
    if not os.path.exists(model_dump):
        os.makedirs(model_dump)

    model_path = model_dump + "/gmm.pkl"

    scaler_dump = dump_dir + "/scaler/"
    if not os.path.exists(scaler_dump):
        os.makedirs(scaler_dump)

    scaler_path = scaler_dump + "/scaler.pkl"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)


def _visualize_clusters(gmm_maps, dump_path):

    n_clusters = len(gmm_maps)
    f_name = dump_path + 'gmm_position.png'

    n_rows = n_clusters/10 + 1

    for i in range(n_clusters):

        gmm_map = gmm_maps[i]
        
        plt.subplot(n_rows, 10 , i+1)
        plt.axis('off')
        plt.imshow(gmm_map, interpolation='nearest')

    plt.savefig(f_name)
    plt.close()


def perform_modeling(dataset_path, master_dump_path, n_components_range, b_dump_model, b_scale, n_iter, b_num_faces=False, b_face_size=False, b_visualize=True, clean=True):

    assert os.path.exists(dataset_path)
    assert os.path.exists(master_dump_path)

    dump_path = master_dump_path + 'position_modeling/'

    # check if compilation required
    if not clean and my_utils.dataset_valid(dump_path):
        print '\nPosition Model Database is up-to-date.\nPass clean=True for fresh compilation\n'
        return
    else:
        my_utils.invalidate_dataset(dump_path)

    # dump file for descriptor
    pi_list = master_dump_path + 'face_info/type_info.list'

    X = None
    data = np.loadtxt(pi_list)
    if b_num_faces:
        if b_face_size:
            X = np.hstack([data[:,0:2], data[:,6:7], data[:,9:10]])
        else:
            X = np.hstack([data[:,0:2], data[:,6:7]])
    else:
        X = data[:,0:2]

    cluster_dump_path = master_dump_path + 'scene_categories/'
    f_n_clusters = cluster_dump_path + 'num_clusters.info'
    n_clusters = np.loadtxt(f_n_clusters, dtype='int')

    f_labels = cluster_dump_path + 'labels.list'
    cluster_labels = np.loadtxt(f_labels, dtype='int')

    # perform modeling
    timer = time.time()
    print 'Performing position modeling...'

    # store the gmm_maps from each cluster for visualization
    gmm_maps = []

    # iterate over all the clusters for gmm
    for i in range(n_clusters):

        # do logical indexing for extracting data
        X_ = X[cluster_labels==i,:]

        # set the dump path for this cluster
        c_dump_path = dump_path + str(i) + '/'

        gmm_map = _modeling(X_, c_dump_path, b_scale=b_scale, n_components_range=n_components_range, n_iter=n_iter, b_visualize=b_visualize, b_num_faces=b_num_faces, b_face_size=b_face_size)
        
        gmm_maps.append(gmm_map)

    print 'Modeling done. '
    print 'Total running time ', time.time() - timer

    # visualize the gmm_maps for each cluster
    if b_visualize:
        _visualize_clusters(gmm_maps, dump_path)

    my_utils.validate_dataset(dump_path)


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print "Usage : dataset_path dump_path"
        sys.exit(0)

    dataset_path = sys.argv[1]
    dump_path = sys.argv[2]

    perform_modeling(dataset_path, dump_path, \
            n_components_range=range(1, 20), \
            b_dump_model=True, \
            b_scale=True, \
            b_num_faces=True, \
            b_face_size=True, \
            n_iter=5000, \
            clean=True)



