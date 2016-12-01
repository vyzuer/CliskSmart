import sys
import time
import os
import glob
import numpy as np
from skimage import io

import shutil
import _mypath

import preprocess.clustering as clustering
import preprocess.image_graph as img_graph
import preprocess.utils as my_utils

import matplotlib.pyplot as plt
from sklearn import preprocessing 
from sklearn.externals import joblib
from sklearn.decomposition import PCA

_DEBUG = False

master_dump_path = None

def dump_cluster_model(model, scaler_model, pca_model, dump_path, n_clusters, b_kmeans=True, b_scale=False, b_pca=False):
    # dump model

    model_path = dump_path + "cluster_model/"
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
        os.makedirs(model_path)
    else:
        os.makedirs(model_path)

    model_file = model_path + "cluster.pkl"
    joblib.dump(model, model_file)

    if b_scale:
        scaler_file = model_path + "scaler.pkl"
        joblib.dump(scaler_model, scaler_file)

    if b_pca:
        pca_file = model_path + "pca.pkl"
        joblib.dump(pca_model, pca_file)

    label_file = dump_path + "labels.list"
    np.savetxt(label_file, model.labels_, fmt='%d')

    num_cluster_file = dump_path + "num_clusters.info"
    np.savetxt(num_cluster_file, [n_clusters], fmt='%d')

    cluster_centers_file = dump_path + "centers.info"
    np.savetxt(cluster_centers_file, model.cluster_centers_)

    if not b_kmeans:
        cluster_centers_idx_file = dump_path + "centers_idx.info"
        np.savetxt(cluster_centers_idx_file, model.cluster_centers_indices_)


def pre_process(data, b_scale=False, b_normalize=False, b_pca=False, n_dims=100):
    scaler_model = None
    pca_model = None

    if b_normalize:
        data = preprocessing.normalize(data, norm='l2')

    if b_scale:
        scaler_model = preprocessing.StandardScaler()
        data = scaler_model.fit_transform(data)

    if b_pca:
        pca_model = PCA(n_components=n_dims)
        data = pca_model.fit_transform(data)
        print 'Total variance:', np.sum(pca_model.explained_variance_ratio_)

    return data, scaler_model, pca_model


def create_clusters(cl_dump_path, n_clusters):

    if os.path.exists(cl_dump_path):
        shutil.rmtree(cl_dump_path)
        os.makedirs(cl_dump_path)
    else:
        os.makedirs(cl_dump_path)

    for i in range(n_clusters):
        os.makedirs(cl_dump_path+str(i))


def dump_cluster_images(dataset_path, dump_path, model, n_clusters, b_kmeans):

    # create directories for images
    cl_dump_path = dump_path + 'clusters/'
    create_clusters(cl_dump_path, n_clusters)

    image_dir = dataset_path + "ImageDB/"
    
    image_list = dataset_path + "photo.info"
    img_details = np.loadtxt(image_list, dtype='string')
    n_images, n_dim = img_details.shape

    img_labels = model.labels_

    for i in range(n_images):
        img_name = img_details[i][0]

        img_src = image_dir + img_name
        img_dst = dump_path + 'clusters/' + str(img_labels[i]) + '/' + img_name

        os.symlink(img_src, img_dst)

    # copy cluster centers
    if not b_kmeans:
        # create directories for images
        cl_dump_path = dump_path + 'cluster_centers/'
        create_clusters(cl_dump_path, n_clusters)

        for i in range(n_clusters):
            img_idx = model.cluster_centers_indices_[i]
            img_name = img_details[img_idx][0]

            img_src = image_dir + img_name
            img_dst = dump_path + 'cluster_centers/' + str(img_labels[i]) + '/' + img_name

            os.symlink(img_src, img_dst)

def dump_s_pca_components(pca, dump_path, n_components=20):

    V = pca.components_

    f_rules = dump_path + 's_eigen_rules.png'

    for i in range(n_components):

        descriptor = V[i,304:]

        eigen_rule = img_graph.visualize_s_descriptor(descriptor)

        plt.subplot(5,4,i+1)
        plt.axis('off')
        plt.imshow(eigen_rule, interpolation='nearest')

    plt.savefig(f_rules)
    plt.close()

def dump_pca_components(pca, dump_path, n_components=20):

    V = pca.components_

    f_rules = dump_path + 'eigen_rules.png'

    for i in range(n_components):

        descriptor = V[i,52:244]

        eigen_rule = img_graph.visualize_descriptor(descriptor)

        plt.subplot(5,4,i+1)
        plt.axis('off')
        plt.imshow(eigen_rule, interpolation='nearest')

    plt.savefig(f_rules)
    plt.close()

def plot_cluster_s_mean(id1, id2, X, num, mean_path, labels):

    f_name = mean_path + str(num) + '_' + str(id1) + '.png'

    n_rows = num/10 + 1

    for i in range(num):

        data = X[labels==i,id1:id2]
        mean = np.mean(data, axis=0)
        mean = img_graph.visualize_s_descriptor(mean)

        plt.subplot(n_rows, 10 , i+1)
        plt.axis('off')
        plt.imshow(mean, interpolation='nearest')

    plt.savefig(f_name)
    plt.close()

        
def plot_cluster_mean(id1, id2, X, num, mean_path, labels):

    f_name = mean_path + str(num) + '_' + str(id1) + '.png'

    n_rows = num/10 + 1

    for i in range(num):

        data = X[labels==i,id1:id2]
        mean = np.mean(data, axis=0)
        mean = img_graph.visualize_descriptor(mean)

        plt.subplot(n_rows, 10 , i+1)
        plt.axis('off')
        plt.imshow(mean, interpolation='nearest')

    plt.savefig(f_name)
    plt.close()

def plot_cluster_mean_0(id1, id2, X, num, mean_path):

    f_name = mean_path + str(num) + '_' + str(id1) + '.png'

    n_rows = num/10 + 1

    for i in range(num):

        mean = X[i,id1:id2]
        mean = img_graph.visualize_descriptor(mean)

        plt.subplot(n_rows, 10 , i+1)
        plt.axis('off')
        plt.imshow(mean, interpolation='nearest')

    plt.savefig(f_name)
    plt.close()

def dump_cluster_s_centers(model, dump_path, num, X, smap):
    mean_path = dump_path + "scene_means_saliency/"
    if not os.path.exists(mean_path):
        os.makedirs(mean_path)

    labels = model.labels_

    for i,j in ((0,12), (12,60), (60,252)):
        plot_cluster_s_mean(i, j, X, num, mean_path, labels)
    
    n_samples, n_dim = smap.shape
    plot_cluster_s_mean(0, n_dim, smap, num, mean_path, labels)
    
def dump_cluster_centers(model, dump_path, num, X):
    mean_path = dump_path + "scene_means_edge/"
    if not os.path.exists(mean_path):
        os.makedirs(mean_path)

    labels = model.labels_

    for i,j in ((4,52), (52,244), (244,1012)):
        plot_cluster_mean(i, j, X, num, mean_path, labels)


def dump_cluster_centers_0(model, dump_path, num, pca_model, scaler_model, b_scale, b_pca):
    mean_path = dump_path + "scene_means/"
    if not os.path.exists(mean_path):
        os.makedirs(mean_path)

    X = model.cluster_centers_

    # pca inverse transform
    if b_pca:
        X = pca_model.inverse_transform(X)

    # inverse scaling
    if b_scale:
        X = scaler_model.inverse_transform(X)
    
    # for i,j in ((4,52), (52,244), (244,1012), (1012,4084)):
    for i,j in ((4,52), (52,244), (244,1012)):
        plot_cluster_mean_0(i, j, X, num, mean_path)
    

def _plot_cost_curve(dump_path, cost, silhouette):

    fig, ax1 = plt.subplots()

    num_points = len(cost)
    x = np.arange(num_points)
    x += 1 # cluster starting from 2

    ax1.plot(x , silhouette, 'b-')
    ax1.set_xlabel('No. of clusters')
    ax1.set_ylabel('Silhouette Coefficient', color='b')
    for tl in ax1.get_yticklabels():
        tl.set_color('b')

    ax2 = ax1.twinx()
    ax2.plot(x, cost, 'r.')
    ax2.set_ylabel('Cost', color='r')
    for tl in ax2.get_yticklabels():
        tl.set_color('r')
    
    dump_file = dump_path + 'cost_plot.png'
    plt.savefig(dump_file)
    plt.close()


def _cluster(dump_path, n_clusters, b_kmeans=True, b_dump_model=False, b_scale=True, b_normalize=True, b_pca=True, search_range=range(2,10), n_dims=100):

    global master_dump_path

    # dump file for descriptor
    # edge features
    essd_list = master_dump_path + 'scene_descriptors/essd.list'
    # saliency features
    sssd_list = master_dump_path + 'scene_descriptors/sssd.list'

    essd = np.loadtxt(essd_list)
    sssd = np.loadtxt(sssd_list)

    # saliency features
    smap_list = master_dump_path + 'scene_descriptors/smap.list'

    smap = np.loadtxt(smap_list)

    data = np.hstack([essd[:,0:244], sssd])

    data, scaler_model, pca_model = pre_process(data, b_scale=b_scale, b_normalize=b_normalize, b_pca=b_pca, n_dims=n_dims)

    model = None

    if _DEBUG:
        cost_list = np.zeros(len(search_range))
        sc_list = np.zeros(len(search_range))
        for i in search_range:
            if b_kmeans:
                print 'number of clusters:', i
                model, silhouette_score = clustering.kmeans(data, n_clusters=i, n_iter=100)
                print 'Cost:', model.inertia_
                print 'Silhouette:', silhouette_score

                idx = i-search_range[0]
                cost_list[idx] = model.inertia_
                sc_list[idx] = silhouette_score

                # for edges
                dump_cluster_centers(model, dump_path, i, essd)
                # for saliency
                dump_cluster_s_centers(model, dump_path, i, sssd, smap)
            else:
                model, n_clusters = clustering.ap(data)

        # plot the cost and silhouette curve
        _plot_cost_curve(dump_path, cost_list, sc_list)

    else:
        if b_kmeans:
            print 'number of clusters:', n_clusters
            model, silhouette_score = clustering.kmeans(data, n_clusters=n_clusters, n_iter=5000)
            print 'Cost:', model.inertia_
            print 'Silhouette:', silhouette_score
            dump_cluster_centers(model, dump_path, n_clusters, essd)
            dump_cluster_s_centers(model, dump_path, n_clusters, sssd, smap)
        else:
            model, n_clusters = clustering.ap(data)

    if b_dump_model:
        dump_cluster_model(model, scaler_model, pca_model, dump_path, n_clusters, b_kmeans=b_kmeans, b_scale=b_scale, b_pca=b_pca)
        if b_pca:
            # visualize pca
            dump_pca_components(pca_model, dump_path)
            dump_s_pca_components(pca_model, dump_path)


    return model, n_clusters


def perform_clustering(dataset_path, dump_path, n_clusters=10, b_kmeans=True, b_dump_model=False, b_scale=True, b_normalize=True, b_pca=True, search_range=range(2,10), n_dims=100, clean=True):

    global master_dump_path, _DEBUG
    master_dump_path = dump_path
    assert os.path.exists(master_dump_path)

    if _DEBUG:
        dump_path += 'scene_categories_debug/'
    else:
        dump_path += 'scene_categories/'

    # check if compilation required
    if not clean and my_utils.dataset_valid(dump_path):
        print '\nScene Categorization Database is up-to-date.\nPass clean=True for fresh compilation\n'
        return
    else:
        my_utils.invalidate_dataset(dump_path)

    # perform clustering
    timer = time.time()
    print 'Performing clustering...'
    model, n_clusters = _cluster(dump_path, n_clusters, b_kmeans=b_kmeans, b_dump_model=b_dump_model, b_scale=b_scale, b_normalize=b_normalize, b_pca=b_pca, search_range=search_range, n_dims=n_dims)
    print 'Clustering done. '
    print 'Total running time ', time.time() - timer

    # dump cluster images
    timer = time.time()
    print 'Dumping cluster images...'
    if not _DEBUG:
        dump_cluster_images(dataset_path, dump_path, model, n_clusters, b_kmeans)

    print 'Dumping done.'
    print 'Total running time ', time.time() - timer

    my_utils.validate_dataset(dump_path)


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print "Usage : dataset_path dump_path"
        sys.exit(0)

    dataset_path = sys.argv[1]
    dump_path = sys.argv[2]

    perform_clustering(dataset_path, dump_path, \
            b_kmeans=True, \
            n_clusters=10, \
            b_dump_model=True, \
            b_scale=True, \
            b_normalize=False, \
            b_pca=True, \
            search_range=range(2,30), \
            n_dims=250, 
            clean=True)


