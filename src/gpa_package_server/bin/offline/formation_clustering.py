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

import matplotlib.pyplot as plt
from sklearn import preprocessing 
from sklearn.externals import joblib
from sklearn.decomposition import PCA

_DEBUG = True

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
    cl_dump_path = dump_path + 'scene_categories/clusters/'
    create_clusters(cl_dump_path, n_clusters)

    image_dir = dataset_path + "ImageDB/"
    
    image_list = dataset_path + "photo.info"
    img_details = np.loadtxt(image_list, dtype='string')
    n_images, n_dim = img_details.shape

    img_labels = model.labels_

    for i in range(n_images):
        img_name = img_details[i][0]

        img_src = image_dir + img_name
        img_dst = dump_path + 'scene_categories/clusters/' + str(img_labels[i]) + '/' + img_name

        os.symlink(img_src, img_dst)

    # copy cluster centers
    if not b_kmeans:
        # create directories for images
        cl_dump_path = dump_path + 'scene_categories/cluster_centers/'
        create_clusters(cl_dump_path, n_clusters)

        for i in range(n_clusters):
            img_idx = model.cluster_centers_indices_[i]
            img_name = img_details[img_idx][0]

            img_src = image_dir + img_name
            img_dst = dump_path + 'scene_categories/cluster_centers/' + str(img_labels[i]) + '/' + img_name

            os.symlink(img_src, img_dst)

def dump_pca_components(pca, dump_path, n_components=20):

    V = pca.components_

    f_rules = dump_path + 'eigen_rules.png'
    n_rows = (n_components-1)/10 + 1

    # this is considering the aspect ratio 3:4
    n_samples, n_dim = V.shape
    assert n_dim%12 == 0
    scale = np.sqrt(n_dim/12)
    x_dim, y_dim = int(3*scale), int(4*scale)

    for i in range(n_components):

        eigen_rule = V[i].reshape(x_dim, y_dim)

        plt.subplot(n_rows,10,i+1)
        plt.axis('off')
        plt.imshow(eigen_rule, interpolation='nearest')

    plt.savefig(f_rules)
    plt.close()

        
def plot_cluster_mean(X, num, mean_path):

    f_name = mean_path + str(num) + '.png'

    n_rows = (num-1)/10 + 1

    # this is considering the aspect ratio 3:4
    n_samples, n_dim = X.shape
    assert n_dim%12 == 0
    scale = np.sqrt(n_dim/12)
    x_dim, y_dim = int(3*scale), int(4*scale)

    for i in range(num):

        mean = X[i].reshape(x_dim, y_dim)

        plt.subplot(n_rows, 10 , i+1)
        plt.axis('off')
        plt.imshow(mean, interpolation='nearest')

    plt.savefig(f_name)
    plt.close()


def dump_cluster_centers(model, dump_path, num, pca_model, scaler_model, b_scale, b_pca):
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
    
    plot_cluster_mean(X, num, mean_path)
    

def _plot_cost_curve(plot_path, cost, silhouette):

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
    
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    dump_file = plot_path + 'cost_plot.png'
    plt.savefig(dump_file)
    plt.close()


def _cluster(dump_path, n_clusters, b_kmeans=True, b_dump_model=False, b_scale=True, b_normalize=True, b_pca=True, search_range=range(2,10), n_dims=100):

    # dump file for descriptor
    pmap_list = dump_path + 'pixel_map.list'
    cluster_dump_path = dump_path + 'cluster_dump/'
    if not os.path.exists(cluster_dump_path):
        os.makedirs(cluster_dump_path)

    data = np.loadtxt(pmap_list)

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

                cost_list[i-2] = model.inertia_
                sc_list[i-2] = silhouette_score

                dump_cluster_centers(model, cluster_dump_path, i, pca_model, scaler_model, b_scale, b_pca)
            else:
                model, n_clusters = clustering.ap(data)

        # plot the cost and silhouette curve
        _plot_cost_curve(cluster_dump_path, cost_list, sc_list)

    else:
        if b_kmeans:
            print 'number of clusters:', n_clusters
            model = clustering.kmeans(data, n_clusters=n_clusters, n_iter=5000)
            print 'Cost:', model.inertia_
            dump_cluster_centers(model, cluster_dump_path, n_clusters, pca_model, scaler_model, b_scale, b_pca)
        else:
            model, n_clusters = clustering.ap(data)

    if b_dump_model:
        dump_cluster_model(model, scaler_model, pca_model, cluster_dump_path, n_clusters, b_kmeans=b_kmeans, b_scale=b_scale, b_pca=b_pca)
        if b_pca:
            # visualize pca
            dump_pca_components(pca_model, cluster_dump_path, n_components=30)

    return model, n_clusters


def perform_clustering(dump_path, n_clusters=10, b_kmeans=True, b_dump_model=False, b_scale=True, b_normalize=True, b_pca=True, search_range=range(2,10), n_dims=100):

    assert os.path.exists(dump_path)
    form_dump_path = dump_path + 'formation/'
    assert os.path.exists(form_dump_path)

    # perform clustering
    timer = time.time()
    print 'Performing clustering...'
    model, n_clusters = _cluster(form_dump_path, n_clusters, b_kmeans=b_kmeans, b_dump_model=b_dump_model, b_scale=b_scale, b_normalize=b_normalize, b_pca=b_pca, search_range=search_range, n_dims=n_dims)
    print 'Clustering done. '
    print 'Total running time ', time.time() - timer


if __name__ == '__main__':

    if len(sys.argv) != 2:
        print "Usage : dump_path"
        sys.exit(0)

    dump_path = sys.argv[1]

    perform_clustering(dump_path, \
            b_kmeans=True, \
            n_clusters=10, \
            b_dump_model=True, \
            b_scale=True, \
            b_normalize=False, \
            b_pca=True, \
            search_range=range(2,50), \
            n_dims=250)


