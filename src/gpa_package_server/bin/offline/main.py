import sys
import scene_decomposition as sd
import face_detection as fd
import scene_clustering as sc
import position_modeling as pm
import pixel_map_formation as pmf
import formation_clustering as fc
import aesthetic_score as a_score
import analyze_face_info as afi

def main(dataset_path, dump_path, clean=False):

    # process the dataset and dump the scene structure
    # descriptor for clustering
    # input - dataset path, dump path
    # output - ssd_grid_size.list in dump path
    sd.process_dataset(dataset_path, dump_path, clean=clean)

    # compute aesthetic score
    a_score.compute(dataset_path, dump_path, clean=clean)

    # dump features related to face in the image frame
    fd.process_dataset(dataset_path, dump_path, clean=clean)

    # scene categorization
    sc.perform_clustering(dataset_path, dump_path, \
            b_kmeans=True, \
            n_clusters=10, \
            b_dump_model=True, \
            b_scale=True, \
            b_normalize=False, \
            b_pca=True, \
            search_range=range(5,20), \
            n_dims=250, \
            clean=clean)


    # analyze the face details of the dataset
    afi.analyze(dataset_path, dump_path, clean=clean)

    # position modeling
    pm.perform_modeling(dataset_path, dump_path, \
            n_components_range=range(2, 20), \
            b_dump_model=True, \
            b_scale=True, \
            b_num_faces=True, \
            b_face_size=True, \
            n_iter=5000, \
            clean=clean)

    """
    # pixel map formation
    pmf.process(dataset_path, dump_path)

    # formation clustering
    fc.perform_clustering(dump_path, \
            b_kmeans=True, \
            n_clusters=10, \
            b_dump_model=True, \
            b_scale=True, \
            b_normalize=False, \
            b_pca=True, \
            search_range=range(2,50), \
            n_dims=250)
    """

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "Usage : dataset_path dump_path"
        sys.exit(0)

    dataset_path = sys.argv[1]
    dump_path = sys.argv[2]
    
    main(dataset_path, dump_path, clean=False)

