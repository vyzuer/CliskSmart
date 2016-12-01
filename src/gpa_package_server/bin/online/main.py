import sys
import time
import os
from skimage import io
import gp_recommendation as gp_rec

def main(dataset_path, dump_path, gp_model_path):
    """ Iterate through all the images and 
        generate recommendation for each
    """
    if not os.path.exists(dump_path):
        os.makedirs(dump_path)

    # browsing the directory
    image_dir = dataset_path + "ImageDB/"
    
    image_list = dataset_path + "image.list"
    fp_image_list = open(image_list, 'r')

    for img_name in fp_image_list:
        img_name = img_name.rstrip("\r\n")
        print img_name

        img_src = image_dir + img_name

        timer = time.time()
        
        img = io.imread(img_src)
        gp_rec.pos_rec(img, img_src, dump_path, gp_model_path, visualise=True)
        
        print "Total run time = ", time.time() - timer, '\n'

    fp_image_list.close()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print "Usage : dataset_path dump_path gp_model_path"
        sys.exit(0)

    dataset_path = sys.argv[1]
    dump_path = sys.argv[2]
    gp_model_path = sys.argv[3]
    
    main(dataset_path, dump_path, gp_model_path)

