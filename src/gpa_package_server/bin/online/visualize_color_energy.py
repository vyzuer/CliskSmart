import sys
import time

# add the package
import _mypath

import segmentation.salient_object_detection as sod

def process(db_path, dump_path):

    image_dir = db_path + 'ImageDB/'
    img_list = db_path + 'photo.info'
    fp_image_list = open(img_list, 'r')

    for image_name in fp_image_list:
        image_name = image_name.rstrip("\n")
        img = image_dir + image_name

        timer = time.time()
        sobj = sod.SalientObjectDetection(img, segment_dump=dump_path)
        print "\nTotal time:", time.time() - timer
        print

    fp_image_list.close()


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print "Usage : img_db dump_path"
        sys.exit(0)

    img_db = sys.argv[1]
    dump_path = sys.argv[2]

    process(img_db, dump_path)

