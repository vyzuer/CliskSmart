#!/bin/csh

#set dump_path = "/home/yogesh/Project/Flickr-YsR/small_merlion/ImageDB/"
#set dataset_path = "/home/yogesh/Project/Flickr-YsR/merlionDB_1/DB/0/ImageDB/"

# set dump_path = "/home/yogesh/Project/Flickr-YsR/merlionImages/top_N_images/ImageDB_640/"
# set dataset_path = "/home/yogesh/Project/Flickr-YsR/merlionImages/top_N_images/ImageDB/"

# set dataset_path = "/home/yogesh/Copy/Flickr-code/PhotographyAssistance/testing/images/"
# set dump_path = "/home/yogesh/Copy/Flickr-code/PhotographyAssistance/testing/images_0/"
# set dataset_path = "/home/yogesh/Project/Flickr-YsR/floatMarina/ImageDB/"
# set dump_path = "/home/yogesh/Project/Flickr-YsR/floatMarina/ImageDB_640/"

set dataset_path = /mnt/project/GroupPhotoDB/results/ImageDB1/
set dump_path = /mnt/project/GroupPhotoDB/results/ImageDB/

set dataset_path = /home/vyzuer/Copy/Research/user-study/gpa/results/syn/more/
set dump_path = /home/vyzuer/Copy/Research/user-study/gpa/syn/more/

mkdir -p $dump_path

# set dump_path = "./temp/"
# set dataset_path = "./temp/images/"

# resizing images in dataset
python resize_image.py $dataset_path $dump_path

