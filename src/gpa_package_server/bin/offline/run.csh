#!/bin/csh 

set dataset_path = /mnt/project/GroupPhotoDB/DB/
set dump_path = /mnt/project/GroupPhotoDB/DUMP/

# set dataset_path = /mnt/project/GroupPhotoDB/GP_DB/
# set dump_path = /mnt/project/GroupPhotoDB/GP_DB/DUMP/

python main.py $dataset_path $dump_path

# process the dataset and dump the scene structure
# descriptor for clustering
# input - dataset path, dump path
# output - ssd_grid_size.list in dump path
# python scene_decomposition.py ${dataset_path} ${dump_path}

# aesthetic score evaluation
# python aesthetic_score.py $dataset_path $dump_path

# detection face
# python face_detection.py ${dataset_path} ${dump_path}

# cluster scene types
# python scene_clustering.py ${dataset_path} ${dump_path}

# position analysis
# python position_modeling.py ${dataset_path} ${dump_path}

# formation analysis
# python pixel_map_formation.py ${dataset_path} ${dump_path}
# python formation_clustering.py ${dump_path}

