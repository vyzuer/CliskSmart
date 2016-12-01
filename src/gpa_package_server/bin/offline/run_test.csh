#!/bin/csh 

set dataset_path = /mnt/project/GroupPhotoDB/DB_Test/
set dump_path = /mnt/project/GroupPhotoDB/DB_Test/DUMP/

set dataset_path = /mnt/project/GroupPhotoDB/test/
set dump_path = /mnt/project/GroupPhotoDB/test/DUMP/

# python main.py $dataset_path $dump_path 

# process the dataset and dump the scene structure
# descriptor for clustering
python scene_decomposition.py ${dataset_path} ${dump_path}

# detection face, full body, upper body, lower body and eyes
# python face_detection.py ${dataset_path} ${dump_path}

# cluster scene types
# python scene_clustering.py ${dataset_path} ${dump_path}

