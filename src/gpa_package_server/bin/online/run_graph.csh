#!/bin/csh

set data_path = /home/vyzuer/Copy/Research/Project/code/group-photography/gpa_package/bin/data/1/
set nodes_list = nodes.list
set edges_list = edges.list

# visualize the graph model
# python visualize_optimization.py $data_path $nodes_list $edges_list

set img_db = /mnt/project/GroupPhotoDB/test/
set dump_path = /mnt/project/GroupPhotoDB/test/color_energy/
# visualize color energy
python visualize_color_energy.py $img_db $dump_path
