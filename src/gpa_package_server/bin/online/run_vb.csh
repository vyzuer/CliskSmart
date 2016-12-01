#!/bin/csh 

set dump_path = /mnt/project/GroupPhotoDB/visual_balance/dump/
set data_path = /mnt/project/GroupPhotoDB/visual_balance/graph/
set graph_id = 1
set nodes_file = nodes.list
set edges_file = edges.list

python visual_balance.py ${data_path}${graph_id}/ $nodes_file $edges_file $dump_path

