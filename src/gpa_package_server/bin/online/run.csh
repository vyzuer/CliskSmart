#!/bin/csh 

set dataset_path = /home/vyzuer/work/data/gpa/test/
set dump_path = /home/vyzuer/work/data/gpa/test/dump/
set gp_dump_path = /home/vyzuer/work/data/server/server/data/gpa/

python main.py $dataset_path $dump_path $gp_dump_path

