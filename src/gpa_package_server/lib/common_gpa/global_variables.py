import os, sys

thisdir = os.path.dirname(__file__)
base_dir = os.path.join(thisdir, '../../../../data')

# gpa global variables
gp_dump_path_ = os.path.join(base_dir, 'dumps/gpa/')

# these are for group photography
gp_model_path_ = os.path.join(base_dir, 'gpa/')

# these are for viewpoint recommenadtion
vp_base_dataset_ = "/home/vyzuer/View-Point/DataSet-VPF/"

vpa_base_dir_ = "/home/vyzuer/View-Point/DUMPS/landmark_objects/"

w_url0_ = "http://www.wunderground.com/history/airport/"
w_url1_ = "/DailyHistory.html?format=1"

w_stations_ = "./data/weather_stations.list"

sun_database_ = "/SunMoonDB/global/"
# s_url_ = "http://www.timeanddate.com/sun/singapore/singapore"

__SERVER = True
