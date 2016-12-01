import os

def touch(fname, times=None):
    with open(fname, 'a'):
        os.utime(fname, times)

def dataset_valid(dump_path):

    v_file = dump_path + '.valid'
    valid = False

    if os.path.isfile(v_file):
        valid = True
    else:
        valid = False

    return valid

def validate_dataset(dump_path):

    v_file = dump_path + '.valid'

    touch(v_file)
    

def invalidate_dataset(dump_path):

    v_file = dump_path + '.valid'
    try:
        os.remove(v_file)
    except OSError:
        pass

