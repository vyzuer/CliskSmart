import numpy as np
import sys
import matplotlib.pyplot as plt
from skimage.measure import block_reduce

def extract_descriptor(p_map, cell_size=(10,10)):

    fd = block_reduce(p_map, block_size=cell_size, func=np.mean)

    return fd.ravel()

def _get_map(p_map, x_pos, y_pos, x_size, y_size):

    p_map[y_pos:y_pos+y_size, x_pos:x_pos+x_size] = 1.0

    return p_map


def get_pixel_map(num_faces=1, positions=[[0.5, 0.5]], sizes=[[0.2, 0.2]], aspect_ratio=(3,4), map_scale=10):
    y_dim, x_dim = map_scale*aspect_ratio[0], map_scale*aspect_ratio[1]
    p_map = np.zeros(shape=(y_dim, x_dim))

    for i in range(num_faces):
        x_pos = int(x_dim*positions[i][0])
        y_pos = int(y_dim*positions[i][1])

        x_size = int(x_dim*sizes[i,0])
        y_size = int(y_dim*sizes[i,1])

        p_map = _get_map(p_map, x_pos, y_pos, x_size, y_size)

    return p_map

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'usage: num_faces file_path'
        sys.exit(0)

    num_faces = int(sys.argv[1])
    data_file = sys.argv[2]

    data = np.loadtxt(data_file)

    positions = data[:,0:2]
    sizes = data[:,2:4]

    pixel_map = get_pixel_map(num_faces, positions, sizes)

    plt.imshow(pixel_map, interpolation='nearest')
    plt.show()
    plt.close('all')
    
