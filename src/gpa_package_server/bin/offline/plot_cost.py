import matplotlib.pyplot as plt
import numpy as np
import sys

def plot_cost(silhouette_file, cost_file, dump_file):
    fig, ax1 = plt.subplots()

    silhouette = np.loadtxt(silhouette_file)
    cost = np.loadtxt(cost_file)

    num_points = len(cost)
    x = np.arange(num_points)
    x += 1 # cluster starting from 2

    ax1.plot(x , silhouette, 'b-')
    ax1.set_xlabel('No. of clusters')
    ax1.set_ylabel('Silhouette Coefficient', color='b')
    for tl in ax1.get_yticklabels():
        tl.set_color('b')

    ax2 = ax1.twinx()
    ax2.plot(x, cost, 'r.')
    ax2.set_ylabel('Cost', color='r')
    for tl in ax2.get_yticklabels():
        tl.set_color('r')

    plt.savefig(dump_file)
    plt.show()
    plt.close()
    

if __name__ == '__main__':

    silhouette_file = 's.list'
    cost_file = 'cost.list'
    if len(sys.argv) == 3:
        silhouette_file = sys.argv[1]
        cost_file = sys.argv[2]
        
    dump_file = 'cost_sil.png'

    plot_cost(silhouette_file, cost_file, dump_file)

