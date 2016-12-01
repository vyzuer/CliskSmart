import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import numpy as np


def analyze(db_path, db_details, out_dir):
    f_db_details = db_path + db_details

    data = np.loadtxt(f_db_details, dtype='string')

    num_images, dim = data.shape

    out_file = out_dir + 'social_media.analysis'
    try:
        os.remove(out_file)
    except OSError:
        pass
    
    fp_out_file = open(out_file, 'a')

    fp_out_file.write('Dataset Size : %d\n' % (num_images))

    tot_views = np.sum(map(int, data[:, 5]))
    min_views = np.min(map(int, data[:, 5]))
    max_views = np.max(map(int, data[:, 5]))
    avg_views = np.mean(map(int, data[:, 5]))
    median_views = np.median(map(int, data[:, 5]))
    fp_out_file.write('Total Views : %d\tMin Views : %d\tMax Views : %d\tMean Views : %f\tMedian Views : %f\n' % (tot_views, min_views, max_views, avg_views, median_views))

    tot_favs = np.sum(map(int, data[:, 6]))
    min_favs = np.min(map(int, data[:, 6]))
    max_favs = np.max(map(int, data[:, 6]))
    avg_favs = np.mean(map(int, data[:, 6]))
    median_favs = np.median(map(int, data[:, 6]))
    fp_out_file.write('Total Favs: %d\tMin Favs: %d\tMax Favs: %d\tMean Favs: %f\tMedian Favs : %f\n' % (tot_favs, min_favs, max_favs, avg_favs, median_favs))

    tot_comments = np.sum(map(int, data[:, 7]))
    min_comments = np.min(map(int, data[:, 7]))
    max_comments = np.max(map(int, data[:, 7]))
    avg_comments = np.mean(map(int, data[:, 7]))
    median_comments = np.median(map(int, data[:, 7]))
    fp_out_file.write('Total Comments : %d\tMin Comments : %d\tMax Comments : %d\tMean Comments : %f\tMedian Comments : %f\n' % (tot_comments, min_comments, max_comments, avg_comments, median_comments))

    num_uniq_users = len(list(set(data[:,1])))
    fp_out_file.write('Number of unique users : %d\n' % (num_uniq_users))

    min_date = np.min([int(x[:4]) for x in data[:, 4]])
    max_date = np.max([int(x[:4]) for x in data[:, 4]])
    fp_out_file.write('Min Year : %d\tMax Year : %d\n' % (min_date, max_date))

    fp_out_file.close()
    
    # plot the histograms for social media data
    x = [int(x[:4]) for x in data[:, 4]]
    hist, bins = np.histogram(x, 16)
    plt.axis([2000, 2015, 0, 1400])
    width = 0.8 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.title('Year Distribution of Photos')
    f_name = out_dir + 'year_dist.png'
    plt.savefig(f_name)
    plt.close()

    x = map(int, data[:, 5])
    hist, bins = np.histogram(x, 5000)
    plt.axis([0, 1000, 0, 4000])
    width = 0.8 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.title('Views Distribution')
    f_name = out_dir + 'views_dist.png'
    plt.savefig(f_name)
    plt.close()

    x = map(int, data[:, 6])
    hist, bins = np.histogram(x, 200)
    plt.axis([0, 200, 0, 500])
    width = 0.8 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.title('Favorites Distribution')
    f_name = out_dir + 'favs_dist.png'
    # plt.show()
    plt.savefig(f_name)
    plt.close()

    x = map(int, data[:, 7])
    hist, bins = np.histogram(x, 200)
    plt.axis([0, 200, 0, 500])
    width = 0.8 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.title('Comments Distribution')
    f_name = out_dir + 'comm_dist.png'
    # plt.show()
    plt.savefig(f_name)
    plt.close()

if __name__ == '__main__':
    
    if len(sys.argv) != 4:
        print "usage: ", __file__, "dataset_path dataset_details_file_name dump_path"
        sys.exit(0)

    db_path = sys.argv[1]
    db_details = sys.argv[2]
    out_dir = sys.argv[3]

    analyze(db_path, db_details, out_dir)

