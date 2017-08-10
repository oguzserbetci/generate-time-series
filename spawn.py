from sklearn.metrics.pairwise import pairwise_distances
from loaddataset import loaddataset, savedataset
from collections import Counter
from functools import partial
from fastdtw import fastdtw
from dtw_mean import ssg
import numpy as np
import argparse
import time
import math
import os

SEED = 0
np.random.seed = SEED


def fastdtw_(x, y):
    return fastdtw(x.copy(), y.copy())[0]


def create_new_data(data, count, k=1, ssg_epochs=None):
    cluster_ind = np.random.choice(len(data), size=count, replace=False)
    clusters = data[cluster_ind]
    alloc = np.zeros(len(data))

    # K-Means clustering
    for _ in range(k):
        dists = pairwise_distances(data, clusters, metric=fastdtw_, n_jobs=4)
        alloc = dists.argmin(1)

        new_clusters = []
        for j, cluster in enumerate(clusters):
            if list(alloc).count(j) < 2:
                alloc[alloc == j] = -1
                continue
            d = data[alloc == j]
            z = ssg(d[:,np.newaxis], return_f=False, n_epochs=ssg_epochs)
            new_clusters.append(z[0])
        clusters = np.array(new_clusters)
    return clusters, alloc


def expand_data_set(data, labels, n_reps, n_base, k=1, ssg_epochs=None, callback=None):
    old_data_offset = len(data)
    for i in range(n_reps):
        for label in Counter(labels).keys():
            l_data = data[labels == label]
            count = math.ceil(len(l_data) / n_base)
            new_clusters, _ = create_new_data(l_data, count + 1, k, ssg_epochs)
            if new_clusters.size:
                new_labels = np.ones(len(new_clusters)) * label
                callback(data=new_clusters, labels=new_labels)
                print('{} new data points for label {}'.format(len(new_clusters), label))
                data = np.concatenate([data, new_clusters])
                labels = np.concatenate([labels, new_labels])
    return data[old_data_offset:], labels[old_data_offset:]


# see argument documentation
def spawn(datasetname, n_reps, n_base=4, k=1, ssg_epochs=None, input_suffix='_TRAIN', output_suffix='_EXP_TRAIN'):
    if os.path.exists('UCR_TS_Archive_2015/{0}/{0}{1}'.format(datasetname, output_suffix)):
        print('WARNING FILE EXISTS WITH SUFFIX:', output_suffix)
    data, labels, class_dist, _, _, _, N, K = loaddataset(datasetname, input_suffix)
    print('expanding {} from {} datapoints, with class distribution of: {}'.format(datasetname, len(data), class_dist))
    print('upper bound for data-points generated:', (N / n_base) * n_reps)
    start = time.time()
    save = partial(savedataset, suffix=output_suffix, dataset_name=datasetname)
    expanded_data_set, expanded_labels = expand_data_set(data, labels, n_reps, n_base, k, ssg_epochs, save)
    duration = time.time() - start
    print('expanded {} to {} datapoints in {} minutes, class distribution is: {}'.format(datasetname, len(expanded_data_set), duration / 60, Counter(expanded_labels)))
    savedataset(datasetname, expanded_data_set, expanded_labels, output_suffix + '.bak')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datasetname', help='Datasetname (=foldername in UCR folder)', type=str)
    parser.add_argument('-r', '--n_reps', help='Number of iterations for the complete procedure', type=int, default=1)
    parser.add_argument('-b', '--n_base', help='Number of data-points to average for creating one new data-point', type=int, default=2)
    parser.add_argument('-k', '--k', help='Number of iterations for K-means clustering', type=int, default=1)
    parser.add_argument('-s', '--ssg_epochs', help='Number of iterations for mean calculation with SSG', type=int, default=None)
    parser.add_argument('-i', '--input_suffix', help='suffix for file to be extended', type=str, default='_TRAIN')
    parser.add_argument('-o', '--output_suffix', help='suffix for created training and test files', type=str, default='_EXP_TRAIN')

    args = parser.parse_args()

    if not (args.datasetname):
        parser.error('No dataset name given, add --datasetname')

    spawn(args.datasetname, args.n_reps, args.n_base, args.k, args.ssg_epochs, args.input_suffix, args.output_suffix)
