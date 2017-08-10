import numpy as np
from collections import Counter, namedtuple

SEED = 0
np.random.seed = SEED


def loaddataset(name, train_suffix='_TRAIN', test_suffix='_TEST'):
    data = np.loadtxt('UCR_TS_Archive_2015/{0}/{0}{1}'.format(name, train_suffix), delimiter=',')
    test_data = np.loadtxt('UCR_TS_Archive_2015/{0}/{0}{1}'.format(name, test_suffix), delimiter=',')

    # get labels
    min_label = min(data[:,0])
    labels = np.array(data[:,0], dtype=int) - min_label
    min_label = min(test_data[:,0])
    test_labels = np.array(test_data[:,0], dtype=int) - min_label

    # remove labels
    data = data[:,1:]
    test_data = test_data[:,1:]

    class_dist = Counter(labels)

    print('Dataset {} has been loaded with class distribution of: {}, test:{}'.format(name, class_dist, len(test_labels)))
    K = len(class_dist)
    T = data.shape[1]
    N = len(data)

    r = namedtuple('dataset', 'data, labels, class_dist, test_data, test_labels, T, N, K')
    return r(data, labels, class_dist, test_data, test_labels, T, N, K)


def savedataset(dataset_name, data, labels=None, suffix=''):
    if labels is not None:
        data = np.concatenate([labels[:,np.newaxis], data], axis=1)
    fname = 'UCR_TS_Archive_2015/{0}/{0}{1}'.format(dataset_name, suffix)
    f = open(fname, 'ab')
    np.savetxt(f, data, fmt='%g', delimiter=',')
    f.close()
