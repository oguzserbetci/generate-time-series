from sklearn.utils import shuffle
from loaddataset import *
import numpy as np
import argparse
import os

SEED = 0


def resplitdatasets(datasetname, suffix="_ALT", ratio=.70):
    if os.path.exists('UCR_TS_Archive_2015/{0}/{0}{1}_TRAIN'.format(datasetname, suffix)):
        print('WARNING TRAIN FILE EXISTS WITH SUFFIX:', suffix)
    if os.path.exists('UCR_TS_Archive_2015/{0}/{0}{1}_TEST'.format(datasetname, suffix)):
        print('WARNING TEST FILE EXISTS WITH SUFFIX:', suffix)
    train_data = np.loadtxt('UCR_TS_Archive_2015/{0}/{0}_TRAIN'.format(datasetname), delimiter=',')
    test_data = np.loadtxt('UCR_TS_Archive_2015/{0}/{0}_TEST'.format(datasetname), delimiter=',')

    data = np.concatenate([train_data, test_data])
    data = shuffle(data, random_state=SEED)
    partition = int(len(data) * ratio)
    train_data = data[:partition]
    test_data = data[partition:]

    print('New training shape:', train_data.shape)
    print('New test shape:', test_data.shape)

    savedataset(datasetname, train_data, suffix=suffix + '_TRAIN')
    savedataset(datasetname, test_data, suffix=suffix + '_TEST')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datasetname', help='Datasetname (=foldername in UCR folder)', type=str)
    parser.add_argument('-r', '--ratio', help='New ratio of training and test dataset', type=float, default=.7)
    parser.add_argument('-o', '--output_suffix', help='Suffix for training and test files created', type=str, default='_ALT')

    args = parser.parse_args()

    if not (args.datasetname):
        parser.error('No dataset name given, add --datasetname')

    resplitdatasets(args.datasetname, args.output_suffix, args.ratio)
