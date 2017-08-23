# Synthetic time series generation for training simple multi-layer-perceptron classifier
This project explores using dynamic-time-warping (DTW) and Stochastic Subgradient averaging (SSG) for synthetic time-series data generation.

See the general [overview of methods and evaluation](https://oguzserbetci.github.io/generate-time-series/).

To test the scripts, download [UCR](http://www.cs.ucr.edu/~eamonn/time_series_data/) dataset in the same folder as the scripts.

Use python3 and install required packages:
`pip3 install -r requirements.txt`

To use the scripts with your own data, use the `expand_data_set` function in `spawn.py`.

# generate new dataset
Generate new data points for given data set.
```bash
$ python3 spawn.py -h
usage: spawn.py [-h] [-d DATASETNAME] [-r N_REPS] [-b N_BASE] [-k K]
                [-s SSG_EPOCHS] [-i INPUT_SUFFIX] [-o OUTPUT_SUFFIX]

optional arguments:
  -h, --help            show this help message and exit
  -d DATASETNAME, --datasetname DATASETNAME
                        Datasetname (=foldername in UCR folder)
  -r N_REPS, --n_reps N_REPS
                        Number of iterations for the complete procedure
  -b N_BASE, --n_base N_BASE
                        Number of data-points to average for creating one new
                        data-point
  -k K, --k K           Number of iterations for K-means clustering
  -s SSG_EPOCHS, --ssg_epochs SSG_EPOCHS
                        Number of iterations for mean calculation with SSG
  -i INPUT_SUFFIX, --input_suffix INPUT_SUFFIX
                        suffix for file to be extended
  -o OUTPUT_SUFFIX, --output_suffix OUTPUT_SUFFIX
                        suffix for created training and test files

$ python3 spawn.py --datasetname=str --n_rep=int --n_base=int --k=int --ssg_epochs=int --input_suffix=str --output_suffix=str
# Example
$ python3 spawn.py --datasetname=InlineSkate --n_reps=10 --n_base=2 --k=1 --ssg_epochs=1 --input_suffix=TRAIN --output_suffix=EXP_TRAIN
# or short:
$ python3 spawn.py -d=InlineSkate -r=10 -b=2 -k=1 -s=1 -i=TRAIN -o=EXP_TRAIN
```

# resplit dataset
Resplit the training and test set to a given ratio
```bash
$ python3 resplit.py -h
usage: resplit.py [-h] [-d DATASETNAME] [-r RATIO] [-o OUTPUT_SUFFIX]

optional arguments:
  -h, --help            show this help message and exit
  -d DATASETNAME, --datasetname DATASETNAME
                        Datasetname (=foldername in UCR folder)
  -r RATIO, --ratio RATIO
                        New ratio of training and test dataset
  -o OUTPUT_SUFFIX, --output_suffix OUTPUT_SUFFIX
                        Suffix for training and test files created

$ python3 resplit.py --datasetname=str --ratio=float --input_suffix=str
# Example
$ python3 resplit.py --datasetname=ArrowHead --ratio=.7 --input_suffix=_ALT70
# or short:
$ python3 resplit.py -d=ArrowHead -r=.7 -i=_ALT70
```
