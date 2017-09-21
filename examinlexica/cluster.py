#! /usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Functions to cluster all sentiments in either the subreddits or the historical
lexica.
For the subreddtis:
    There are three ways to cluster the data, each a using different a feature
    matrix. Each matrix is accessed using 'matrix'.
    matrix        feature for each word
    minimum     sentiment - standard derivation
    maximum     sentiment + standard derivation
    normal      sentiment
    all         minimum, normal, maximum

For the historical data:
    There are currently two historical data sets in use: frequencies and adjectives.
    For more information on the data sets see their specific READMEs (taken from
    socialsent).

The data set must be chosen before clustering.

All results are stored in a specified folder (results option).
'''

import os
import sys
import argparse
import hdbscan
import sklearn.cluster as cluster
import numpy as np
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from examinlexica.original.subreddit_data import SubredditData
from examinlexica.original.historical_data import HistoricalData
from examinlexica.constants import (
    PATH_CLUSTERS,
    HISTORICAL_OPTIONS,
    ACCEPTABLE_OPTIONS
    )

def cluster_data(data, algorithm, args, kwds, name, result_folder):
    '''
    Cluster the given data and save the results in a npy-file

    Arguments:
        data: data to be clusterd
        algorithm: the algorithm used for clustering (method call!)
        args: all arguments to be passed to the algorithm method
        kwds: all  arguments to be passed to the algorithm method via key words
        name: name of the file that will be used to save the result
        result_folder: path to a folder, in which the results are stored (if folder
            does not exists it will be created by the function)
    '''
    results = result_folder + '_results'
    u, s, v = np.linalg.svd(data, full_matrices=False)
    if not os.path.exists(results):
        os.makedirs(results)
    d = TSNE(n_components=3).fit_transform(data)
    vis_x = d[:, 0]
    vis_y = d[:, 1]
    vis_z = d[:, 2]
    fig = plt.figure(figsize=(32,30))
    ax = fig.add_subplot(111, projection='3d')
    labels = algorithm(*args, **kwds).fit_predict(u*s)
    ax.scatter(vis_x, vis_y, vis_z, c=labels, cmap='viridis', s=300)
    plt.savefig('graphs/' + name + '.png')
    np.save(results + '/' + name + '_labels.npy', labels)

def start_cluster(data, result_path, matrix, number_of_clusters=0):
    '''
    Function to start clustering, results are saved in a seperate folder.
    All clustering algorithms are applied to the given data.

    Arguments:
        data: data to be clustered
        result_path: path to the folder, in which the results will be stored
        number_of_clusters:
            number of clusters to be used for KMeans, spectral and agglomerative
            clustering
        matrix:
            whether to use original sentiments (normal), negative values (minimum),
            maximum values (maximum) or all three (all)
    '''
    data = data[matrix][:]
    name = matrix + '_'
    if number_of_clusters:
#   same as old version; running yet agoin would be a waste of time
        cluster_data(
            data,
            cluster.KMeans,
            (),
            {'n_clusters':number_of_clusters},
            name + "Kmeans_"+str(number_of_clusters),
            result_path
        )
        cluster_data(
            data,
            cluster.AgglomerativeClustering,
            (),
            {'n_clusters':number_of_clusters, 'linkage':'average', 'affinity':'canberra'},
            name + 'aggl_' + str(number_of_clusters),
            result_path
        )
    cluster_data(
        data,
        cluster.MeanShift,
        (),
        {'min_bin_freq':2, 'cluster_all':True},
        name + 'meanShift',
        result_path
    )
    cluster_data(
        data,
        hdbscan.HDBSCAN,
        (),
        {
            'min_cluster_size':2,
            'min_samples':1,
            'cluster_selection_method':'leaf',
            'metric':'canberra'
        },
        name + 'HDBSCAN',
        result_path
    )


def clarguments_checks(data, matrix, clusters):
    ''' basic plausability checks of command line arguments '''
    if clusters < 0:
        print('number of clusters cannot be smaller than 0!')
        return False
    else:
        if not matrix:
            print('specify a feature matrix!')
            return False
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'data',
        help='data to be clustered',
        choices=['subreddits', 'adjectives', 'frequencies'],
    )
    parser.add_argument(
        '-r',
        '--results',
        default='./',
        help='folder for the results of clustering'
    )
    parser.add_argument(
        '-c',
        '--clusters',
        default=0,
        help='number of clusters used for aggl. clustering',
        type=int
    )
    parser.add_argument(
        'matrix',
        help='create feature matrix using normal, minimal, maximal or all three values',
        choices=ACCEPTABLE_OPTIONS,
        default=None
    )
    args = vars(parser.parse_args())
    if not clarguments_checks(args['data'], args['matrix'], args['clusters']):
        sys.exit()
    if args['data'] in HISTORICAL_OPTIONS.keys():
        path = HISTORICAL_OPTIONS[args['data']]
        data = HistoricalData(path)
    else:
        data = SubredditData(PATH_CLUSTERS)
    start_cluster(
        data.sentiments,
        args['results'],
        args['matrix'],
        args['clusters']
    )
