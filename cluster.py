#! /usr/bin/env python3
# -*- coding: utf-8 -*-

''' Functions to cluster all subreddits in three ways using different feature vectors.
    view (feature for each word):
    - minimum (sentiment - standard derivation)
    - maximum (sentiment + standard derivation)
    - normal (sentiment)
    - all (minimu, normal, maximum)
'''

import os
import sys
import hdbscan
import sklearn.cluster as cluster
import numpy as np

from subreddit_data import SubredditData
from historical_data import HistoricalData
from constants import (
                        PATH,
                        HISTORICAL_OPTIONS
                        )


def cluster_data(data, algorithm, args, kwds, name, result_folder):
    """
    Cluster the hist_adj and save results in a npy-file

    Arguments:
        algorithm: the algorithm used for clustering (method call!)
        args: all arguments to be passed to the algorithm method
        kwds: all  arguments to be passed to the algorithm method
        name: name of the file that will be used to save the result
        view: sentiment view (normal, maximum, normal, all)
    """
    results = result_folder + "results"
    if not os.path.exists(results):
        os.makedirs(results)
    labels = algorithm(*args, **kwds).fit_predict(data)
    np.save(results + "/" + name + "_labels.npy", labels)


def start_cluster(data, result_path, times=0, view=None):
    """
    Function to start clustering, resutls are saved in a seperate file

    Arguments:
        view: whether to use original sentiments (normal), negative values (minimum),
            maximum values (maximum) or all three (all)
        times:
            maximum number of clusters to be used for KMeans, spectral and
            agglomerative clustering
        """
    if view:
        data = data[view]
    else:
        view = ''
    for number in range(2,times+1):
        # too many dimensions to suse KMeans effectevly
    #    cluster_data(
    #        data[view],
    #        cluster.MiniBatchKMeans,
    #        (),
    #        {'n_clusters':number, 'batch_size':350},
    #        view + "_miniBatchKmeans_"+str(number)
    #    )
        cluster_data(
            data,
            cluster.AgglomerativeClustering,
            (),
            {'n_clusters':number, 'linkage':'average', 'affinity':'canberra'},
            view + "_aggl_" + str(number),
            result_path
        )
    cluster_data(
        data,
        cluster.MeanShift,
        (),
        {'min_bin_freq':2, 'cluster_all':True},
        view + "_meanShift",
        result_path
    )
    cluster_data(
        data,
        hdbscan.HDBSCAN,
        (),
        {'min_cluster_size':2, 'min_samples':1, 'cluster_selection_method':'leaf', 'metric':'canberra'},
        view + "_HDBSCAN",
        result_path
    )

if __name__ == '__main__':
    if len(sys.argv) < 2:
        subreddits = SubredditData(PATH)
        start_cluster(subreddits.sentiments, "", 200, "normal")
        start_cluster(subreddits.sentiments, "", 200, "minimum")
        start_cluster(subreddits.sentiments, "", 200, "maximum")
        start_cluster(subreddits.sentiments, "", 200, "all")
    else:
        if sys.argv[1] not in HISTORICAL_OPTIONS.keys():
            print('usage: python3 cluster.py adjectives|frequencies')
            sys.exit()
        path = HISTORICAL_OPTIONS[sys.argv[1]]
        hist_adj = HistoricalData(path)
        start_cluster(hist_adj.sentiments, 8, path)
