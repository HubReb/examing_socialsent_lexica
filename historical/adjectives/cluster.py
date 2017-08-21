#! /usr/bin/env python3
# -*- coding: utf-8 -*-

''' Functions to cluster all hist_adj in three ways using different feature vectors.
    view (feature for each word):
    - minimum (sentiment - standard derivation)
    - maximum (sentiment + standard derivation)
    - normal (sentiment)
    - all (minimu, normal, maximum)
'''

import os
import hdbscan
import sklearn.cluster as cluster
import numpy as np

from historical_data import HistoricalData
from constants import PATH_HISTORICAL_ADJECTIVES

def cluster_data(data, algorithm, args, kwds, name):
    """
    Cluster the hist_adj and save results in a npy-file

    Arguments:
        algorithm: the algorithm used for clustering (method call!)
        args: all arguments to be passed to the algorithm method
        kwds: all  arguments to be passed to the algorithm method
        name: name of the file that will be used to save the result
        view: sentiment view (normal, maximum, normal, all)
    """
    labels = algorithm(*args, **kwds).fit_predict(data)
    np.save("results/"+name+"_labels.npy", labels)

def start_cluster(data, times):
    """
    Function to start clustering, resutls are saved in a seperate file

    Arguments:
        view: whether to use original sentiments (normal), negative values (minimum),
            maximum values (maximum) or all three (all)
        times:
            maximum number of clusters to be used for KMeans, spectral and
            agglomerative clustering
        """
    if not os.path.exists("results"):
        os.makedirs("results")
    for number in range(2,times):
        # too many dimensions to suse KMeans effectevly
        cluster_data(
            data,
            cluster.MiniBatchKMeans,
            (),
            {'n_clusters':number, 'batch_size':350},
            "miniBatchKMeans_"+str(number)
        )
        cluster_data(
            data,
            cluster.AgglomerativeClustering,
            (),
            {'n_clusters':number, 'linkage':'average', 'affinity':'canberra'},
            "aggl_" + str(number)
        )
    cluster_data(
        data,
        cluster.MeanShift,
        (),
        {'min_bin_freq':2, 'cluster_all':True},
        "meanShift"
    )
    cluster_data(
        data,
        hdbscan.HDBSCAN,
        (),
        {'min_cluster_size':2, 'min_samples':1, 'cluster_selection_method':'leaf', 'metric':'canberra'},
        "HDBSCAN"
    )

if __name__ == '__main__':
    hist_adj = HistoricalData(PATH_HISTORICAL_ADJECTIVES)
    start_cluster(hist_adj.sentiments, 8)
