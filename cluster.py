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
import hdbscan
import sklearn.cluster as cluster
import numpy as np

from subreddit_data import SubredditData
from constants import PATH

def cluster_data(data, algorithm, args, kwds, name):
    """
    Cluster the subreddits and save results in a npy-file

    Arguments:
        algorithm: the algorithm used for clustering (method call!)
        args: all arguments to be passed to the algorithm method
        kwds: all  arguments to be passed to the algorithm method
        name: name of the file that will be used to save the result
        view: sentiment view (normal, maximum, normal, all)
    """
    labels = algorithm(*args, **kwds).fit_predict(data)
    np.save("results/"+name+"_labels.npy", labels)

def start_cluster(data, view, times):
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
    for number in range(2,times+1):
        # too many dimensions to suse KMeans effectevly
        cluster_data(
            data[view],
            cluster.MiniBatchKMeans,
            (),
            {'n_clusters':number, 'batch_size':350},
            view + "_miniBatchKmeans_"+str(number)
        )
        cluster_data(
            data[view],
            cluster.AgglomerativeClustering,
            (),
            {'n_clusters':number, 'linkage':'average', 'affinity':'canberra'},
            view + "_aggl_" + str(number)
        )
    cluster_data(
        data[view],
        cluster.MeanShift,
        (),
        {'min_bin_freq':2, 'cluster_all':True},
        view + "_meanShift"
    )
    cluster_data(
        data[view],
        hdbscan.HDBSCAN,
        (),
        {'min_cluster_size':2, 'min_samples':1, 'cluster_selection_method':'leaf', 'metric':'canberra'},
        view + "_HDBSCAN"
    )

if __name__ == '__main__':
    subreddits = SubredditData(PATH)
    start_cluster(subreddits.sentiments, "normal", 200)
    start_cluster(subreddits.sentiments, "minimum", 200)
    start_cluster(subreddits.sentiments, "maximum", 200)
    start_cluster(subreddits.sentiments, "all", 200)
