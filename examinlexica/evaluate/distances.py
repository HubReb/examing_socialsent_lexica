#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from random import randint

import numpy as np
import sklearn.cluster as cluster
import matplotlib
from examinlexica.original.subreddit_data import SubredditData
from examinlexica.original.historical_data import HistoricalData
from examinlexica.constants import (
    PATH_CLUSTERS,
    HISTORICAL_OPTIONS,
    ACCEPTABLE_OPTIONS
    )
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def cluster_data(data, algorithm, args, kwds):
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
    u, s, _ = np.linalg.svd(data, full_matrices=False)
    data_unclustered = u*s
    clusterer = algorithm(*args, **kwds).fit(data_unclustered)
    centroids = clusterer.cluster_centers_
    labels = clusterer.labels_
    distances = []
    for cluster_number, centroid in enumerate(centroids):
        distance = get_distances(data_unclustered, cluster_number, centroid, labels)
        distances.append(distance)
    return np.mean(distances), get_cluster_distances(centroids)

def get_distances(data, cl_number, centroid, labels):
    ''' Calculate mean distance between centroid and its datapoints '''
    distances = [np.linalg.norm(point-centroid) for point in data[labels == cl_number]]
    if not distances:
        labels[randint(0, len(data)-1)] = cl_number
        distances = [np.linalg.norm(point-centroid) for point in data[labels == cl_number]]
    return np.mean(distances)

def get_cluster_distances(centroids):
    ''' Calculate inter cluster distance '''
    distances = []
    for centroid in centroids:
        dist = []
        for c in centroids:
            if np.all(c == centroid):
                continue
            dist.append(np.linalg.norm(c - centroid))
        distances.append(np.mean(dist))
    return np.mean(distances)

def all_distances(data):
    ''' Return all distances '''
    inner_distances = []
    inter_distances = []
    for i in range(2, 100):
        distance = cluster_data(
                        data,
                        cluster.KMeans,
                        (),
                        {'n_clusters':i},
                    )

        inner_distances.append(distance[0])
        inter_distances.append(distance[1])
    return inner_distances, inter_distances


if __name__ == '__main__':
    data = SubredditData(PATH_CLUSTERS)
#    inner_distances_min, inter_distances_min= all_distances(data.sentiments['minimum'])
#    inner_distances_max, inter_distances_max = all_distances(data.sentiments['maximum'])
    inner_distances_norm, inter_distances_norm = all_distances(data.sentiments['normal'])
#    inner_distances_all, inter_distances_all = all_distances(data.sentiments['all'])
#    inner_distances = [inner_distances_min, inner_distances_max, inner_distances_norm, inner_distances_all]
#    inter_distances = [inter_distances_min, inter_distances_max, inter_distances_norm, inter_distances_all]
    subplots = [221, 222, 223, 224]
    matrix = ['minimum', 'maximum', 'normal', 'all']
    plt.plot([i for i in range(2,100)], inner_distances_norm)
#    for plot in range(len(subplots)):
#        plt.subplot(subplots[plot])
    plt.title('normal values')
    plt.grid(True)
    plt.xlabel('Number of Clusters')
    plt.ylabel('inner centroid distance')
    plt.savefig('inner_cluster_distances.png')
    plt.close()
    plt.plot([i for i in range(2,100)], inter_distances_norm)
    plt.title('normal values')
    plt.grid(True)
    plt.xlabel('Number of Clusters')
    plt.ylabel('inter centroid distance')
    plt.savefig('inter_cluster_distances.png')
