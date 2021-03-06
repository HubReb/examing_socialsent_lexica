#! /usr/bin/env python3
# -*- coding: utf-8

'''
Calculates inter and inner cluster distances over each result and draws the
corresponding graph.
'''
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
    Cluster the given data and calculate inner or inter cluster distance

    Arguments:
        data: data to be clusterd
        algorithm: the algorithm used for clustering (method call!)
        args: all arguments to be passed to the algorithm method
        kwds: all  arguments to be passed to the algorithm method via key words
    '''
    u, s, _ = np.linalg.svd(data, full_matrices=False)
    data_unclustered = u*s
    clusterer = algorithm(*args, **kwds).fit(data_unclustered)
    centroids = clusterer.cluster_centers_
    labels = clusterer.labels_
    distances = []
    # comment this out to get inter cluster distances
    for cluster_number, centroid in enumerate(centroids):
        distance = get_distances(data_unclustered, cluster_number, centroid, labels)
        distances.append(distance)
    return np.mean(distances)
    # uncomment this for inter cluster didtances
#    return get_cluster_distances(centroids)

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
    print(len(centroids))
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
    distances = []
    for i in range(2, 200):
        distances.append(
            cluster_data(
                data,
                cluster.KMeans,
                (),
                {'n_clusters':i},
            )
        )
    return distances


if __name__ == '__main__':
    data = SubredditData(PATH_CLUSTERS)
    distances_min = all_distances(data.sentiments['minimum'])
    distances_max = all_distances(data.sentiments['maximum'])
    distances_norm = all_distances(data.sentiments['normal'])
    distances_all = all_distances(data.sentiments['all'])
    distances = [distances_min, distances_max, distances_norm, distances_all]
    plt.figure(figsize=(22, 20), dpi=120)
    subplots = [221, 222, 223, 224]
    matrix = ['minimum', 'maximum', 'normal', 'all']
    for plot in range(len(subplots)):
        plt.subplot(subplots[plot])
        plt.plot([i for i in range(2, 200)], distances[plot])
        plt.title(matrix[plot] + ' values')
        plt.grid(True)
        plt.xlabel('Number of Clusters')
        plt.ylabel('inner centroid distance')
plt.savefig('distances.png')
