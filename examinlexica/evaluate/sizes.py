#! /usr/bin/env python3
# -*- coding: utf-8 -*-

''' Calculate average cluster size and visualise graphe '''

import os
import sys
import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from examinlexica.constants import PATH_CLUSTERS

# constants
CLUSTER_SIZE = {'norm':{'AGGL':[], 'Kmeans':[]}, 'min':{'AGGL':[], 'Kmeans':[]},
                'max':{'AGGL':[], 'Kmeans':[]}, 'all':{'AGGL':[], 'Kmeans':[]}}
FILENAME = 'cosine_euclidean_svd_single/comparision_labels_'

VIEWS = ['normal', 'minimum', 'all', 'maximum']
RANGE = [i for i in range(20, 106)]

def get_clustered_subreddits(numbers, subreddits, algorithms, views):
    ''' Extract clusters from file ans return average cluster size '''
    clusters = []
    for view in VIEWS:
        matrix = views[view]
        for algorithm in ['AGGL', 'Kmeans']:
            clusters = get_data(FILENAME + view + str(numbers), algorithm)
            alg = algorithms[algorithm]
            size_clusters = []
            if clusters == []:
                size_clusters.append(0)
            for cluster in clusters:
                size_clusters.append(len(cluster))
            CLUSTER_SIZE[matrix][algorithm].append(np.mean(size_clusters))
    return np.mean(size_clusters)

def get_subreddits():
    ''' Return subreddit names '''
    subreddits = {}
    for filename in os.listdir(PATH_CLUSTERS):
        if filename.endswith('tsv'):
            subreddits[filename] = 0
    return subreddits

def get_data(filename, algorithm):
    ''' Returns list of lists of subreddits created by the alg. Each list is a cluster. '''
    clusters = []
    algorithms = ['AGGL', 'HDBSCAN', 'Kmeans']
    try:
        with open(filename) as f:
            data = f.read().split('\n')
    except FileNotFoundError:
        print('File ', filename, 'does not exist!')
        sys.exit()
    start_point = False
    for cluster in data:
        if '__' in cluster:
            continue
        if cluster in algorithms:
            if cluster not in algorithm:
                start_point = False
                continue
            else:
                start_point = True
                continue
        if start_point:
            try:
                number, datapoints = cluster.split(':')
                if number.endswith('-1'):
                    datapoints = datapoints.split(', ')
                    for d in datapoints:
                        clusters.append([d])
                else:
                    clusters.append([d.strip(',') for d in datapoints.split()])
            except ValueError:
                continue        # empty line, can be ignored
    return clusters

views = {'normal' : 'norm', 'minimum':'min', 'maximum':'max', 'all':'all'}
algorithms = {'AGGL' : 'aggl', 'Kmeans':'kmeans', 'HDBSCAN':'hdbscan'}
reddits = {}
size_clusters = []
clusters = RANGE[:]
matrix = list(views.values())

if __name__ == '__main__':
    subplots = [221, 222, 223, 224]
    for cluster_number in clusters:
        subreddits = get_subreddits()
        size = get_clustered_subreddits(cluster_number, subreddits, algorithms, views)
    for number, plot in enumerate(subplots):
        for algorithm in ['AGGL', 'Kmeans']:
            plt.subplot(plot)
            plt.plot(clusters, CLUSTER_SIZE[matrix[number]][algorithm], label=algorithm)
            plt.title(matrix[number] + ' values')
            plt.xlabel('Number of Clusters')
            plt.ylabel('data points in cluster')
        plt.grid(True)
        plt.legend()
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.45,
                    wspace=0.4)
    plt.savefig('subreddits_in_cluster.png', bbox_inches='tight')
