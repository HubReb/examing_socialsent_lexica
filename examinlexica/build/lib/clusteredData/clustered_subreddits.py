#! /usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Extract all labels created by the clustering algorithms and store them in a single
ClusteredSubreddits object.
Each cluster can be accessed via view(:number of cluster) key.
'''

import os
import sys
from collections import defaultdict
import numpy as np

from examinlexica.helpers import get_lexica_order
from examinlexica.constants import PATH

class ClusteredSubreddits:
    '''
    An object containing all results of the clustering algorithms in every view.

    Attributes:
        clusters:
            dictionary containing all labels. Access labels using view (feature
            matrix)as key.
            If the number of clusters had to be specified manually use number of
            clusters as second key.
        subreddit_list:
            list of subreddit feature vectors to access order of the labels.
            (number of subreddit in list == number of label in vecto.
    '''
    def __init__(self, path, path_clusters):
        '''
        Initialize object containing all labels of the clustering algorithms.

        Arguments:
            path:
                path to the folder containing results of the clustering algorithms
        '''
        self.clusters = {
            'normal' : {},
            'minimum' : {},
            'maximum' : {},
            'all' : {}
            }
        algorithms = ['meanShift', 'HDBSCAN', 'aggl', 'spectral', 'miniBatchKmeans']
        self.set_clusters(algorithms)
        self.subreddit_list = get_lexica_order(path)
        clustered_data = os.listdir(path_clusters)
        self.get_clusters(clustered_data, path_clusters)

    def set_clusters(self, algorithms):
        ''' Initialize dictionary of clusters to store results of algorithms '''
        for algorithm in algorithms:
            for view, _ in self.clusters.items():
                self.clusters[view][algorithm] = {}


    def get_clusters(self, data, path):
        ''' Access results of clustering algorithms and store them in clusters '''
        path = path + '/'
        for cluster_file in data:
            path_file = path + cluster_file
            view, algorithm, number_of_clusters = cluster_file.split('_')[0:3]
            if number_of_clusters != 'labels.npy':
                self.clusters[view][algorithm][int(number_of_clusters)] = np.load(path_file)
            else:
                self.clusters[view][algorithm] = np.load(path_file)

    def sort_clusters(self, algorithm, view, number_of_clusters=0):
        ''' Return clusters created using algorithm with number_of_clusters clusters'''
        if number_of_clusters == 0:
            return self.clusters[view][algorithm]
        try:
            return self.clusters[view][algorithm][number_of_clusters]
        except KeyError:
            print('No corresponding result found. Did you use clster the data correctly?')
            sys.exit()

    def view_cluster(self, algorithm, view, number_of_clusters=0):
        ''' Return result of algorithm (i. e. clusters) '''
        clusters = self.sort_clusters(algorithm, view, number_of_clusters)
        dictio = defaultdict(list)
        for rank in range(len(self.subreddit_list)):
            dictio[clusters[rank]].append(self.subreddit_list[rank])
        return dictio


    def view_hist_of_cluster(self, algorithm, view, number_of_clusters=0):
        ''' Return np.histogram of labels '''
        array = self.sort_clusters(algorithm, view, number_of_clusters)
        if number_of_clusters != 0:
            return np.histogram(array, bins=number_of_clusters)
        return np.histogram(array)

if __name__ == '__main__':
    clSub = ClusteredSubreddits(PATH, 'subreddits_results')
