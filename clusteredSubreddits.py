#! /usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Extract all labels created by the clustering algorithms and store them in a single
ClusteredSubreddits object.
Each cluster can be accesed via view(:number of cluser) key.
For now, no specific comparision method is implemented
'''

import numpy as np
import os
from collections import defaultdict

class ClusteredSubreddits:
    '''
    An object containing all results of the clustering algorithms in every view.

    Attributes:
        clusters:
            dictionary containing all labels. Access labels using view as key.
            If the number of clusters had to be specified manually use number of
            clusters as second key.
        subreddit_list:
            list of subreddit feature vectors to access order of the labels.
            (number of subreddit in list == number of label in vecto.
    '''
    def __init__(self, path):
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
        self.order = self.get_order()
        self.set_clusters(algorithms)
        path = path + '/results'
        clustered_data = os.listdir(path)
        self.get_clusters(clustered_data, path)
        self.subreddit_list = self.get_subreddit_order()

    def set_clusters(self, algorithms):
        ''' Initialize dictionary of clusters to store results of algorithms '''
        for algorithm in algorithms:
            for view in self.clusters.keys():
                self.clusters[view][algorithm] = {}

    def get_order(self):
        with open('order_subreddits.txt') as f:
            return f.read().split('\n')

    def get_clusters(self, data, path):
        ''' Access results of clustering algorithms and store them clusters '''
        path = path + '/'
        for cluster_file in data:
            path_file = path + cluster_file
            view, algorithm, number_of_clusters = cluster_file.split('_')[0:3]
            if number_of_clusters != 'labels.npy':
                self.clusters[view][algorithm][int(number_of_clusters)] = np.load(path_file)
            else:
                self.clusters[view][algorithm] = np.load(path_file)

    def sort_clusters(self, algorithm, view, number_of_clusters=0):
        if number_of_clusters == 0:
            return self.clusters[view][algorithm]
        return self.clusters[view][algorithm][number_of_clusters]

    def view_cluster(self, algorithm, view, number_of_clusters=0):
        ''' Return result of algorithm (i. e. clusters) '''
        clusters = self.sort_clusters(algorithm, view, number_of_clusters)
        dictio = defaultdict(list)
        for d in range(len(self.order)):
            dictio[clusters[d]].append(self.order[d])
        return dictio


    def view_hist_of_cluster(self, algorithm, view, number_of_clusters=0):
        ''' Return np.histogram of labels '''
        array = self.sort_clusters(algorithm, view, number_of_clusters)
        if number_of_clusters != 0:
            return np.histogram(array, bins=number_of_clusters)
        return np.histogram(array)


    def get_subreddit_order(self):
        ''' Return order of the feature vectors in matrix '''
        with open('order_subreddits.txt') as f:
            order = f.read().split('\n')
        return order

    def show_word_order(self):
        with open('words.txt') as f:
            words = f.read()
        return words


