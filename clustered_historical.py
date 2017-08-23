#! /usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Extract all labels creted by the clustering algorithms and store them in a
single object.
Access each cluster using algorithm:(number of clusters:) key
'''

import os
from collections import defaultdict
import numpy as np

from examinlexica.constants import PATH_HISTORICAL_FREQUENCIES
from examinlexica.helpers import get_lexica_order

class ClusteredLexica:
    '''
    An object containing all results of the clustering algorithms in every view.

    Attributes:
        clusters:
            dictionary containing all labels. Access labels using view as key.
            If the number of clusters had to be specified manually use number of
            clusters as second key.
        lexica_list:
            list of lexica feature vectors to access order of the labels.
            (number of lexica in list == number of label in vecto.
    '''
    def __init__(self, path):
        '''
        Initialize object containing all labels of the clustering algorithms.

        Arguments:
            path:
                path to the folder containing results of the clustering algorithms
        '''
        self.clusters = {}
        algorithms = ['meanShift', 'HDBSCAN', 'aggl', 'spectral', 'miniBatchKMeans']
        self.set_clusters(algorithms)
        self.lexica_list = get_lexica_order(path)
        path = path + 'results'
        clustered_data = os.listdir(path)
        self.get_clusters(clustered_data, path)

    def set_clusters(self, algorithms):
        ''' Initialize dictionary of clusters to store results of algorithms '''
        for algorithm in algorithms:
            self.clusters[algorithm] = {}

    def get_clusters(self, data, path):
        ''' Access results of clustering algorithms and store them clusters '''
        path = path + '/'
        for cluster_file in data:
            path_file = path + cluster_file
            algorithm, number_of_clusters = cluster_file.split('_')[0:2]
            if number_of_clusters != 'labels.npy':
                self.clusters[algorithm][int(number_of_clusters)] = np.load(path_file)
            else:
                self.clusters[algorithm] = np.load(path_file)

    def sort_clusters(self, algorithm, number_of_clusters=0):
        ''' Return clusters created using algorithm with number_of_clusters clusters'''
        if number_of_clusters == 0:
            return self.clusters[algorithm]
        return self.clusters[algorithm][number_of_clusters]

    def view_cluster(self, algorithm, number_of_clusters=0):
        ''' Return result of algorithm (i. e. clusters) '''
        clusters = self.sort_clusters(algorithm, number_of_clusters)
        dictio = defaultdict(list)
        for rank in range(len(self.lexica_list)):
            dictio[clusters[rank]].append(self.lexica_list[rank])
        return dictio


    def view_hist_of_cluster(self, algorithm, number_of_clusters=0):
        ''' Return np.histogram of labels '''
        array = self.sort_clusters(algorithm, number_of_clusters)
        if number_of_clusters != 0:
            return np.histogram(array, bins=number_of_clusters)
        return np.histogram(array)

if __name__ == '__main__':
    clSub = ClusteredSubreddits(PATH_HISTORICAL_FREQUENCIES)
    print(clSub.view_cluster('meanShift', 0))
