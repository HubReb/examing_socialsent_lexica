#! /usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Extract all labels created by clustering and store them in a single dictionary in
form:
    algorithm : labels (feature vector)
The lexica names are stored in a list, whose order corresponds to the order of
the features in the feature vectors (labels).
Access each cluster using algorithm:(number of clusters:) key
'''

import os
from collections import defaultdict
import numpy as np

from examinlexica.constants import PATH_HISTORICAL_FREQUENCIES
from examinlexica.helpers import get_lexica_order

class ClusteredLexica:
    '''
    An object containing all results of the clustering algorithms in every view /
    feature matrix.

    Attributes:
        clusters:
            dictionary of form: algorithm:labels
            For the algorithms which need the the number of clusters to be
            specified beforehand, the dictionary has the form:
            algorithm:[number_of_clusters:labels]
        lexica_list:
            list of lexica feature vectors to access order of the lexica in the
            labels.
            (number of lexicon in list == number of label in vector.
    '''
    def __init__(self, path, path_clusters):
        '''
        Initialize object containing all labels of the clustering algorithms.

        Arguments:
            path:
                path to the folder containing the original data of the lexica
            path_clusters:
                path to the folder containin the clustered lexica
        '''
        self.clusters = {}
        algorithms = ['meanShift', 'HDBSCAN', 'aggl', 'spectral', 'miniBatchKmeans']
        self.set_clusters(algorithms)
        self.lexica_list = get_lexica_order(path)
        clustered_data = os.listdir(path_clusters)
        self.get_clusters(clustered_data, path_clusters)

    def set_clusters(self, algorithms):
        ''' Initialize dictionary of clusters to store results of algorithms in '''
        for algorithm in algorithms:
            self.clusters[algorithm] = {}

    def get_clusters(self, data, path):
        ''' Access results of clustering algorithms and store them in clusters
            Arguments:
                data: names of the files containing the clustered lexica
                path: path to the folder containing the clustered lexica
        '''
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
    histSub = ClusteredLexica(PATH_HISTORICAL_FREQUENCIES, PATH_HISTORICAL_FREQUENCIES + 'results')
