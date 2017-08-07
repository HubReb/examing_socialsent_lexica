#!/usr/bin/env python3
# -*- coding: utf-8 -*-


'''
Create an ClusteredSubreddits object and run basic evaluation on the clusters:
For each clustering algorithm with manually selected number of clusters:
    - compare two following iterations
    - look into histograms of clusters
    - note if there is no longer any change
For all other algorithms:
    - look into histograms of clusters

Further evaluation methods will be implemented later.
'''

import numpy as np
from collections import defaultdict
from sklearn.metrics import pairwise_distances_argmin_min

from clusteredSubreddits import ClusteredSubreddits

def evaluate_time_clusters_changes(data, view, times, algorithm):
    ''' basic evaluation of algorithms with manually selected n. of clusters '''
    old_result = data.view_cluster(algorithm, view, 3)
    print(old_result)
    for time in range(4, times):
        print('\n', '----' * 38, "\nusing %s clusters" % time)
    #    print('Comparing next pair: ')
        result = data.view_cluster(algorithm, view, time)
        print(data.view_hist_of_cluster(algorithm, view, time))
        if result ==  old_result:
            print('No change despite increasing number of clusters')
            continue
        print('Change: ')
        print('Clusters: ')
        for i, key in enumerate(result.keys()):
            print('Cluster %i: %s' % ((i + 1), ', '.join(result[key])))

def evaluate_time_clusters(data, view, times, algorithm):
    ''' basic evaluation of algorithms with manually selected n. of clusters '''
    time_result = {}
    for time in range(3, times):
        result = data.view_cluster(algorithm, view, time)
        number_of_cluster = time
        res = defaultdict(list)
        for number, resu in enumerate(result):
            res[resu].append(number)
            time_result[number_of_cluster] = res
    print(time_result)


def evaluate_clusters(data, view, algorithm):
    '''evaluation algorithm for algorithms selecting n. of clusters automatically'''
    result = data.view_cluster(algorithm, view)
    res = defaultdict(list)
    for number, resu in enumerate(result):
        res[resu].append(number)
    print(res)
    print(data.view_hist_of_cluster(algorithm, view))
    print(result)


def evaluateMiniBatch(data, view, times):
    evaluate_time_clusters(data, view, times, 'miniBatchKmeans')

def evaluateAgg(data, view, times):
    evaluate_time_clusters_changes(data, view, times, 'aggl')

def evaluateSpectral(data, view, times):
    evaluate_time_clusters(data, view, times, 'spectral')

def evaluateMeanShift(data, view):
    evaluate_clusters(data, view, 'meanShift')

def evaluateHDBSCAN(data, view):
    evaluate_clusters(data, view, 'HDBSCAN')

if __name__ == '__main__':
    from sys import argv, exit
    if len(argv) < 2:
        print("usage: python3 evaluate_methods.py normal|maximum|all|minimum")
        exit()
    path = '/home/students/hubert/socialsent/socialsent/socialsent/data/lexicons/subreddits'
    clusters = ClusteredSubreddits(path)
    #evaluateMiniBatch(clusters, argv[1], 200)
    print('AGGL\n', '__' * 30)
    evaluateAgg(clusters, argv[1], 200)
    print('SPECTRAL\n', '__' *30)
    evaluateSpectral(clusters, argv[1], 200)
    print('MEANSHIFT\n', '__' *30)
    evaluateMeanShift(clusters, argv[1])
    print('HDBSCAN\n', '__' *30)
    evaluateHDBSCAN(clusters, argv[1])
