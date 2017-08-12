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
from collections import defaultdict

from clustered_subreddits import ClusteredSubreddits


ACCEPTABLE_OPTIONS = ["normal", "minimum", "maximum", "all"]
PATH = '/home/students/hubert/socialsent/socialsent/socialsent/data/lexicons/subreddits'

def evaluate_time_clusters_changes(data, view, times, algorithm):
    ''' basic evaluation of algorithms with manually selected n. of clusters '''
    old_result = data.view_cluster(algorithm, view, 3)
    for i, key in enumerate(old_result.keys()):
        print('Cluster %i: %s' % ((i + 1), ', '.join(old_result[key])))
    for time in range(4, times):
        print('\n', '----' * 38, "\nusing %s clusters" % time)
        result = data.view_cluster(algorithm, view, time)
        print(data.view_hist_of_cluster(algorithm, view, time))
        if result == old_result:
            print('No change despite increasing number of clusters')
            continue
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
    for i, key in enumerate(result.keys()):
        print('Cluster: %i: %s' % ((i+1), ', '.join(result[key])))


def evaluate_mini_batch(data, view, times):
    ''' evaluate clusters computed with mini batch kmeans '''
    evaluate_time_clusters(data, view, times, 'miniBatchKmeans')

def evaluate_agg(data, view, times):
    ''' evaluate clusters computed with agglomerative clustering '''
    evaluate_time_clusters_changes(data, view, times, 'aggl')

def evaluate_spectral(data, view, times):
    ''' evaluate clusters computed with spectral clustering '''
    evaluate_time_clusters(data, view, times, 'spectral')

def evaluate_mean_shift(data, view):
    ''' evaluate clusters computed with meanShift algorithm '''
    evaluate_clusters(data, view, 'meanShift')

def evaluate_hdbscan(data, view):
    ''' evaluate clusters computed with HDBSCAN algorithm '''
    evaluate_clusters(data, view, 'HDBSCAN')

if __name__ == '__main__':
    import sys
    import sys
    if len(sys.argv) < 2 or sys.argv[1] not in ACCEPTABLE_OPTIONS:
        print("usage: python3 evaluate_methods.py normal|maximum|all|minimum")
        sys.exit()
    clusters = ClusteredSubreddits(PATH)
    print('AGGL\n', '__' * 30)
    evaluate_agg(clusters, sys.argv[1], 200)
    print('MEANSHIFT\n', '__' *30)
    evaluate_mean_shift(clusters, sys.argv[1])
    print('HDBSCAN\n', '__' *30)
    evaluate_hdbscan(clusters, sys.argv[1])
