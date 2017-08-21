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
from constants import ACCEPTABLE_OPTIONS, PATH

def evaluate_time_clusters(data, view, times, algorithm):
    ''' basic evaluation of algorithms with manually selected n. of clusters '''
    time_result = {}
    for time in range(2, times+1):
        result = data.view_cluster(algorithm, view, time)
        number_of_cluster = time
        res = defaultdict(list)
        for number, resu in enumerate(result):
            res[resu].append(number)
            time_result[number_of_cluster] = res
    return time_result

def evaluate_clusters(data, view, algorithm, times=0):
    ''' basic evaluation of algorithms with manually selected n. of clusters '''
    result_total = []
    if times == 0:
        result_total.append(data.view_cluster(algorithm, view))
    else:
        for time in range(2, times+1):
#            result_total.append('----' * 38 + '\nusing %s clusters' % time)
           # result = data.view_cluster(algorithm, view, time)
            result_total.append(data.view_cluster(algorithm, view, time))
#            print(result_total)
    return result_total

def evaluate_clusters_readable_output(data, view, algorithm, times=0):
    ''' Returns easily readable output of cluster algorithms '''
    results = evaluate_clusters(data, view, algorithm, times)
    result_total = []
    for time, result in enumerate(results):
        if times != 0:
            result_total.append('----' * 38 + '\nusing %s clusters' % (time+3))
        result_total.append(pretty_print(result))
    return '\n'.join(result_total)

def pretty_print(result):
    ''' Return pretty print of cluster entry '''
    result_total = ''
    for key, value in result.items():
        result_total += '\nCluster %i: %s' % (key, ', '.join(value))
    return result_total

def evaluate_mini_batch(data, view, times):
    ''' evaluate clusters computed with mini batch kmeans '''
    return evaluate_clusters_readable_output(data, view, 'miniBatchKmeans', times)

def evaluate_agg(data, view, times):
    ''' evaluate clusters computed with agglomerative clustering '''
    return evaluate_clusters_readable_output(data, view, 'aggl', times)

def evaluate_spectral(data, view, times):
    ''' evaluate clusters computed with spectral clustering '''
    return evaluate_clusters_readable_output(data, view, 'spectral', times,)

def evaluate_mean_shift(data, view):
    ''' evaluate clusters computed with meanShift algorithm '''
    return evaluate_clusters_readable_output(data, view, 'meanShift')

def evaluate_hdbscan(data, view):
    ''' evaluate clusters computed with HDBSCAN algorithm '''
    return evaluate_clusters_readable_output(data, view, 'HDBSCAN')

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2 or sys.argv[1] not in ACCEPTABLE_OPTIONS:
        print("usage: python3 evaluate_methods.py normal|maximum|all|minimum")
        sys.exit()
    clusters = ClusteredSubreddits(PATH)
    print('KMeans\n', '__' * 30)
    #print(evaluate_mini_batch(clusters, sys.argv[1], 200))
    print('AGGL\n', '__' * 30)
    #print(evaluate_agg(clusters, sys.argv[1], 200))
    print('MEANSHIFT\n', '__' *30)
    print(evaluate_mean_shift(clusters, sys.argv[1]))
    print('HDBSCAN\n', '__' *30)
    print(evaluate_hdbscan(clusters, sys.argv[1]))
