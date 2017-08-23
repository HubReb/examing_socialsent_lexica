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

from examinlexica.clustered_historical import ClusteredLexica
from examinlexica.constants import ACCEPTABLE_OPTIONS, HISTORICAL_OPTIONS
import examinlexica.evaluate_all

def evaluate_clusters(data, algorithm, times=0, view=None):
    ''' basic evaluation of algorithms with manually selected n. of clusters '''
    result_total = []
    if times == 0:
        result_total.append(data.view_cluster(algorithm))
    else:
        for time in range(2, times+1):
            result_total.append(data.view_cluster(algorithm, time))
    return result_total

evaluate_all.evaluate_clusters = evaluate_clusters

def evaluate_mini_batch(data, times):
    ''' evaluate clusters computed with mini batch kmeans '''
    return evaluate_all.evaluate_clusters_readable_output(data, 'miniBatchKmeans', times)

def evaluate_agg(data, times):
    ''' evaluate clusters computed with agglomerative clustering '''
    return evaluate_all.evaluate_clusters_readable_output(data, 'aggl', times)

def evaluate_spectral(data, times):
    ''' evaluate clusters computed with spectral clustering '''
    return evaluate_all.evaluate_clusters_readable_output(data, 'spectral', times,)

def evaluate_mean_shift(data):
    ''' evaluate clusters computed with meanShift algorithm '''
    return evaluate_all.evaluate_clusters_readable_output(data, 'meanShift')

def evaluate_hdbscan(data):
    ''' evaluate clusters computed with HDBSCAN algorithm '''
    return evaluate_all.evaluate_clusters_readable_output(data, 'HDBSCAN')

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2 or sys.argv[1] not in HISTORICAL_OPTIONS.keys():
        print('usage: python3 cluster.py adjectives|frequencies')
        sys.exit()
    path = HISTORICAL_OPTIONS[sys.argv[1]]
    clusters = ClusteredLexica(path)
    print('KMeans\n', '__' * 30)
    #print(evaluate_mini_batch(clusters, sys.argv[1], 200))
    print('AGGL\n', '__' * 30)
    print(evaluate_agg(clusters, 6))
    print('MEANSHIFT\n', '__' *30)
    print(evaluate_mean_shift(clusters))
    print('HDBSCAN\n', '__' *30)
    print(evaluate_hdbscan(clusters))
