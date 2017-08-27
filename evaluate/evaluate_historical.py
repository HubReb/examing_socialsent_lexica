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

import argparse
from examinlexica.clusteredData.clustered_historical import ClusteredLexica
from examinlexica.constants import HISTORICAL_OPTIONS
import examinlexica.evaluate.evaluate_all as evaluate_all

def evaluate_clusters(data, algorithm, time=0, view=None):
    ''' basic evaluation of algorithms with manually selected n. of clusters '''
    if time == 0:
        return data.view_cluster(algorithm)
    else:
        return data.view_cluster(algorithm, time)
    return result_total

evaluate_all.evaluate_clusters = evaluate_clusters

def evaluate_mini_batch(data, times):
    ''' evaluate clusters computed with mini batch kmeans '''
    return evaluate_all.evaluate_clusters_readable_output(data, 'miniBatchKmeans', times)

def evaluate_agg(data, times):
    ''' evaluate clusters computed with agglomerative clustering '''
    if times == 0:
        print('Clustering without clusters?!')
        return
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-r',
        '--results',
        default='.',
        help='folder for the results of clustering'
    )
    parser.add_argument(
        '-c',
        '--clusters',
        default=0,
        help='number of clusters used for aggl. clustering',
        type=int
    )
    parser.add_argument(
        'data',
        help='clustered data',
        choices=HISTORICAL_OPTIONS.keys()
    )
    args = vars(parser.parse_args())
    path = HISTORICAL_OPTIONS[args['data']]
    clusters = ClusteredLexica(path, args['results'])
    print('KMeans\n', '__' * 30)
    clusters_number = args['clusters']
    #print(evaluate_mini_batch(clusters, clusters_number))
    print('AGGL\n', '__' * 30)
    print(evaluate_agg(clusters, clusters_number))
    print('MEANSHIFT\n', '__' *30)
    print(evaluate_mean_shift(clusters))
    print('HDBSCAN\n', '__' *30)
    print(evaluate_hdbscan(clusters))
