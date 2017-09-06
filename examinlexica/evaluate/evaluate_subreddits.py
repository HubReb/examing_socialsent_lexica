#! /usr/bin/env python3
# -+- coding: utf-8 -*-

'''
Create an CLusteredData object and evaluate it.
Return an easily readable string for every used clustering algorithm.
If no clusters are specified, clustering algorithms, which need to have
a predifined number of clusters to work, are skipped.
'''

import argparse

from examinlexica.evaluate.evaluate_all import evaluate_clusters_readable_output
from examinlexica.constants import (
    PATH_CLUSTERS,
    PATH,
    ACCEPTABLE_OPTIONS,
    HISTORICAL_OPTIONS
    )
from examinlexica.clusteredData.clustered_data import ClusteredData

def check_number_of_clusters(nc):
    if nc == 0:
        print('Clustering without clusters?!')
        return False
    return True

def evaluate_mini_batch(data, times, view):
    ''' evaluate clusters computed with mini batch kmeans '''
    if check_number_of_clusters(times):
        clusters = evaluate_clusters_readable_output(
            data,
            'miniBatchKmeans',
            times,
            view=view
        )
        return clusters

def evaluate_agg(data, times, view):
    ''' evaluate clusters computed with agglomerative clustering '''
    if check_number_of_clusters(times):
        return evaluate_clusters_readable_output(data, 'aggl', times, view=view)

def evaluate_spectral(data, times, view):
    ''' evaluate clusters computed with spectral clustering '''
    return evaluate_clusters_readable_output(data, 'spectral', times, view=view)

def evaluate_mean_shift(data, view):
    ''' evaluate clusters computed with meanShift algorithm '''
    return evaluate_clusters_readable_output(data, 'meanShift', view=view)

def evaluate_hdbscan(data, view):
    ''' evaluate clusters computed with HDBSCAN algorithm '''
    return evaluate_clusters_readable_output(data, 'HDBSCAN', view=view)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'data',
        help='data to be clustered',
        choices=['subreddits', 'adjectives', 'frequencies'],
    )

    parser.add_argument(
        '-r',
        '--results',
        default='subreddits_results',
        help='folder of the results of clustering'
    )
    parser.add_argument(
        '-c',
        '--clusters',
        default=0,
        help='number of clusters used for aggl. clustering',
        type=int
    )
    parser.add_argument(
        'matrix',
        help='create feature matrix using normal, minimal, maximal or all three values',
        choices=ACCEPTABLE_OPTIONS,
    )
    args = vars(parser.parse_args())
    if args['data'] in HISTORICAL_OPTIONS.keys():
        path = HISTORICAL_OPTIONS[args['data']]
        clusters = ClusteredData(path, args['results'])
    else:
        clusters = ClusteredData(PATH_CLUSTERS, PATH + args['results'])
    print('MBatchKmeans', '-' * 30)
    print(evaluate_mini_batch(clusters, args['clusters'], args['matrix']))
    print('AGGL\n', '__' * 30)
    print(evaluate_agg(clusters, args['clusters'], args['matrix']))
    print('MEANSHIFT\n', '__' *30)
    print(evaluate_mean_shift(clusters, args['matrix']))
    print('HDBSCAN\n', '__' *30)
    print(evaluate_hdbscan(clusters, args['matrix']))
