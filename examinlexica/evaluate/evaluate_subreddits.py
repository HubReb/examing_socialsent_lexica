#! /usr/bin/env python3
# -+- coding: utf-8 -*-

'''
Create an CLusteredSubreddits object and evaluate it.
Return an easily readable string for every used clustering algorithm.
If no clusters are specified, clustering algorithms, which need to have
a predifined number of clusters to work, are skipped.
'''

import argparse

from examinlexica.evaluate.evaluate_all import evaluate_clusters_readable_output
from examinlexica.constants import (
    PATH,
    ACCEPTABLE_OPTIONS
    )
from examinlexica.clusteredData.clustered_subreddits import ClusteredSubreddits


def evaluate_mini_batch(data, times, view):
    ''' evaluate clusters computed with mini batch kmeans '''
    clusters = evaluate_clusters_readable_output(
        data,
        'miniBatchKmeans',
        times,
        view=view
    )
    return clusters

def evaluate_agg(data, times, view):
    ''' evaluate clusters computed with agglomerative clustering '''
    print(data, times, view)
    if times == 0:
        print('Clustering without clusters?!')
        return
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
        'matrix',
        help='create feature matrix using normal, minimal, maximal or all three values',
        choices=ACCEPTABLE_OPTIONS,
    )
    args = vars(parser.parse_args())
    clusters = ClusteredSubreddits(PATH, args['results'])
    print('AGGL\n', '__' * 30)
    print(evaluate_agg(clusters, args['clusters'], args['matrix']))
    print('MEANSHIFT\n', '__' *30)
    print(evaluate_mean_shift(clusters, args['matrix']))
    print('HDBSCAN\n', '__' *30)
    print(evaluate_hdbscan(clusters, args['matrix']))
