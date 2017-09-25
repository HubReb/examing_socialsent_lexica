#! /usr/bin/env python3
# -+- coding: utf-8 -*-

'''
Create an CLusteredData object and evaluate it.
Return an easily readable string for every used clustering algorithm.
If no clusters are specified, clustering algorithms needing to have
a predifined number of clusters to work are skipped.
'''

import argparse

from examinlexica.constants import (
    PATH_CLUSTERS,
    PATH,
    ACCEPTABLE_OPTIONS,
    HISTORICAL_OPTIONS
    )
from examinlexica.clusteredData.clustered_data import ClusteredData

def evaluate_clusters(data, algorithm, time=0, view=None):
    ''' basic evaluation of algorithms with manually selected n. of clusters '''
    if time == 0:
        return data.view_cluster(algorithm, view)
    return data.view_cluster(algorithm, view, time)

def evaluate_clusters_readable_output(data, algorithm, time=0, view=None):
    ''' Returns easily readable output of cluster algorithms '''
    result = evaluate_clusters(data, algorithm, time, view=view)
    result_total = []
    result_total.append(pretty_print(result))
    return '\n'.join(result_total)

def pretty_print(result):
    ''' Return pretty print of cluster entry '''
    result_total = ''
    for key, value in result.items():
        result_total += '\nCluster %i: %s' % (key, ', '.join(value))
    return result_total


def check_number_of_clusters(nc):
    ''' prevent evaluation attempt of Kmeans and aggl. without specifying clusters '''
    if nc == 0:
        print('Clustering without clusters?!')
        return False
    return True

def evaluate_kmeans(data, times, view):
    ''' evaluate clusters computed with mini batch kmeans '''
    if check_number_of_clusters(times):
        clusters = evaluate_clusters_readable_output(
            data,
            'Kmeans',
            times,
            view=view
        )
        return clusters

def evaluate_agg(data, times, view):
    ''' evaluate clusters computed with agglomerative clustering '''
    if check_number_of_clusters(times):
        return evaluate_clusters_readable_output(data, 'aggl', times, view=view)

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
        default='_results',
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
        help='create feature matrix using specified feature vectors',
        choices=ACCEPTABLE_OPTIONS,
    )
    parser.add_argument(
        '-a',
        '--algorithm',
        help='algorithm to use for clustering',
        default='all',
        choices=['Aggl', 'Kmeans', 'HDBSCAN'],
    )

    args = vars(parser.parse_args())
    if args['data'] in HISTORICAL_OPTIONS.keys():
        path = HISTORICAL_OPTIONS[args['data']]
        clusters = ClusteredData(path, args['results'])
    else:
        clusters = ClusteredData(PATH_CLUSTERS, PATH + args['results'])
    algorithms = {
        'Aggl' : evaluate_agg(clusters, args['clusters'], args['matrix']),
        'Kmeans' : evaluate_kmeans(clusters, args['clusters'], args['matrix']),
        'HDBSCAN' : evaluate_hdbscan(clusters, args['matrix'])
    }
    if args['algorithm'] == 'all':
        print('Kmeans\n', '_' * 30)
        print(evaluate_kmeans(clusters, args['clusters'], args['matrix']))
        print('AGGL\n', '__' * 30)
        print(evaluate_agg(clusters, args['clusters'], args['matrix']))
        print('HDBSCAN\n', '__' *30)
        print(evaluate_hdbscan(clusters, args['matrix']))
    else:
        cluster_algorithm = algorithms[args['algorithm']]
        print(cluster_algorithm)
