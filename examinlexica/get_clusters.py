#! /usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Complete cluster and evaluation progress in one file. Use this to start entire
progress with one call.
'''

import os
import sys
import argparse
import shutil

from examinlexica.cluster import start_cluster
from examinlexica.original.subreddit_data import SubredditData
from examinlexica.original.historical_data import HistoricalData
from examinlexica.clusteredData.clustered_data import ClusteredData
from examinlexica.constants import (
    PATH_CLUSTERS,
    HISTORICAL_OPTIONS,
    ACCEPTABLE_OPTIONS,
    )
from examinlexica.evaluate.evaluate import (
    evaluate_agg,
    evaluate_hdbscan,
    evaluate_kmeans
)

def cluster_process(data, result_folder, matrix, number_of_clusters, algorithm):
    '''
    Start clustering process.

    Arguments:
        data: data to be clutered
        result_folder: path to a folder, in which the results are stored
        matrix: whether to use unchanged, minimal, maximal or all three values
        number_of_clusters: number of clusters to use for Kmeans and aggl. Clust.
        algorithm: algorithm to use for clustering
    Returns:
        Easily readable string of clusters. The result is also written in a file
        in the result_folder named 'matrix_number_of_clusters.txt.
    '''
    hist = False
    if data in HISTORICAL_OPTIONS.keys():
        path = HISTORICAL_OPTIONS[data]
        data = HistoricalData(path)
        hist = True
    else:
        data = SubredditData(PATH_CLUSTERS)
        path = PATH_CLUSTERS
    start_cluster(data.sentiments, 'temp', matrix, number_of_clusters, algorithm)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    if hist:
        clusters = ClusteredData(path, 'temp_results')
    else:
        clusters = ClusteredData(PATH_CLUSTERS, 'temp_results')
    shutil.rmtree('temp_results')
    results = ""
    if algorithm == 'all':
        if number_of_clusters:
            results += evaluate_kmeans(clusters, number_of_clusters, matrix)
            results += evaluate_agg(clusters, number_of_clusters, matrix)
        results += evaluate_hdbscan(clusters, matrix)
    else:
        algorithms = {
            'Aggl' : evaluate_agg(clusters, args['clusters'], args['matrix']),
            'Kmeans' : evaluate_kmeans(clusters, args['clusters'], args['matrix']),
            'HDBSCAN' : evaluate_hdbscan(clusters, args['matrix'])
        }
        if number_of_clusters == 0 and algorithm != 'HDBSCAN':
            print('Cannot cluster without clusters!')
            sys.exit()
        evaluate_function = algorithms[algorithm]
        results += evaluate_function
    filename = result_folder + '/' + algorithm + '_' + matrix + '_' + str(number_of_clusters) + '.txt'
    with open(filename, 'w') as f:
        f.write(results)
    return results

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
        default=None
    )
    parser.add_argument(
        '-a',
        '--algorithm',
        help='algorithm to use for clustering',
        default='all',
        choices=['Aggl', 'Kmeans', 'HDBSCAN'],
    )
    args = vars(parser.parse_args())
    print(
        cluster_process(
            args['data'],
            args['results'],
            args['matrix'],
            args['clusters'],
            args['algorithm']
        )
    )
