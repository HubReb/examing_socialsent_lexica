#! /usr/bin/env python3
# -*- coding: utf-8 -*-

''' Evaluate results of clustering '''

import sys
import argparse
from collections import defaultdict

from examinlexica.clusteredData.clustered_data import ClusteredData
from examinlexica.evaluate.evaluate_all import evaluate_clusters
from examinlexica.constants import (
    ACCEPTABLE_OPTIONS,
    PATH_CLUSTERS,
    PATH,
    ALGORITHMS
    )

def prettier_print(results):
    ''' Print cluster dictionary in a pretty and readable format '''
    for n_c, clusters in results.items():
        print('-'*48, '\nCluster %s: ' % n_c)
        print('\n'.join(clusters))

def compare_results(clusters_one, clusters_two):
    ''' Compares two results of clustering algorithms and returns result.

    Compares the results of two clustering algorithms and looks for different clusters.
    All clusters differenting in their content (e. g. cluster 1.1 has 'a.tsv', but
    cluster 1.2 has 'a.tst' and 'b.tsv') are saved in a dictionary.
    This dictionary is returned.

    Parameters:
        clusters_one: list of attributes defining one specific clustering algorithm
        result. This list contains:
            data: object which stores all clustered data
            view: 'view' on the sentiments as specified in SubredditData class
            algorithm: algorithm used for clustering
            number of clusters: predifined number of clusters in result; if algorithm
                chooses number of clusters automatically use 0
        clusters_two: list of attributes defining another result of a clustering
        algorithm. The list contains the same attributes as clusters_one

    Returns:
        a dictionary containing all different clusters
    '''
    clusters_one = get_clusters(clusters_one)
    clusters_two = get_clusters(clusters_two)
    different_cluster = defaultdict(list)
    try:
        for key, cluster in clusters_one.items():
            if cluster not in clusters_two.values():
                new_cluster = 'First alg %i: %s' % (key, ', '.join(cluster))
                different_cluster[key].append(new_cluster)
            else:
                clusters_two = {k:v for k, v in clusters_two.items() if v != cluster}
        different_cluster = merge_clusters(different_cluster, clusters_two)
    except IndexError:
        print('Uneven number of clusters! Use evalualte_all.py for comparisions.')
        return {}
    return different_cluster

def get_clusters(clusters):
    ''' Returns result of a clustering algorithm'''
    number_of_clusters = int(clusters[2])
    if number_of_clusters:
         calculated_clusters = evaluate_clusters(
            clusters[0],
            clusters[1],
            int(clusters[2]),
            clusters[3]
        )
    else:
        return evaluate_clusters(clusters[0], clusters[1], clusters[3])
    return calculated_clusters

def merge_clusters(dictionary, second_dict):
    ''' Merge two dictionaries into one '''
    for key, value in second_dict.items():
        dictionary[key].append('Second alg %i: %s' % (key, ', '.join(value)))
    return dictionary

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-r',
        '--results',
        default=PATH+'subreddits_results',
        help='folder of the results of clustering'
    )
    parser.add_argument(
        '-c',
        '--clusters',
        default=0,
        help='number of clusters used for aggl./KMeans clustering (alg. 1)',
        type=int
    )
    parser.add_argument(
        '-c_s',
        '--clusters_second',
        default=0,
        help='number of clusters used for aggl./KMeans clustering (alg. 2)',
        type=int
    )
    parser.add_argument(
        'matrix_algorithm_one',
        help='feature matrix composed of normal, minimal, maximal, all values',
        choices=ACCEPTABLE_OPTIONS,
    )
    parser.add_argument(
        'matrix_algorithm_two',
        help='feature matrix composed of normal, minimal, maximal, all values',
        choices=ACCEPTABLE_OPTIONS,
    )
    parser.add_argument(
        'algorithm_one',
        help='clustering algorithm used to create the first clusters',
        choices=ALGORITHMS,
    )
    parser.add_argument(
        'algorithm_two',
        help='clustering algorithm used to create the second clusters',
        choices=ALGORITHMS,
    )
    args = vars(parser.parse_args())
    clustered_data = ClusteredData(PATH_CLUSTERS, args['results'])
    print(
        prettier_print(
            compare_results(
                [
                    clustered_data,
                    args['algorithm_one'],
                    args['clusters'],
                    args['matrix_algorithm_one']
                ],
                [
                    clustered_data,
                    args['algorithm_two'],
                    args['clusters_second'],
                    args['matrix_algorithm_two']
                ]
            )
        )
    )
