#! /usr/bin/env python3
# -*- coding: utf-8 -*-

''' Evaluate results of clustering '''

import sys
from collections import defaultdict

from clustered_subreddits import ClusteredSubreddits
from evaluate_all import evaluate_clusters
from constants import ACCEPTABLE_OPTIONS, PATH

def prettier_print(dictionary):
    ''' Print cluster dictionary in a pretty and readable format '''
    for n_c, clusters in dictionary.items():
        print('-'*48, '\nCluster %s: ' % n_c)
        print('\n'.join(clusters))

def compare_results(clusters_one, clusters_two):
    ''' Compares two results of clustering and returns those with differences in them '''
    clusters_one = get_clusters(clusters_one)
    clusters_two = get_clusters(clusters_two)
    different_cluster = defaultdict(list)
    try:
        for key, cluster in clusters_one.items():
            if cluster not in clusters_two.values():
                different_cluster[key].append(('First alg %i: %s' % (key, ', '.join(cluster))))
            else:
                clusters_two = {key:val for key, val in clusters_two.items() if val != cluster}
        different_cluster = merge_clusters(different_cluster, clusters_two)
    except IndexError:
        print('Uneven number of clusters! Use evalualte_all.py for comparisions.')
        return {}
    return different_cluster

def get_clusters(clusters):
    ''' Returns result of a clustering algorithm'''
    number_of_clusters = int(clusters[3])
    if number_of_clusters <= 2:
        return evaluate_clusters(clusters[0], clusters[1], clusters[2])[0]
    else:
        calculated_clusters = evaluate_clusters(clusters[0], clusters[1], clusters[2], int(clusters[3]))
    return calculated_clusters[int(clusters[3])-2]

def merge_clusters(dictionary, second_dict):
    for key, value in second_dict.items():
        dictionary[key].append('Second alg %i: %s' % (key, ', '.join(value)))
    return dictionary

if __name__ == '__main__':
    if len(sys.argv) < 7 or sys.argv[2] not in ACCEPTABLE_OPTIONS:
        print(
            "usage: python3 compare.py aggl|meanShift|HDBSCAN normal|maximum|all|minimum n_cl",
            "aggl|meanShift|HDBSCAN normal|maximum|all|minimum n_cl"
        )
        sys.exit()
    clustered_data = ClusteredSubreddits(PATH)
    print(
        prettier_print(
            compare_results(
                [clustered_data, sys.argv[2], sys.argv[1], sys.argv[3]],
                [clustered_data, sys.argv[5], sys.argv[4], sys.argv[6]]
            )
        )
    )
