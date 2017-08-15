#! /usr/bin/env python3
# -*- coding: utf-8 -*-

''' Evaluate results of clustering '''

import sys

from clustered_subreddits import ClusteredSubreddits
from evaluate_all import evaluate_clusters
from constants import ACCEPTABLE_OPTIONS, PATH

def prettier_print(dictionary):
    ''' Print cluster dictionary in a pretty and readable format '''
    for key, value in dictionary.items():
        print('-'*48, '\nCluster %i: ' % key)
        for cluster in value:
            print(', '.join(cluster))

def compare_results(clusters_one, clusters_two):
    ''' Compares two results of clustering and returns those with differences in them '''
    clusters_one = get_clusters(clusters_one)
    clusters_two = get_clusters(clusters_two)
    different_cluster = {}
    try:
        for number_of_clusters, clusters in clusters_one.items():
            if set(clusters) != set(clusters_two[number_of_clusters]):
                different_cluster[number_of_clusters] = (clusters, clusters_two[number_of_clusters])
    except IndexError:
        print('Unenven number of clusters! Use evalualte_all.py for comparisions.')
        return {}
    return different_cluster

def get_clusters(clusters):
    ''' Returns result of a clustering algorithm'''
    number_of_clusters = int(clusters[3])
    if number_of_clusters <= 2:
        return evaluate_clusters(clusters[0], clusters[1], clusters[2])
    else:
        calculated_clusters = evaluate_clusters(clusters[0], clusters[1], clusters[2], int(clusters[3]))
    return calculated_clusters[int(clusters[3])-2]

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
