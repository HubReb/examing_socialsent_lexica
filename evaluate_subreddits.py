#! /usr/bin/env python3
# -+- coding: utf-8 -*-

import sys

from examinlexica.evaluate_all import evaluate_clusters_readable_output
from examinlexica.constants import (
                        PATH,
                        ACCEPTABLE_OPTIONS
                        )
from examinlexica.clustered_subreddits import ClusteredSubreddits


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
    if len(sys.argv) < 2 or sys.argv[1] not in ACCEPTABLE_OPTIONS:
        print("usage: python3 evaluate_methods.py normal|maximum|all|minimum")
        sys.exit()
    clusters = ClusteredSubreddits(PATH)
    print('KMeans\n', '__' * 30)
    #print(evaluate_mini_batch(clusters, sys.argv[1], 200))
    print('AGGL\n', '__' * 30)
    print(evaluate_agg(clusters, 200, sys.argv[1]))
    print('MEANSHIFT\n', '__' *30)
    print(evaluate_mean_shift(clusters, sys.argv[1]))
    print('HDBSCAN\n', '__' *30)
    print(evaluate_hdbscan(clusters, sys.argv[1]))
