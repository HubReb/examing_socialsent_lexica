#! /usr/bin/env python3
# -*- coding: utf-8 -*-


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
    PATH
    )
from examinlexica.evaluate.evaluate_subreddits import (
    evaluate_agg,
    evaluate_hdbscan,
    evaluate_kmeans
)

def cluster_process(data, result_folder, matrix, number_of_clusters):
    hist = False
    if data in HISTORICAL_OPTIONS.keys():
        path = HISTORICAL_OPTIONS[data]
        data = HistoricalData(path)
        hist = True
    else:
        data = SubredditData(PATH_CLUSTERS)
        path = PATH_CLUSTERS
    start_cluster(data.sentiments, 'temp', matrix, number_of_clusters)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    if hist: 
        clusters = ClusteredData(path, 'temp_results')
    else:
        clusters = ClusteredData(PATH_CLUSTERS, 'temp_results')
    shutil.rmtree('temp_results')
    results = ""
    results += evaluate_kmeans(clusters, number_of_clusters, matrix)
    results += evaluate_agg(clusters, number_of_clusters, matrix)
    results += evaluate_hdbscan(clusters, matrix)
    return results

if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'data',
        help='data to be clustered',
        choices=['subreddits', 'adjectives', 'frequencies'],
    )
    parser.add_argument(
        '-r',
        '--results',
        default='./',
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
    args = vars(parser.parse_args())
    print(cluster_process(args['data'], args['results'], args['matrix'], args['clusters']))
