#! /usr/bin/env python3
# -*- coding: utf-8

'''
Automatic evaluation of clusters using two measures
    measures:
        purity, adjusted mutual information
'''
import os
import argparse

from collections import Counter, defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics

from examinlexica.evaluate.sizes import get_data
# constants:
# change this to determine used feature matrix
MATRIX = ['minimum', 'maximum', 'normal', 'all']
# change this to change look of graphics
SUBPLOTS = [221, 222, 223, 224]
# change this to look at different clusters
PURITY = {
    'AGGL':{'normal':[], 'minimum':[], 'maximum':[], 'all':[]},
    'Kmeans':{'normal':[], 'minimum':[], 'maximum':[], 'all':[]}
}

AD_INF = {
    'AGGL':{'normal':[], 'minimum':[], 'maximum':[], 'all':[]},
    'Kmeans':{'normal':[], 'minimum':[], 'maximum':[], 'all':[]}
}

FOWLKES_MALLOWS = {
    'AGGL':{'normal':[], 'minimum':[], 'maximum':[], 'all':[]},
    'Kmeans':{'normal':[], 'minimum':[], 'maximum':[], 'all':[]}
}

def get_subreddit_labels(filename):
    ''' Return labels of subreddits '''
    with open(filename) as f:
        clusters = f.read().split('\n')[:-1]
    reddit_labels = {}
    labels = defaultdict(list)
    for reddit in clusters:
        reddit, label = reddit.split()
        reddit_labels[reddit] = label
    return reddit_labels

def evaluate_clusters(clusters, labels):
    '''
    Calculate measures and return them
    Each cluster is first asigned a label by majority voting.
    All labels of the subreddits in the cluster are compared to that of the
    cluster. A label is correct if it is identical to the label of the cluster.

    Parameters:
        clusters: list of clusters
        labels: dictionary of form: subreddit : label
    Returns:
        calculated: purity, adjusted mutual information
    '''
    correct = 0
    percentage = []
    labels_predicted = []
    labels_true = []
    correct_clusters = 0
    all_c = 250
    for cluster in clusters:
        label = []
        for reddit in cluster:
            label.append(labels[reddit])
            labels_true.append(labels[reddit])
        cluster_label = Counter(label).most_common()[0][0]
        labels_predicted.extend([cluster_label for i in range(len(label))])
        correct_clusters = len([reddit for reddit in cluster if labels[reddit] == cluster_label])
        correct += correct_clusters
    purity_clusters = correct/all_c
    ad_info = metrics.adjusted_mutual_info_score(labels_true, labels_predicted)
    if ad_info > 1:
        print("ERROR normalized", ad_info)
        print(len(clusters))
    if ad_info < 0:
        print("ERROR ami_score", ad_info)
        print(len(clusters))
    return purity_clusters, ad_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s',
        '--source',
        default='../../../vergleiche/graphs/cosine_euclidean_svd_single',
        help='folder of the results of clustering'
    )
    parser.add_argument(
        '-c',
        '--clusters',
        default=10,
        help='number of clusters used for clustering',
        type=int
    )

    parser.add_argument(
        '-c_end',
        '--clusters_end',
        default=100,
        help='number of clusters used for clustering',
        type=int
    )
    args = vars(parser.parse_args())
    FILENAME = args['source'] + '/comparision_labels_'
    RANGES = [i for i in range(args['clusters'], args['clusters_end']+1)]
    plt.figure(figsize=(22, 16))
    for algorithm, vectors in PURITY.items():
        for cluster_number in RANGES:
            for matrix in vectors.keys():
                data = get_data(
                    FILENAME  + matrix + str(cluster_number),
                    algorithm
                )
                purity, ad_sc = evaluate_clusters(
                    data,
                    get_subreddit_labels('subreddits.txt')
                )
                vectors[matrix].append(purity)
                AD_INF[algorithm][matrix].append(ad_sc)

    for number, plot in enumerate(SUBPLOTS):
        plt.subplot(plot)
        plt.plot(
            RANGES,
            AD_INF['AGGL'][MATRIX[number]],
            '-',
            c='b',
            label='Aggl. (AMI)'
        )
        plt.plot(
            RANGES,
            PURITY['AGGL'][MATRIX[number]],
            ':',
            c='b',
            label='Aggl. (Purity)'
        )
        plt.plot(
            RANGES,
            AD_INF['Kmeans'][MATRIX[number]],
            '-',
            c='r',
            label='Kmeans'+' (AMI)'
        )
        plt.plot(
            RANGES,
            PURITY['Kmeans'][MATRIX[number]],
            ':',
            c='r',
            label='Kmeans (Purity)'
        )
        plt.title(MATRIX[number] + ' values')
        plt.legend()
        plt.grid(True)
    plt.subplots_adjust(top=0.92, bottom=0.10, left=0.10, right=0.95, hspace=0.45,
                    wspace=0.4)

    plt.savefig('graphs/purity_and_adjusted_rand.png', bbbox='tight')
    plt.close()
