#! /usr/bin/env python3
# -*- coding: utf-8

'''
Automatic evaluation of clusters using two measures:
    TP = subreddits whose labels are identical with the label of their cluster
    FP = subreddits whose labels differs from the label of their cluster
    TN = subreddits whose labels differ from a cluster and are not part of it
    FN = subreddits whose labels are identical with that of a cluster, but are not
        part od that cluster
    measures:
        purity, Fowlkes Mallows, adjusted rand index
'''

from collections import Counter, defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics

# constants:
# change this to select outher source files
FILENAME = 'graphs/cosine_euclidean_svd_single/comparision_labels_'
# change this to determine used feature matrix
MATRIX = ['minimum', 'maximum', 'normal', 'all']
# change this to change look of graphics
SUBPLOTS = [221, 222, 223, 224]
# change this to look at different clusters
RANGES = [i for i in range(10, 106)]
PURITY = {
    'AGGL':{'normal':[], 'minimum':[], 'maximum':[], 'all':[]},
    'Kmeans':{'normal':[], 'minimum':[], 'maximum':[], 'all':[]}
}

RAND_INDEX = {
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
        calculated: precision, recall, f-measure
    '''

    correct = 0
    percentage = []
    false_p = 0
    labels_predicted = []
    labels_true = []
    correct_clusters = 0
    for cluster in clusters:
        if cluster == []:
            percentage.append(0)
        label = []
        for reddit in cluster:
            label.append(labels[reddit])
            labels_true.append(labels[reddit])
        cluster_label = Counter(label).most_common()[0][0]
        labels_predicted.extend([cluster_label for i in range(len(label))])
        correct_clusters = len([reddit for reddit in cluster if labels[reddit] == cluster_label])
        correct += correct_clusters
    if correct_clusters == 0:
        print(clusters)
    purity_clusters = correct_clusters/250
    folks = metrics.fowlkes_mallows_score(labels_true, labels_predicted)
    ad_rand_score = metrics.adjusted_rand_score(labels_true, labels_predicted)
    if folks > 1:
        print("ERROR folks", folks)
        print(len(clusters))
    if folks < 0:
        print("ERROR hom", folks)
        print(len(clusters))
    if ad_rand_score > 1:
        print("ERROR ad_rand_score", ad_rand_score)
        print(len(clusters))
    if ad_rand_score < 0:
        print("ERROR ad_rand_score", ad_rand_score)
        print(len(clusters))
    return purity_clusters, folks, ad_rand_score

def get_data(filename, cluster_algorithm):
    ''' Returns list of lists of subreddits created by the alg. Each list is a cluster. '''
    clusters = []
    algorithms = ['AGGL', 'HDBSCAN', 'MEANSHIFT', 'Kmeans']
    with open(filename) as f:
        all_clusters = f.read().split('\n')
    start_point = False
    for cluster in all_clusters:
        if '__' in cluster:
            continue
        if '--' in cluster:
            cluster = cluster.split()[0]
        if cluster in algorithms:
            if cluster not in cluster_algorithm:
                start_point = False
                continue
            else:
                start_point = True
                continue
        if start_point:
            try:
                label_number, datapoints = cluster.split(':')
                if label_number.endswith('-1'):
                    datapoints = datapoints.split(', ')
                    for point in datapoints:
                        clusters.append([point])
                else:
                    clusters.append([point.strip(',') for point in datapoints.split()])
            except ValueError:
                continue
    return clusters




for algorithm, vectors in PURITY.items():
    for cluster_number in RANGES:
        for matrix in vectors.keys():
            data = get_data(
                FILENAME  + matrix + str(cluster_number),
                algorithm
            )
            purity, complete, homogen = evaluate_clusters(
                data,
                get_subreddit_labels('subreddits.txt')
            )
            vectors[matrix].append(purity)
            RAND_INDEX[algorithm][matrix].append(complete)
            FOWLKES_MALLOWS[algorithm][matrix].append(homogen)

plt.figure(figsize=(22, 20), dpi=120)
for number, plot in enumerate(SUBPLOTS):
    plt.subplot(plot)
    plt.plot(
        RANGES,
        RAND_INDEX['AGGL'][MATRIX[number]],
        '-',
        c='b',
        label='Aggl. (adjusted rand index)'
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
        RAND_INDEX['Kmeans'][MATRIX[number]],
        '-',
        c='r',
        label='Kmeans'+' (adjusted rand index)'
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
plt.subplot(222)
plt.savefig('graphs/purity_and_adjusted_rand_sc_svd_cosine.png')
plt.close()
plt.figure(figsize=(22, 20), dpi=120)
for number, plot in enumerate(SUBPLOTS):
    plt.subplot(plot)
    for algorithm, vectors in FOWLKES_MALLOWS.items():
        plt.plot(
            RANGES,
            vectors[MATRIX[number]],
            label=algorithm + ' (Fowlkes Mallows)'
        )
    plt.grid(True)
    plt.xlabel('Number of Clusters')
    plt.ylabel('correct classified subreddits')
    plt.title(MATRIX[number] + ' values')
    plt.legend()
plt.savefig('graphs/fowlkes_cosine_svd.png')
