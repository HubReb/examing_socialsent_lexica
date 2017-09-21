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
        precision, recall
'''

from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# constants:
# change this to select outher source files
FILENAME = 'graphs/cosine_euclidean_svd_single/comparision_labels_'
# change this to determine used feature matrix
MATRIX = ['minimum', 'maximum', 'normal', 'all']
# change this to change look of graphics
SUBPLOTS = [221, 222, 223, 224]
# change this to look at different clusters
RANGES = [i for i in range(2, 106)]

def get_subreddit_labels(filename):
    ''' Return labels of subreddits '''
    with open(filename) as f:
        clusters = f.read().split('\n')[:-1]
    reddit_labels = {}
    for reddit in clusters:
        reddit, label = reddit.split()
        reddit_labels[reddit] = label
    return reddit_labels

def evaluate_clusters(clusters, labels, all_subs=False):
    '''
    Calculate measures and return them
    Each cluster is first asigned a label by majority voting.
    All labels of the subreddits in the cluster are compared to that of the
    cluster. A label is correct if it is identical to the label of the cluster.

    Parameters:
        clusters: list of clusters
        labels: dictionary of form: subreddit : label
        all_subs: Boolean, determines if clusters containing only one subreddit are
            considered
    Returns:
        calculated: precision, recall, f-measure
    '''

    correct = 0
    all_s = 259
    percentage = []
    for cluster in clusters:
        if all_subs:
            if len(cluster) == 1:
                all_s -= 1
                continue
        if cluster == []:
            percentage.append(0)
        label = []
        for reddit in cluster:
            label.append(labels[reddit])
        cluster_label = Counter(label).most_common()[0][0]
        correct_clusters = len([reddit for reddit in cluster if labels[reddit] == cluster_label])
        correct += correct_clusters
        percentage.append(correct_clusters/len(cluster))
    if all_subs:
        recall_cluster = correct/all_s
    else:
        recall_cluster = correct/250
    precision_cluster = np.mean(percentage)
    f_m = (2*precision_cluster*recall_cluster)/(precision_cluster + recall_cluster)
    return recall_cluster, precision_cluster, f_m

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


precision = {
    'AGGL':{'normal':[], 'minimum':[], 'maximum':[], 'all':[]},
    'Kmeans':{'normal':[], 'minimum':[], 'maximum':[], 'all':[]}
}

recall = {
    'AGGL':{'normal':[], 'minimum':[], 'maximum':[], 'all':[]},
    'Kmeans':{'normal':[], 'minimum':[], 'maximum':[], 'all':[]}
}

f_measure = {
    'AGGL':{'normal':[], 'minimum':[], 'maximum':[], 'all':[]},
    'Kmeans':{'normal':[], 'minimum':[], 'maximum':[], 'all':[]}
}

for algorithm, vectors in precision.items():
    for cluster_number in RANGES:
        for matrix in vectors.keys():
            data = get_data(
                FILENAME  + matrix + str(cluster_number),
                algorithm
            )
            recalls, precisions, f_measures = evaluate_clusters(
                data,
                get_subreddit_labels('subreddits.txt')
            )
            vectors[matrix].append(precisions)
            recall[algorithm][matrix].append(recalls)
            f_measure[algorithm][matrix].append(f_measures)

plt.figure(figsize=(22, 20), dpi=120)
for number, plot in enumerate(SUBPLOTS):
    plt.subplot(plot)
    plt.plot(
        RANGES,
        recall['AGGL'][MATRIX[number]],
        '-',
        c='b',
        label='Aggl. (Recall)'
    )
    plt.plot(
        RANGES,
        precision['AGGL'][MATRIX[number]],
        ':',
        c='b',
        label='Kmeans (Precision)'
    )
    plt.plot(
        RANGES,
        recall['Kmeans'][MATRIX[number]],
        '-',
        c='r',
        label='Kmeans'+' (Recall)'
    )
    plt.plot(
        RANGES,
        precision['Kmeans'][MATRIX[number]],
        ':',
        c='r',
        label='Kmeans (Precision)'
    )
    plt.title(MATRIX[number] + ' values')
    plt.legend()
plt.subplot(222)
plt.savefig('graphs/precision_recall_cosine_svd.png')
plt.close()
plt.figure(figsize=(22, 20), dpi=120)
for number, plot in enumerate(SUBPLOTS):
    plt.subplot(plot)
    for algorithm, vectors in f_measure.items():
        plt.plot(
            RANGES,
            vectors[MATRIX[number]],
            label=algorithm + ' (F1-Measure)',
        )
    plt.grid(True)
    plt.xlabel('Number of Clusters')
    plt.ylabel('correct classified subreddits')
    plt.title(MATRIX[number] + ' values')
    plt.legend()
plt.savefig('graphs/f_measure_cosine_svd.png')
