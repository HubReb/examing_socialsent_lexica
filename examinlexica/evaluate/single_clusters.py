#! /usr/bin/env python3
# -*- encoding: utf-8 -*-

''' Calculate number of one element clusters and visualise them '''

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# constants
data = {'normal':{}, 'minimum':{}, 'maximum':{}, 'all':{}}
algorithms = ['Kmeans', 'AGGL']
FILENAME = 'cosine_euclidean_svd_single/comparision_labels_'
RANGE = [i for i in range(2, 106)]
SUBPLOTS = [221, 222, 223, 224]

for matrix in data:
    for alg in algorithms:
        data[matrix][alg] = []

# almost the same as in sizes.py
def get_data(filename, algorithm):
    ''' Read clusters from a file and return number of 1-Element clusters '''
    clusters = []
    algorithms = ['AGGL', 'HDBSCAN', 'Kmeans']
    with open(filename) as f:
        data = f.read().split('\n')
    start_point = False
    ones = 0
    for cluster in data:
        if '__' in cluster:
            continue
        if cluster in algorithms:
            if cluster not in algorithm:
                start_point = False
                continue
            else:
                start_point = True
                continue
        if start_point:
            try:
                number, datapoints = cluster.split(':')
                if number.endswith('-1'):
                    datapoints = datapoints.split(', ')
                    for d in datapoints:
                        clusters.append([d])
                else:
                    if len(datapoints.split()) == 1:
                        ones += 1
                    clusters.append([d.strip(',') for d in datapoints.split()])
            except ValueError:
                continue            # ignore empty line
    return ones

for i in RANGE:
    for matrix in data.keys():
        for algorithm in algorithms:
            filename = FILENAME + matrix + str(i)
            data[matrix][algorithm].append(
                get_data(
                    filename,
                    algorithm
                )
            )
views = list(data.keys())
for number, plot in enumerate(SUBPLOTS):
    for algorithm in algorithms:
        plt.subplot(plot)
        plt.plot(RANGE, data[views[number]][algorithm], label=algorithm)
        plt.title(views[number] + ' values')
        plt.xlabel('Number of Clusters')
        plt.ylabel('clusters of one data point')
    plt.grid(True)
    plt.legend()
plt.subplots_adjust(top=0.92, bottom=0.10, left=0.10, right=0.95, hspace=0.45,
                    wspace=0.4)
plt.savefig('single_clusters_cosine_complete_svd.png')
