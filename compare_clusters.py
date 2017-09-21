#! /usr/bin/env python3
# -*- coding: utf-8 -*-

''' Checks for identical clusters in two files and writes them in a new txt-file
    of the form: results + name of algorithm used for clustering.
'''


def get_double(list1, list2):
    ''' Find identical clusters'''
    double = []
    for cluster in list1:
#        if len(cluster) == 1:
#            continue
        cluster = set(cluster)
        for other_cluster in list2:
            other_cluster = set(other_cluster)
            if len(other_cluster) == len(cluster):
                if cluster == other_cluster:
                    double.append(cluster)
    return double

def get_clusters(data_file, other_file, algorithm, other_algorithm=None):
    ''' Return clusters identical clsuters in two files '''
    if not other_algorithm:
        other_algorithm = algorithm
    data_one = get_data(data_file, algorithm)
    data_two = get_data(other_file, other_algorithm)
    cl = get_double(data_one, data_two)
    clusters = ''
    for i in cl:
        clusters += ', '.join(i)
        clusters += '\n'
    return clusters

def get_data(filename, algorithm):
    ''' Return clusters in a txt-file in a useable format '''
    clusters = []
    algorithms = ['AGGL', 'HDBSCAN', 'MEANSHIFT', 'Kmeans']
    with open(filename) as f:
        data = f.read().split('\n')
    start_point = False
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
                    for point in datapoints:
                        clusters.append([point])
                else:
                    clusters.append([point.strip(',') for point in datapoints.split()])
            except ValueError:
                print('ValueError in ', cluster)
    return clusters


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 4:
        print('No clustering algorithm specified.')
        sys.exit()
    if len(sys.argv) > 4:
        identical_clusters = get_clusters(
            sys.argv[1],
            sys.argv[2],
            sys.argv[3],
            sys.argv[4]
        )
        name = 'results'+ '_' + sys.argv[3] + '_' + sys.argv[4]
    else:
        print('Using ' + sys.argv[3] + 'for both files')
        identical_clusters = get_clusters(sys.argv[1], sys.argv[2], sys.argv[3])
        name = 'results'+ '_' + sys.argv[3]

    with open(name, 'w') as f:
        f.write(identical_clusters)
