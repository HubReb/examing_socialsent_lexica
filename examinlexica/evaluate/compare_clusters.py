#! /usr/bin/env python3
# -*- coding: utf-8 -*-

''' 
Checks for identical clusters in two files and writes them in a new txt-file
of the form: results + name of algorithm used for clustering.
'''

from examinlexica.evaluate.sizes import get_data

def get_double(list1, list2):
    ''' Find identical clusters'''
    double = []
    for cluster in list1:
#        if len(cluster) == 1:          # uncomment this to ignore 1-el. clusters
#            continue
        cluster = set(cluster)
        for other_cluster in list2:
            other_cluster = set(other_cluster)
            if len(other_cluster) == len(cluster):
                if cluster == other_cluster:
                    double.append(cluster)
    return double

def get_clusters(data_file, other_file, algorithm, other_algorithm=None):
    ''' Return identical clusters in two files '''
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

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 4:
        print('usage: python3 compare_clusters.py filename1 filename2 agl1 alg 2')
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
