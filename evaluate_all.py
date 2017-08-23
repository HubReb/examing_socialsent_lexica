#!/usr/bin/env python3
# -*- coding: utf-8 -*-


'''
Create an ClusteredSubreddits object and run basic evaluation on the clusters:
For each clustering algorithm with manually selected number of clusters:
    - compare two following iterations
    - look into histograms of clusters
    - note if there is no longer any change
For all other algorithms:
    - look into histograms of clusters

Further evaluation methods will be implemented later.
'''
from collections import defaultdict

from examinlexica.clustered_subreddits import ClusteredSubreddits
from examinlexica.constants import ACCEPTABLE_OPTIONS, PATH

def evaluate_clusters(data, algorithm, times=0, view=None):
    ''' basic evaluation of algorithms with manually selected n. of clusters '''
    result_total = []
    if times == 0:
        result_total.append(data.view_cluster(algorithm, view))
    else:
        for time in range(2, times+1):
#            result_total.append('----' * 38 + '\nusing %s clusters' % time)
           # result = data.view_cluster(algorithm, view, time)
            result_total.append(data.view_cluster(algorithm, view, time))
#            print(result_total)
    return result_total

def evaluate_clusters_readable_output(data, algorithm, times=0, view=None):
    ''' Returns easily readable output of cluster algorithms '''
    results = evaluate_clusters(data, algorithm, times, view=view)
    result_total = []
    for time, result in enumerate(results):
        if times != 0:
            result_total.append('----' * 38 + '\nusing %s clusters' % (time+3))
        result_total.append(pretty_print(result))
    return '\n'.join(result_total)

def pretty_print(result):
    ''' Return pretty print of cluster entry '''
    result_total = ''
    for key, value in result.items():
        result_total += '\nCluster %i: %s' % (key, ', '.join(value))
    return result_total

if __name__ == '__main__':
    import sys
    clusters = ClusteredSubreddits(PATH)
