#!/usr/bin/env python3
# -*- coding: utf-8 -*-


'''
Functions used to evaluate both CLusteredSubreddits and ClusteredLexica objects.
Create an Clustered* object and run basic evaluation on the clusters:
Return an easily readable output.
'''

from examinlexica.clusteredData.clustered_subreddits import ClusteredSubreddits
from examinlexica.constants import PATH

def evaluate_clusters(data, algorithm, time=0, view=None):
    ''' basic evaluation of algorithms with manually selected n. of clusters '''
    if time == 0:
        return data.view_cluster(algorithm, view)
    else:
        return data.view_cluster(algorithm, view, time)

def evaluate_clusters_readable_output(data, algorithm, time=0, view=None):
    ''' Returns easily readable output of cluster algorithms '''
    result = evaluate_clusters(data, algorithm, time, view=view)
    result_total = []
    result_total.append(pretty_print(result))
    return '\n'.join(result_total)

def pretty_print(result):
    ''' Return pretty print of cluster entry '''
    result_total = ''
    for key, value in result.items():
        result_total += '\nCluster %i: %s' % (key, ', '.join(value))
    return result_total

if __name__ == '__main__':
    clusters = ClusteredSubreddits(PATH, "subreddits_results")
