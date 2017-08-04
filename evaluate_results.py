#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
from collections import defaultdict

class ClusteredSubreddits:

    def __init__(self, path):
        self.clusters = {
            "normal" : {},
            "minimum" : {},
            "maximum" : {},
            "all" : {}
            }
        algorithms = ["meanShift", "HDBSCAN", "aggl", "spectral", "miniBatchKmeans"]
        self.set_clusters(algorithms)
        path = path + "/results"
        clustered_data = os.listdir(path)
        self.get_clusters(clustered_data, path)
        self.subreddit_list = self.get_subreddit_order()

    def set_clusters(self, algorithms):
        for algorithm in algorithms:
            for view in self.clusters.keys():
                self.clusters[view][algorithm] = {}

    def get_clusters(self, data, path):
        path = path + "/"
        for cluster_file in data:
            path_file = path + cluster_file
            view, algorithm, number_of_clusters = cluster_file.split("_")[0:3]
            if number_of_clusters != '':
                self.clusters[view][algorithm][int(number_of_clusters)] = np.load(path_file)
            else:
                self.clusters[view][algorithm] = np.load(path_file)

    def view_cluster(self, algorithm, view, number_of_clusters=0):
        if number_of_clusters == 0:
            return self.clusters[view][algorithm]
        else:
            return self.clusters[view][algorithm][number_of_clusters]

    def view_hist_of_cluster(self, algorithm, view, number_of_clusters=0):
        array = self.view_cluster(algorithm, view, number_of_clusters)
        return np.histogram(array, bins=number_of_clusters)

    def get_subreddit_order(self):
        with open("order_subreddits.txt") as f:
            order = f.read().split("\n")
        return order

    def show_word_order(self):
        with open("words.txt") as f:
            words = f.read()
        return words

def evaluate_time_clusters(data, view, times, algorithm):
    old_result = data.view_cluster(algorithm, view, 3)
    #print(old_result)
    for time in range(4, times):
    #    print("Comparing next pair: ")
        result = data.view_cluster(algorithm, view, time)
        print(data.view_hist_of_cluster(algorithm, view, time))
        if np.array_equal(result, old_result):
            print("No change despite increasing number of clusters")
            continue
        print("Change: ")
        print("Clusters: ")
   #     print(old_result, result)
        print("Differences: ")
        print(result - old_result)
        old_result = np.array(result)

def evaluate_clusters(data, view, algorithm):
    result = data.view_cluster(algorithm, view)
    print(data.view_hist_of_cluster(algorithm, view))
    print(result)


def evaluateMiniBatch(data, view, times):
    evaluate_time_clusters(data, view, times, "miniBatchKmeans")

def evaluateAgg(data, view, times):
    evaluate_time_clusters(data, view, times, "aggl")

def evaluateSpectral(data, view, times):
    evaluate_time_clusters(data, view, times, "spectral")

def evaluateMeanShift(data):
    evaluate_clusters(data, view, "meanShift")

def evaluateHDBSCAN(data):
    evaluate_clusters(data, view, "HDBSCAN")

if __name__ == '__main__':
    from sys import argv
    path = "/home/students/hubert/socialsent/socialsent/socialsent/data/lexicons/subreddits"
    clusters = ClusteredSubreddits(path)
    evaluateMiniBatch(clusters, argv[1], 200)
    print("AGGL\n", "__" * 30)
    evaluateAgg(clusters, argv[1], 200)
    print("SPECTRAL\n", "__" *30)
    evaluateSpectral(clusters, argv[1], 200)
    print("MEANSHIFT\n", "__" *30)
    evaluateMeanShift(clusters, argv[1])
    print("HDBSCAN\n", "__" *30)
    evaluateHDBSCAN(clusters, argv[1])
