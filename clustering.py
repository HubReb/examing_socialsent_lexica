#! /ust/bin/env python3
# -*- coding: utf-8 -*_
#

import os
import numpy as np
import csv
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import seaborn as sns
import spacy
import hdbscan

class SubredditData:

    def __init__(self, path):

        subreddits = os.listdir(path)
        self.subreddits_sentiments = {}
        self.subreddits_sentiments_all = {}
        self.subreddits_sentiments_minimum = {}
        self.subreddits_sentiments_maximum = {}

        if "words.txt" in subreddits:
            with open("words.txt") as f:
                words = f.read().split("\n")
        else:
            words = self.get_words(subreddits)

        sents = {}
#        if "sentiment.npy" in subreddits:
#            self.sentiment = np.load("sentiment.npy")
#            self.sentiment_min = np.load("sentimentMin.npy")
#            self.sentiment_max = np.load("sentimentMax.npy")
#            return

        for subreddit in subreddits:
            sentiment_normal = []
            sentiment_minimum = []
            sentiment_maximum = []
            sentiment_all = []
            if not subreddit.endswith("tsv"):
                continue
            try:
                with open(subreddit) as f:
                    tsvreader = csv.reader(f, delimiter="\t")
                    for line in tsvreader:
                        sentiment = float(line[1])
                        standard_variance = float(line[2])
                        sents[line[0]] = [sentiment, standard_variance]
            except Exception as e:
#               raise TypeError("Cannot read file %s" % subreddit)
                print(e)

            for word in words:
                if word not in sents.keys():
                    sentiment_min, sentiment, sentiment_max = 0, 0,0
                else:
                    sentiment, variance = sents[word]
                    sentiment_min, sentiment_max = round(sentiment - variance, 2), round(sentiment + variance, 2)
                sentiment_normal.append(sentiment)
                sentiment_minimum.append(sentiment_min)
                sentiment_maximum.append(sentiment_max)
                sentiment_all.extend([sentiment_min, sentiment, sentiment_max])
               # quick test
            if len(sentiment_normal) != len(sentiment_minimum):
                raise AssertionError("Length of sentiments is not equal")
            if 3*len(sentiment_normal) != len(sentiment_all):
                print("sentiment_normal %s" % len(sentiment_normal))
                print("sentiment_all %s" % 3*len(sentiment_all))
                raise AssertionError("Length of all sentiment vector is not correct!")
            self.subreddits_sentiments[subreddit] = np.array(sentiment_normal)
            self.subreddits_sentiments_minimum[subreddit] = np.array(sentiment_minimum)
            self.subreddits_sentiments_maximum[subreddit] = np.array(sentiment_maximum)
            self.subreddits_sentiments_all[subreddit] = np.array(sentiment_all)


    def get_words(self, subreddits):
        words = set({})
        nlp = spacy.load("en")
        for subreddit in subreddits:
            if not subreddit.endswith("tsv"):
                continue
            try:
                with open(subreddit) as f:
                    tsvreader = csv.reader(f, delimiter="\t")
                    for line in tsvreader:
                        lemmatized_word = nlp(line[0])
                        words.add(str(lemmatized_word))
            except Exception as e:
                print(e)
        words = list(words)
        with open("words.txt", "w") as f:
            f.write("\n".join(words))
        return words


#        np.save("sentiment.npy", self.subreddits_sentiment)
#        np.save("sentimentMin.npy", self.sentiment_min)
#        np.save("sentimentMax.npy", self.sentiment_max)



def plot_clusters(data, algorithm, args, kwds, name):
    labels = algorithm(*args, **kwds).fit_predict(data)
    np.save(name+"_labels.npy", labels)
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
#    plt.scatter([i for i in range(len(data))], data, c=colors, **plot_kwds)
#    frame = plt.gca()
#    frame.axes.get_xaxis().set_visible(False)
 #   frame.axes.get_yaxis().set_visible(False)
 #   plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)


if __name__ == '__main__':
    subreddits = SubredditData(".")
    plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0}
#    arrays = np.array([x for x in subreddits.subreddits_sentiments.values()]).reshape(-1, 1)
    #arrays = np.array([x for x in subreddits.subreddits_sentiments.values()])
    keys = []
    value_normal = []
    value_minimum = []
    value_maximum = []
    value_all = []
    for key in subreddits.subreddits_sentiments.keys():
        keys.append(key)
        value_normal.append(subreddits.subreddits_sentiments[key])
        value_minimum.append(subreddits.subreddits_sentiments_minimum[key])
        value_maximum.append(subreddits.subreddits_sentiments_maximum[key])
        value_all.append(subreddits.subreddits_sentiments_all[key])
    arrays = np.array(value_normal)
    with open("subreddits.txt", "w") as f:
        f.write("\n".join(keys))
    np.save("sentiments_normal.npy", arrays)
    arrays = np.array(value_normal)
    for number in range(3, 100):
        plot_clusters(arrays, cluster.MiniBatchKMeans, (), {'n_clusters':number}, "mini_batch_kmeans"+str(number))
        plot_clusters(arrays, cluster.SpectralClustering, (), {'n_clusters':number}, "spectral"+str(number))
        plot_clusters(arrays, cluster.AgglomerativeClustering, (), {'n_clusters':number, 'linkage':'ward'}, "aggl"+str(number))
    plot_clusters(arrays, cluster.MeanShift, (0.175,), {'cluster_all':False}, "mean_shift")
    plot_clusters(arrays, hdbscan.HDBSCAN, (), {'min_cluster_size':15}, "HDBSCAN")

    arrays = np.array(value_minimum)
    np.save("sentiments_min.npy", arrays)
    for number in range(3, 100):
        plot_clusters(arrays, cluster.MiniBatchKMeans, (), {'n_clusters':number}, "mini_batch_kmeans_min"+str(number))
        plot_clusters(arrays, cluster.SpectralClustering, (), {'n_clusters':number}, "spectral_min"+str(number))
        plot_clusters(arrays, cluster.AgglomerativeClustering, (), {'n_clusters':number, 'linkage':'ward'}, "aggl_min"+str(number))
    plot_clusters(arrays, cluster.MeanShift, (0.175,), {'cluster_all':False}, "mean_shift_min")
    plot_clusters(arrays, cluster.DBSCAN, (), {'eps':0.025}, "DBSCAN_min")
    plot_clusters(arrays, hdbscan.HDBSCAN, (), {'min_cluster_size':15}, "HDBSCAN_min")

    arrays = np.array(value_maximum)
    np.save("sentiments_max.npy", arrays)
    for number in range(3, 100):
        plot_clusters(arrays, cluster.MiniBatchKMeans, (), {'n_clusters':number}, "mini_batch_kmeans_min"+str(number))
        plot_clusters(arrays, cluster.SpectralClustering, (), {'n_clusters':number}, "spectral_min"+str(number))
        plot_clusters(arrays, cluster.AgglomerativeClustering, (), {'n_clusters':number, 'linkage':'ward'}, "aggl_min"+str(number))
    plot_clusters(arrays, cluster.MeanShift, (0.175,), {'cluster_all':False}, "mean_shift_max")
    plot_clusters(arrays, cluster.DBSCAN, (), {'eps':0.025}, "DBSCAN_max")
    plot_clusters(arrays, hdbscan.HDBSCAN, (), {'min_cluster_size':15}, "HDBSCAN_max")


    arrays = np.arrays(value_all)
    np.save("sentiments_all.npy", arrays)
    for number in range(3, 100):
        plot_clusters(arrays, cluster.MiniBatchKMeans, (), {'n_clusters':number}, "mini_batch_kmeans_min"+str(number))
        plot_clusters(arrays, cluster.SpectralClustering, (), {'n_clusters':number}, "spectral_min"+str(number))
        plot_clusters(arrays, cluster.AgglomerativeClustering, (), {'n_clusters':number, 'linkage':'ward'}, "aggl_min"+str(number))
    plot_clusters(arrays, cluster.MeanShift, (0.175,), {'cluster_all':False}, "mean_shift_all")
    plot_clusters(arrays, cluster.DBSCAN, (), {'eps':0.025}, "DBSCAN_all")
    plot_clusters(arrays, hdbscan.HDBSCAN, (), {'min_cluster_size':15}, "HDBSCAN_all")
