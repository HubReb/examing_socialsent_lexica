#!/usr/bin/env python3
# -*- coding: utf-8 -*-


'''
Extract all sentiments in each reddit and create a corresponding feature vector
for each reddit. There are four types of feature vectores created:
    normal: each feature has the sentiment value as specified in the subreddit
    minimum: each feature has the sentiment value normal - standard derivation
    maximum: each feture has the sentiment value normal + standard derivation
    all: all three types of values are included in the feature vector making it
        thrice as big
For each view all feature vectors are clustered to detec similarities between the
subreddits. For now, all clustering algorithms are applied.
'''

import os
import numpy as np
import csv
import sklearn.cluster as cluster
import spacy
import hdbscan
import json


class SubredditData:
    """
    An object containing all raw and processed data in every given subreddit.

    Attributes:
        subreddits:
            dictionary of form: subreddit:word:sentiment, standard derivation
        sentiments:
            dictionary of views (normal, min, max, all) containing feature vectors
            of all subreddits
        order:
            list representing the order of the word in each feature vector
    """

    def __init__(self, path):
        """
        Initialize an object containing all raw and processed data

        Arguments:
            path:   Path to the folder containing the subreddits
        """

        files = os.listdir(path)
        words = self.get_words(files)
        self.subreddits = self.get_subreddits(files)
        self.order = []
        self.sentiments = {
            "normal": [],
            "maximum": [],
            "minimum": [],
            "all" : []
        }
        self.set_sentiments(words)

    def set_sentiments(self, words):
        """Process sentiments in all subreddits, thus create feature matrix

        Arguments:
            words: list of all words used in the subreddits
        """
        for sent in self.subreddits.keys():
            normal, maximum, minimum, sentiment_all = self.get_sentiment(words, sent)
            self.sentiments["normal"].append(normal)
            self.sentiments["maximum"].append(maximum)
            self.sentiments["minimum"].append(minimum)
            self.sentiments["all"].append(sentiment_all)
        self.check_sentiments()
        self.transform_sentiments()

    def transform_sentiments(self):
        """ Transform list of list of word sentiments into a numpy array """
        for view in self.sentiments.keys():
            self.sentiments[view] = np.array(self.sentiments[view])

    def get_sentiment(self, words, sent):
        """
        Create feature vector of a subreddit

        Arguments:
            words: list of all words used in the subreddits
            sent: name of the subreddit to process
        Returns:
            three lists containing either:
                - sentiment of each word:
                  either value in subreddit or 0 if word is not in subreddit
                - minimum sentiment of each word or 0
                - maximum sentiment of each word or 0
                - original sentiment, minimum sentiment, maximum sentiment or 0, 0, 0
        """
        sentiment_normal = []
        sentiment_maximum = []
        sentiment_minimum = []
        sentiment_all = []
        for word in words:
            if word not in self.subreddits[sent].keys():
                sentiment_min, sentiment, sentiment_max = 0, 0,0
            else:
                sentiment, variance = self.subreddits[sent][word]
                sentiment_min = round(sentiment - variance, 2)
                sentiment_max = round(sentiment + variance, 2)
            sentiment_normal.append(sentiment)
            sentiment_minimum.append(sentiment_min)
            sentiment_maximum.append(sentiment_max)
            sentiment_all.extend([sentiment_min, sentiment, sentiment_max])
        return sentiment_normal, sentiment_minimum, sentiment_maximum, sentiment_all

    def check_sentiments(self):
        """ simple check of dimensions of the feature matrices """
        normal = self.sentiments["normal"]
        if len(normal) != len(self.sentiments["minimum"]):
                raise AssertionError("Length of sentiments is not equal")
        if len(normal) != len(self.sentiments["all"]):
            raise AssertionError("Length of sentiment vectors is not correct!")

    def get_subreddits(self, files):
        """
        Create or load subreddit dictionary

        Arguments:
            files: list of files that will be used to create subreddit dictionary

        Returns:
            dictionary of subreddits
        """
        if "subreddits.json" in files:
            with open("subreddits.json") as f:
                return json.load(f)
        subreddits = {}
        for subreddit in files:
            if not subreddit.endswith("tsv"):
                # file is not a subreddit
                continue
            try:
                with open(subreddit) as f:
                    subreddits[subreddit] = {}
                    tsvreader = csv.reader(f, delimiter="\t")
                    for line in tsvreader:
                        sentiment = float(line[1])
                        standard_variance = float(line[2])
                        subreddits[subreddit][line[0]] = [sentiment, standard_variance]
            except Exception as e:
                print(e)
        with open("subreddits.json", "w") as f:
            json.dump(subreddits, f)
        return subreddits

    def get_words(self, files):
        """
        Create or load list of all words in the subreddits

        Arguments;
            files: list of files that will be used to create word list

        Returns:
            list of all words in the subreddits
        """
        if "words.txt" in files:
            with open("words.txt") as f:
                words = f.read().split("\n")
        else:
            print("Need to get all words in subreddits first. This will take up to 2 hours.")
            words = self.get_words_from_scratch(subreddits)
        return words

    def get_words_from_scratch(self, files):
        """ Create word list out of all given subreddits """
        words = set({})
        nlp = spacy.load("en")
        for subreddit in files:
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


    def cluster(self, algorithm, args, kwds, name, view):
        """
        Cluster the subreddits and save results in a npy-file

        Arguments:
            algorithm: the algorithm used for clustering (method call!)
            args: all arguments to be passed to the algorithm method
            kwds: all  arguments to be passed to the algorithm method
            name: name of the file that will be used to save the result
            view: sentiment view (normal, maximum, normal, all)
        """
        labels = algorithm(*args, **kwds).fit_predict(self.sentiments[view])
        np.save(name+"_labels.npy", labels)

    def start_cluster(self, view, times):
        """
        Method to start clustering, resutls are saved in a seperate file

        Arguments:
            view: whether to use original sentiments (normal), negative values (minimum),
                maximum values (maximum) or all three (all)
            times:
                maximum number of clusters to be used for KMeans, spectral and
                agglomerative clustering
        """
        for number in range(3, times):
            self.cluster(
                cluster.MiniBatchKMeans,
                (),
                {'n_clusters':number},
                "mini_batch_kmeans"+str(number)+view,
                view
            )
            self.cluster(
                cluster.SpectralClustering,
                (),
                {'n_clusters':number},
                "spectral"+str(number)+view,
                view
            )
            self.cluster(
                cluster.AgglomerativeClustering,
                (),
                {'n_clusters':number, 'linkage':'ward'},
                "aggl" + str(number) + view,
                view
            )
        self.cluster(
            cluster.MeanShift,
            (0.175,),
            {'cluster_all':False},
            "mean_shift" + view,
            view
        )
        self.cluster(
            hdbscan.HDBSCAN,
            (),
            {'min_cluster_size':15},
            "HDBSCAN" + view,
            view
        )



if __name__ == '__main__':
    path = "/home/students/hubert/socialsent/socialsent/socialsent/data/lexicons/subreddits"
    subreddits = SubredditData(path)
    print("Subreddit Class works!")
    subreddits.start_cluster("normal", 300)
    subreddits.start_cluster("minimum", 300)
    subreddits.start_cluster("maximum", 300)
    subreddits.start_cluster("all", 300)
