#! /usr/bin/env python3
# -*- coding: utf-8 -*-

''' Constants used in various scipts '''

PATH = "/home/students/hubert/socialsent/socialsent/socialsent/data/lexicons/subreddits/cosine/examinlexica/"
PATH_CLUSTERS = "/home/students/hubert/socialsent/socialsent/socialsent/data/lexicons/subreddits/cosine/examinlexica/subreddits/"
PATH_HISTORICAL_ADJECTIVES = "/home/students/hubert/socialsent/socialsent/socialsent/data/lexicons/subreddits/cosine/examinlexica/historical/adjectives/"
PATH_HISTORICAL_FREQUENCIES = "/home/students/hubert/socialsent/socialsent/socialsent/data/lexicons/subreddits/cosine/examinlexica/historical/frequencies/"
ACCEPTABLE_OPTIONS = ["normal", "minimum", "maximum", "all"]
HISTORICAL_OPTIONS = {
        'adjectives' : PATH_HISTORICAL_ADJECTIVES,
        'frequencies' : PATH_HISTORICAL_FREQUENCIES
}
ALGORITHMS = ['aggl', 'Kmeans', 'meanShift', 'HDBSCAN']
