#! /usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Class HistoricalData:
    Extract all sentiments of the words in each lexicon and store them in a
    feature matrix of form:
    lexicon x words in all lexica

All feature vectors are clusters to detec similiarities between the decades.
Applied clustering algorithms:
    Mini Batch KMeans
    Agglomarative Clustering
    MeanShift Clustering
    HDBSCAN Clustering
'''

import numpy as np
import pandas as pd

from examinlexica.constants import PATH_HISTORICAL_ADJECTIVES
from examinlexica.helpers import get_historical_adj
from examinlexica.original.data import Data

class HistoricalData(Data):
    '''
    Initialize an object containing all sentiments in the given lexica.

    Attributes:
        lexica: dictionary of form:
            lexicon:word:sentiment, standard derivation
        order:
            list representing the order of the lexica in the sentiment dictionary
    '''

    def __init__(self, path):
        '''
        Initialize an object containing all sentiment data

        Arguments:
            path: Path to the folder containing the lexica
        '''
        super().__init__(path)
        self.lexica = get_historical_adj(path, self.files)
        self.sentiments = {
            'normal': [],
            'maximum': [],
            'minimum': [],
            'all' : []
        }
        self.set_sentiments()
        self.save_order()

    def set_sentiments(self):
        '''
        Process sentiments in all lexica and store them in a dictionary
        (feature matrix)
        '''
        for lexicon in self.lexica.keys():
            self.order.append(lexicon)
            normal, maximum, minimum, sentiment_all = self.get_sentiment(lexicon)
            self.sentiments['normal'].append(normal)
            self.sentiments['maximum'].append(maximum)
            self.sentiments['minimum'].append(minimum)
            self.sentiments['all'].append(sentiment_all)
        self.transform_sentiments()
        self.create_data_frame()

    def create_data_frame(self):
        ''' Transform dictionary values from arrays into panda dataframes '''
        for view, sentiments in self.sentiments.items():
            if view == 'all':
                extended_words = []
                for word in self.words:
                    extended_words.extend([word + 'min', word, word + 'max'])
                self.sentiments[view] = pd.DataFrame(sentiments, columns=extended_words)
            else:
                self.sentiments[view] = pd.DataFrame(sentiments, columns=self.words)
            self.sentiments[view].index = self.order

    def get_sentiment(self, lexicon):
        ''' return sentiments of words in lexicon '''
        s_normal, s_maximum, s_minimum, s_all = [], [], [], []
        words_with_sentiment = self.lexica[lexicon].keys()
        for word in self.words:
            if word not in words_with_sentiment:
                sentiment_min, sentiment, sentiment_max = 0, 0, 0
            else:
                sentiment, variance  = self.lexica[lexicon][word]
                sentiment_min = round(sentiment - variance, 2)
                sentiment_max = round(sentiment + variance, 2)
            s_normal.append(sentiment)
            s_minimum.append(sentiment_min)
            s_maximum.append(sentiment_max)
            s_all.extend([sentiment_min, sentiment, sentiment_max])
        return s_normal, s_maximum, s_minimum, s_all

if __name__ == '__main__':
    histo = HistoricalData(PATH_HISTORICAL_ADJECTIVES)
    print(histo.compare_data_frames('1850.tsv', '1900.tsv', 'wretched', 'normal'))
