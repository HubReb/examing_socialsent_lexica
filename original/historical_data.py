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
        self.sentiments = []
        self.set_sentiments()
        self.save_order()

    def set_sentiments(self):
        '''
        Process sentiments in all lexica and store them in a dictionary
        (feature matrix)
        '''
        for lexicon in self.lexica.keys():
            self.order.append(lexicon)
            sentiment = self.get_sentiment(lexicon)
            self.sentiments.append(sentiment)
        self.transform_sentiments()
        self.sentiments = pd.DataFrame(self.sentiments, columns=self.words)

    def transform_sentiments(self):
        ''' Transform list of lists of sentiments into a numpy array '''
        self.sentiments = np.array(self.sentiments)

    def get_sentiment(self, lexicon):
        ''' return sentiments of words in lexicon '''
        words_with_sentiment = self.lexica[lexicon].keys()
        sentiment_in_lexicon = []
        for word in self.words:
            if word not in words_with_sentiment:
                sentiment = 0
            else:
                sentiment, _ = self.lexica[lexicon][word]
            sentiment_in_lexicon.append(sentiment)
        return sentiment_in_lexicon

if __name__ == '__main__':
    histo = HistoricalData(PATH_HISTORICAL_ADJECTIVES)
