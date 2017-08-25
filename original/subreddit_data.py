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

import pandas as pd

from examinlexica.helpers import get_subreddits
from examinlexica.constants import PATH
from examinlexica.original.data import Data

class SubredditData(Data):
    '''
    An object containing all raw and processed data in every given subreddit.

    Attributes:
        subreddits:
            dictionary of form: subreddit:word:sentiment, standard derivation
        sentiments:
            dictionary of views (normal, min, max, all) containing feature vectors
            of all subreddits
        order:
            list representing the order of the subreddits in the sentiment dictionary
    '''

    def __init__(self, path):
        '''
        Initialize an object containing all raw and processed data

        Arguments:
            path:   Path to the folder containing the subreddits
        '''
        super().__init__(path)
        self.subreddits = get_subreddits(path, self.files)
        self.sentiments = {
            'normal': [],
            'maximum': [],
            'minimum': [],
            'all' : []
        }
        self.set_sentiments()
        self.save_order()

    def set_sentiments(self):
        '''Process sentiments in all subreddits, thus create feature matrix

        Arguments:
            words: list of all words used in the subreddits
        '''
        for sent in self.subreddits.keys():
            self.order.append(sent)
            normal, maximum, minimum, sentiment_all = self.get_sentiment(self.words, sent)
            self.sentiments['normal'].append(normal)
            self.sentiments['maximum'].append(maximum)
            self.sentiments['minimum'].append(minimum)
            self.sentiments['all'].append(sentiment_all)
        self.check_sentiments()
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

    def get_sentiment(self, words, sent):
        '''
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
                - original sentiment, minimum sentiment, maximum sentiment
        '''
        s_normal, s_maximum, s_minimum, s_all = [], [], [], []
        words_with_sentiment = self.subreddits[sent].keys()
        for word in words:
            if word not in words_with_sentiment:
                sentiment_min, sentiment, sentiment_max = 0, 0, 0
            else:
                sentiment, variance = self.subreddits[sent][word]
                sentiment_min = round(sentiment - variance, 2)
                sentiment_max = round(sentiment + variance, 2)
            s_normal.append(sentiment)
            s_minimum.append(sentiment_min)
            s_maximum.append(sentiment_max)
            s_all.extend([sentiment_min, sentiment, sentiment_max])
        return s_normal, s_maximum, s_minimum, s_all

    def check_sentiments(self):
        ''' simple check of dimensions of the feature matrices '''
        normal = self.sentiments['normal']
        if len(normal) != len(self.sentiments['minimum']):
            raise AssertionError('Length of sentiments is not equal')
        if len(normal) != len(self.sentiments['all']):
            raise AssertionError('Length of sentiment vectors is not correct!')

if __name__ == '__main__':
    subreddits = SubredditData(PATH)
    print('Subreddit Class works!')
