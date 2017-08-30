#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
An abstract class with common methods and attributs of classes SubredditData and
HistoricalData
'''

import os
import numpy as np

from examinlexica.helpers import get_words

class Data:
    '''
    An abstract object containing all sentiment information in all files in the folder
    specified by path.
    '''
    def __init__(self, path):
        ''' Initialize an object containing all sentiment data '''
        self.files = os.listdir(path)
        self.words = get_words(path, self.files)
        self.path = path
        self.order = []
        self.sentiments = {}

    def transform_sentiments(self):
        ''' Transform list of lists of sentiments into a numpy array '''
        for view, sentiments in self.sentiments.items():
            self.sentiments[view] = np.array(sentiments)

    def compare_data_frames(self, index_one, index_two, word, view=None):
        if view:
            print(self.sentiments[view])
            df = self.sentiments[view].loc[[index_one]][word]
            df2 = self.sentiments[view].loc[[index_two]][word]
        else:
            df = self.sentiments.loc[[index_one]][word]
            df2 = self.sentiments.loc[[index_two]][word]

        return df, df2



    def save_order(self):
        ''' save order of reddits/decades in featurematrix '''
        with open(self.path + '/' + 'order.txt', 'w') as order_in_file:
            order_in_file.write('\n'.join(self.order))
