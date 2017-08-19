#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
An abstract class with common methods and attributs of classes SubredditData and
HistoricalData
'''

import os
import numpy as np

from helpers import get_words

class Data:
    '''
    An abstract object containing all sentiment information in all files in the folder
    specified by path.
    '''
    def __init__(self, path):
        ''' Initialize an object containing all sentiment data '''
        self.files = os.listdir(path)
        self.words = get_words(self.files)
        self.sentiments = {}

    def transform_sentiments(self):
        ''' Transform list of lists of sentiments into a numpy array '''
        for view, sentiments in self.sentiments.items():
            self.sentiments[view] = np.array(sentiments)

    def save_order(self):
        ''' save order of reddits/decades in featurematrix '''
        with open('order.txt', 'w') as order_in_file:
            order_in_file.write('\n'.join(self.order))
