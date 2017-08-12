#!/usr/bin/env python3
# -*- cosing: utf-8 -*-

''' Helper function to determine order of subreddits in sentiment arrays '''

def get_subreddit_order():
    ''' Return order of the feature vectors in matrix '''
    with open('order_subreddits.txt') as order_file:
        order = order_file.read().split('\n')
    return order
