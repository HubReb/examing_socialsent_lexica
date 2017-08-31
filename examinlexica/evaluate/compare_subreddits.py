#! /usr/bin/env python3 # -*- coding: utf-8 -*-

''' Functions to compare sentiment values of words in two subreddits '''

import sys
from collections import defaultdict
import csv

def compare_reddits(reddit_one, reddit_two):
    '''
    Load two subreddits and compare their values.
    Both subreddits must be placed in folder specified by PATH

    Arguments:
        reddit_one: first subreddit
        reddit_two: second subreddit to compare

    Returns:
        dictionay of the form:
        word : value in reddit_one, derivation) (value in reddit two, derivation)
    '''
    if not (reddit_one.endswith('tsv') and reddit_two.endswith('tsv')):
        print('Function arguments must be subreddit.tsv files!')
        sys.exit()
    sentiments_one = get_sentiments(reddit_one)
    sentiments_two = get_sentiments(reddit_two)
    sentiments = add_sentiments(sentiments_one, sentiments_two)
    return sentiments

def get_sentiments(subreddit):
    ''' Returns word sentiments in a subreddit file '''
    with open(subreddit) as f:
        subreddit_sentiments = {}
        tsvreader = csv.reader(f, delimiter='\t')
        for line in tsvreader:
            sentiment = float(line[1])
            derivation = float(line[2])
            subreddit_sentiments[line[0]] = [sentiment, derivation]
        return subreddit_sentiments

def add_sentiments(old_sentiments, new_sentiments):
    ''' Stores sentiments in both subreddits in one dictionary and returns it '''
    sentiments = defaultdict(list)
    for word, sentiment in old_sentiments.items():
        sentiments[word].append(sentiment)
    known_words = sentiments.keys()
    for word, sentiment in new_sentiments.items():
        if word not in known_words:
            sentiments[word].append(['Nan', 'Nan'])
        sentiments[word].append(sentiment)
    return sentiments

def save_results(result_string, name):
    ''' Saves sentiments in a .txt file '''
    with open(name + '.txt', 'w') as f:
        f.write(result_string)

def pretty_print(sentiments):
    ''' Return an easy to read string '''
    result = 'word\t\t sentiment\t\tstandard derivation'
    for word, sentiment in sentiments.items():
        if len(sentiment) == 1:
            sentiment.append(['Nan', 'Nan'])
        result += '\n%s:\t\t%s, %s\t\t%s, %s' % (
            word,
            sentiment[0][0],
            sentiment[0][1],
            sentiment[1][0],
            sentiment[1][1]
        )
    return result

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('usage: python3 compare_subreddits.py subreddit_one subreddit_two filename')
        sys.exit()
    save_results(pretty_print(compare_reddits(sys.argv[1], sys.argv[2])), sys.argv[3])
