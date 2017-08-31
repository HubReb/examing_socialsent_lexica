#! /usr/bin/env python3 # -*- coding: utf-8 -*-

''' Functions to compare sentiment values of words in two subreddits '''

import sys
import argparse
from collections import defaultdict
import csv

from examinlexica.constants import PATH

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
    result = 'word\t\t\t\t\t\tsentiment\t\tstandard derivation'.ljust(20)
    for word, sentiment in sentiments.items():
        if len(sentiment) == 1:
            sentiment.append(['Nan', 'Nan'])
        result += ('\n%s:' % word).ljust(20)
        result += ('%s, %s' % (sentiment[0][0], sentiment[0][1])).rjust(20)
        result += ('%s, %s' % (sentiment[1][0], sentiment[1][1])).rjust(20)
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f',
        '--folder',
        default=PATH + 'subreddits/',
        help='folder containing the subreddits'
    )
    parser.add_argument(
        'lexicon',
        help='first lexicon to be compared',
    )
    parser.add_argument(
        'second_lexicon',
        help='second lexicon to be compared',
    )
    parser.add_argument(
        '-r',
        '--result',
        default=PATH + 'compared_subreddits/',
        help='folder containing the result of the comparision'
    )
    args = vars(parser.parse_args())
    lexicon = args['folder'] + args['lexicon']
    second_lexicon = args['folder'] + args['second_lexicon']
    save_results(
        pretty_print(
            compare_reddits(
                lexicon,
                second_lexicon
            )
        ),
        args['result'] + args['lexicon'] + '_' + args['second_lexicon'] + '_comparision'
    )
