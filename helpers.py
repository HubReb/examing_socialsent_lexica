#! /usr/bin/env python3
# -*- oding: utf-8 -*-

''' Helper functions to create a SubredditData object '''

import json
import csv
import spacy

def get_subreddits(path, files):
    """
    Create or load subreddit dictionary

    Arguments:
        files: list of files that will be used to create subreddit dictionary

    Returns:
        dictionary of subreddits
    """
    subreddits_file = path + "subreddits.json"
    if "subreddits.json" in files:
        with open(subreddits_file) as subreddits_file:
            return json.load(subreddits_file)
    sub_reddits = create_data(files)
    with open(subreddits_file, "w") as subreddits_in_file:
        json.dump(sub_reddits, subreddits_in_file)
    return sub_reddits

def get_historical_adj(path, files):
    ''' Create or load historical dictionary

    Arguments:
        files: list of files used to create the dictionary
    Returns:
        dictionary of historical lexica
    '''
    adjectives_file = path + "adjectives.json"
    if adjectives_file in files:
        with open(adjectives_file) as f:
            return json.load(f)
    adjectives = create_data(path, files)
    with open(path + "adjectives.json", "w") as f:
        json.dump(adjectives, f)
    return adjectives

def get_historical_freq(path, files):
    ''' Create or load historical dictionary

    Arguments:
        files: list of files used to create the dictionary
    Returns:
        dictionary of historical lexica
    '''
    if "frequencies.json" in files:
        with open("frequencies.json") as f:
            return json.load(f)
    frequencies = create_data(files)
    with open(path + "frequencies.json", "w") as f:
        json.dump(frequencies, f)
    return frequencies

def create_data(path, files):
    ''' Create a dictionary containing all files and and the sentiment of each word '''
    data = {}
    for data_file in files:
        if not data_file.endswith("tsv"):
            # file is not a subreddit
            continue
        with open(path + data_file) as f:
            data[data_file] = {}
            tsvreader = csv.reader(f, delimiter="\t")
            for line in tsvreader:
                sentiment = float(line[1])
                standard_variance = float(line[2])
                data[data_file][line[0]] = [sentiment, standard_variance]
    return data

def get_words_from_scratch(path, files):
    """ Create word list out of all given subreddits """
    words = set({})
    nlp = spacy.load("en")
    files = [path +  f for f in files]
    for data_file in files:
        if not data_file.endswith("tsv"):
            continue
        with open(data_file) as f:
            tsvreader = csv.reader(f, delimiter="\t")
            for line in tsvreader:
                lemmatized_word = nlp(line[0])
                words.add(str(lemmatized_word))
    words = list(words)
    with open(path + "words.txt", "w") as f:
        f.write("\n".join(words))
    return words

def get_words(path, files):
    """
    Create or load list of all words in the data filess

    Arguments:
        files: list of files that will be used to create word list

    Returns:
        list of all words in the data filess
    """
    words_file = path + "words.txt"
    if "words.txt" in files:
        with open(words_file) as f:
            words = f.read().split("\n")
    else:
        print("Need to get all words first. This will take a while.")
        words = get_words_from_scratch(path, files)
    return words

def get_lexica_order(path):
    ''' Return order of the feature vectors in matrix '''
    with open(path + 'order.txt') as f:
        order = f.read().split('\n')
    return order
