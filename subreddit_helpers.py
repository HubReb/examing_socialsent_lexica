#! /usr/bin/env python3
# -*- oding: utf-8 -*-

''' Helper functions to create a SubredditData object '''

import json
import csv
import spacy

def get_subreddits(files):
    """
    Create or load subreddit dictionary

    Arguments:
        files: list of files that will be used to create subreddit dictionary

    Returns:
        dictionary of subreddits
    """
    if "subreddits.json" in files:
        with open("subreddits.json") as subreddits_file:
            return json.load(subreddits_file)
    sub_reddits = {}
    for subreddit in files:
        if not subreddit.endswith("tsv"):
            # file is not a subreddit
            continue
        try:
            with open(subreddit) as f:
                sub_reddits[subreddit] = {}
                tsvreader = csv.reader(f, delimiter="\t")
                for line in tsvreader:
                    sentiment = float(line[1])
                    standard_variance = float(line[2])
                    sub_reddits[subreddit][line[0]] = [sentiment, standard_variance]
        except IOError as exception:
            print(exception)
    with open("subreddits.json", "w") as subreddits_in_file:
        json.dump(sub_reddits, subreddits_in_file)
    return sub_reddits

def get_words_from_scratch(files):
    """ Create word list out of all given subreddits """
    words = set({})
    nlp = spacy.load("en")
    for subreddit in files:
        if not subreddit.endswith("tsv"):
            continue
        try:
            with open(subreddit) as f:
                tsvreader = csv.reader(f, delimiter="\t")
                for line in tsvreader:
                    lemmatized_word = nlp(line[0])
                    words.add(str(lemmatized_word))
        except IOError as exception:
            print(exception)
    words = list(words)
    with open("words.txt", "w") as f:
        f.write("\n".join(words))
    return words

def get_words(files):
    """
    Create or load list of all words in the subreddits

    Arguments;
        files: list of files that will be used to create word list

    Returns:
        list of all words in the subreddits
    """
    if "words.txt" in files:
        with open("words.txt") as f:
            words = f.read().split("\n")
    else:
        print("Need to get all words in subreddits first. This will take a while.")
        words = get_words_from_scratch(files)
    return words
