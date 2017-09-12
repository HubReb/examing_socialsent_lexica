#! /usr/bin/env python3
# -+- coding: utf-8
import sys, os

from examinlexica.evaluate.compare_subreddits import pretty_print, compare_reddits

original_result, original_average, original_average_d, count = pretty_print(compare_reddits(sys.argv[1], sys.argv[2]))

for subreddit in os.listdir('subreddits'):
    if not subreddit.endswith('tsv'):
        continue
    result, average, average_d, not_both = pretty_print(compare_reddits(sys.argv[1], 'subreddits/' + subreddit))
    if not_both <= count:
        print(subreddit)
        print(average, average_d, not_both)
        print()
