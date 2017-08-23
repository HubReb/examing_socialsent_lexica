#!/bin/bash

# cluster data first
python3 cluster.py
# evaluate clusters
python3 evaluate_subreddits.py normal > comparision_labels_normal.txt
python3 evaluate_subreddits.py minimum > comparision_labels_minimum.txt
python3 evaluate_subreddits.py maximum > comparision_labels_maximum.txt
python3 evaluate_subreddits.py all > comparision_labels_all.txt
python3 evaluate_historical.py adjectives > comparision_labels_adjectives.txt
python3 evaluate_historical.py frequencies > comparision_labels_frequencies.txt
