#!/bin/bash

# cluster data first
python3 cluster.py subreddits -r subreddits -c 200 -m normal
python3 cluster.py subreddits -r subreddits -c 200 -m minimum
python3 cluster.py subreddits -r subreddits -c 200 -m maximum
python3 cluster.py subreddits -r subreddits -c 200 -m all
python3 cluster.py frequencies -r frequencies -c -8
python3 cluster.py adjectives -r adjectives -c -8
# evaluate clusters
mkdir compas
for i in {2..200}; do
    python3 evaluate_subreddits.py normal -r subreddits_results -c $i > compas/comparision_labels_normal$i
    python3 evaluate_subreddits.py minimum -r subreddits_results -c $i > compas/comparision_labels_minimum$i
    python3 evaluate_subreddits.py maximum -r subreddits_results -c $i > compas/comparision_labelsi_maximum$i
    python3 evaluate_subreddits.py all -r subreddits_results -c $i> comparision_labels_all$i
done
for i in {2..8}; do
    python3 evaluate_historical.py frequencies -r frequencies_results -c $i > compas/comparision_labels_adjectives$i
    python3 evaluate_historical.py adjectives -r adjectives_results -c $i > compas/comparision_labels_adjectives$i
done
