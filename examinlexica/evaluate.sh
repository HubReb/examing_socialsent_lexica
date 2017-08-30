#!/bin/bash

echo "Evaluating everything. This will take up to 40 hours!"
# cluster data first
mkdir compas
for i in {2..125}; do
    python3 cluster.py subreddits -r subreddits_cosine_infinity -c $i -m normal
    python3 evaluate/evaluate_subreddits.py normal -r subreddits_cosine_infinity_results -c $i > compas/comparision_labels_normal$i
done
cp ../order.txt compas/order_normal.txt
for i in {2..125}; do
    python3 cluster.py subreddits -r subreddits -c $i -m minimum
    python3 evaluate/evaluate_subreddits.py minimum -r subreddits_cosine_infinity_results -c $i > compas/comparision_labels_minimum$i
done
cp ../order.txt compas/order_minimum.txt
for i in {2..125}; do
    python3 cluster.py subreddits -r subreddits -c $i -m maximum
    python3 evaluate/evaluate_subreddits.py maximum -r subreddits_cosine_infinity_results -c $i > compas/comparision_labels_maximum$i
done
cp ../order.txt compas/order_maximum.txt
for i in {2..125}; do
    python3 cluster.py subreddits -r subreddits -c $i -m all
    python3 evaluate/evaluate_subreddits.py all -r subreddits_cosine_infinity_results -c $i > compas/comparision_labels_all$i
done
cp ../order.txt compas/order_all.txt

for i in {2..8}; do
    python3 cluster.py frequencies -r frequencies -c $i
    python3 evaluate/evaluate_historical.py frequencies -r frequencies_results -c $i > compas/comparision_labels_frequencies$i
done
# evaluate clusters
for i in {2..8}; do
    python3 cluster.py adjectives -r adjectives -c $i
    python3 evaluate/evaluate_historical.py adjectives -r adjectives_results -c $i > compas/comparision_labels_adjectives$i
done
