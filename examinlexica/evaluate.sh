#!/bin/bash

echo "Evaluating everything. This will take up to 30 hours!"
# cluster data first
python3 cluster.py subreddits -r subreddits -c 200 -m normal
mkdir compas
for i in {2..200}; do
    python3 evaluate/evaluate_subreddits.py normal -r subreddits_results -c $i > compas/comparision_labels_normal$i
done
cp ../order.txt compas/order_normal.txt
python3 cluster.py subreddits -r subreddits -c 200 -m minimum
for i in {2..200}; do
    python3 evaluate/evaluate_subreddits.py minimum -r subreddits_results -c $i > compas/comparision_labels_minimum$i
done
cp ../order.txt compas/order_minimum.txt
python3 cluster.py subreddits -r subreddits -c 200 -m maximum
for i in {2..200}; do
    python3 evaluate/evaluate_subreddits.py maximum -r subreddits_results -c $i > compas/comparision_labels_maximum$i
done
cp ../order.txt compas/order_maximum.txt
python3 cluster.py subreddits -r subreddits -c 200 -m all
for i in {2..200}; do
    python3 evaluate/evaluate_subreddits.py all -r subreddits_results -c $i > compas/comparision_labels_all$i
done
cp ../order.txt compas/order_all.txt

python3 cluster.py frequencies -r frequencies -c 8
for i in {2..8}; do
    python3 evaluate/evaluate_historical.py frequencies -r frequencies_results -c $i > compas/comparision_labels_adjectives$i
done
python3 cluster.py adjectives -r adjectives -c 8
# evaluate clusters
for i in {2..8}; do
    python3 evaluate/evaluate_historical.py adjectives -r adjectives_results -c $i > compas/comparision_labels_adjectives$i
done
