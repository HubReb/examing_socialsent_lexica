#!/bin/bash

python3 evaluate_results.py normal > comparision_labels_normal.txt
python3 evaluate_results.py minimum > comparision_labels_minimum.txt
python3 evaluate_results.py maximum > comparision_labels_maximum.txt
python3 evaluate_results.py all > comparision_labels_all.txt
