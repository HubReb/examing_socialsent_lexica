#!/bin/bash

python3 evaluate_all.py normal > comparision_labels_normal.txt
python3 evaluate_all.py minimum > comparision_labels_minimum.txt
python3 evaluate_all.py maximum > comparision_labels_maximum.txt
python3 evaluate_all.py all > comparision_labels_all.txt
