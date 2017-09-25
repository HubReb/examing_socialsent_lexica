examinlexica - evaluate-Module
==============================

## Overview
This module can be used to visualise your results once you have enough
cluster-files.

### Usage
You have three options to visualize your data: `evaluate_auto.py`, `size.py`, 
`single\_clusters.py`.

#### evaluate\_auto.py
Use this script to evaluate your results using purity and adjusted mutual
information. 

In order to do this, you must first have a few more results (at least ten
different cluster numbers). Then you have to call the `evaluate_auto.py` script. This
will evaluate the purity of the clusters, the adjusted Rand Index and the
Fowles Mallows Metric.
The corresponding graphs are then stored in the graphs folder. 
You can use the following arguments for the script: 
* c: number of cluster, the algorithm will evaluate all results starting at the
  specified number
* c\_end: last number of clusters to be evaluated
  e.g: using -c 5 and -c\_end results in the evaluation of the results
  using 5, 6, 7, 8, 9 and 10 clusters
* s: source file, this is where your clustering results have to be stored

A typical call is 
> `python3 evaluate/evaluate_auto.py -c 10 -c_end 100 -s normal_sentimens`

This results in the evaluation of all results using the cluster numbers 10
through 100.

The current version assumes you have used both agglomerative Clustering and
Kmeans clustering. If you have used only one of them, you need to change the
last few lines of evaluate\_auto.py accordingly.

### size.py 
You can get the average size of your clusters using this file. 
Change the constans in this file to fit your setup before using it. Just like
`evaluate_auto.py`, this file assumes you have used both Kmeans and aggl. 
Clustering. It also assumes you have used all four matrices. If you haven't, you
need to change the constants defined in this file.

###  single\_clusters.py
You can get the number of one element clusters using this file. Just like
before, all constants in it must be changed to fit your setup.

