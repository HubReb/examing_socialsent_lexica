examinlexica
============

## Overview
examinlexica is a package for inspection the sentiment lexica created by [1].
Using this package you can cluster the lexica and check for semantic
similarities between the lexica.

examinlexica has two purposes: Clustering the lexica created from the 250
biggest subreddits ans clustering the historical lexica created by the authors
of socialsent.
This package provides a fast way of checking for a correlation of theme and 
word sentiments.

### Prerequisites
examinlexica is written in python3, thus all requirements refer to python3 as
well.
The following packages are required to use the examinlexica package:

* numpy
* scipy
* cython
* setuptools

All further requirements can be installed using the setup script. If you do not
wish to use it, all further prerequisites are listed there. 
However, you have to set your PYTHONPATH environment variable yourself then.


### Installing
I strongly recommend using this package in a virtual environment to avoid
difficulties. 
First you have to change the constants of the package to suit your setup.
In order to do this, simply edit the `constants.py` script.
All required constants are explained there.


The actual installation is done using

> `python3 setup.py install`

If you encounter any errors, consult the documentation in the setup script.

### Usage
The easiest way to use this package is to simply call the `get_clusters.py`
script. This allows you to both set all basic parameters for the clustering as
well as see your results immediately.

These basic, required parameters parameters are:
* the feature matrix, specifing whether to consider the standard derivation
  of the word semantics and, if it is to be considered, in which way (you
  can find further information in `cluster.py`
* data: whether to cluster the subreddit lexica or the historical lexica

You can also specify other parameters, though the script will work without
using them:
* the number of clusters to use (both for Kmeans as well as agglomerative
  clustering)
* the folder in which your results are saved (the default folder for
  this is \_results in your current directory).

All parameters of the clustering algorithms themselves must be specified in
`cluster.py`.

A typical call of get\_clusters.py would be:

> `python3 get_clusters.py subreddits normal -c 10 -a Kmeans -r normal_sentiments`

This starts the algorithm with the subreddit lexica, the original socialsent
word sentiments and an end result of ten clusters. Your results are saved in
the file `normal_sentiments_results/Kmeans_normal_10.txt`.
In this txt-file you will find a listing of the achieved clusters in an easily
readable way.

This package also contains the option to evaluate your results and visualize
them in the evaluate module. Their usage is explained in the modules README.

## Version
This is version 1.0

## Authors
R. Hubert, email: hubert@cl.uni-heidelberg.de

## License

## Acknowledgements
All lexica used in this project are taken from the socialsent package. You can
find the original socialsent packages and further explanations regarding the
lexica [here](https://github.com/williamleif/socialsent).

This work is based on the following paper:
> Inducing Domain-Specific Sentiment Lexicons from Unlabeled Corpora.
> Proceedings of EMNLP. 2016.
> --<cite> William L. Hamilton, Kevin Clark, Jure Leskovec, and Dan
> Jurafsky[1]</cite>

[1] http://nlp.stanford.edu/projects/socialsent
