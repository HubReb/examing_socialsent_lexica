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
The installation is started using

> python3 setup.py install

### Running the tests
TODO
## Verion
This is version 1.0
## Authors
R. Hubert, email: hubert@cl.uni-heidelberg.de
## License
## Acknowledgements
All lexica used in this project are taken from the socialsent package. You can
find the original socialsent packages and further explantions regarding the
lexica [here][https://github.com/williamleif/socialsent].

All further work is thus based on the following paper:
> Inducing Domain-Specific Sentiment Lexicons from Unlabeled Corpora.
> Proceedings of EMNLP. 2016.
> --<cite> William L. Hamilton, Kevin Clark, Jure Leskovec, and Dan
> Jurafsky[1]</cite>

[1] http://nlp.stanford.edu/projects/socialsent
