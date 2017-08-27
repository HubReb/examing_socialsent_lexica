#! /usr/bin/env python3
# -*- coding: utf-8 -*-
try:
    from setuptools import setup
except:
    from distutils.core import setup

setup(name='examinlexica',
      version='1.0',
      url='https://gitlab.cl.uni-heidelberg.de/hubert/examining_socialsent_lexica',
      author='Rebekka Hubert',
      author_email='hubert@cl.uni-heidelberg.de',
      packages=['', 'clusteredData', 'evaluate', 'original'],
      package_data = {'examinlexica' : ['historical/*.tsv', 'subreddits/*.tsv']},
      install_requires=['csv', 'hdbscan', 'scikit-learn', 'scipy', 'pandas',
                'numpy'],
     )
