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
      packages=[
        'examinlexica',
        'examinlexica.clusteredData',
        'examinlexica.evaluate',
        'examinlexica.original'
    ],
      package_data = {'examinlexica' : ['historical/*.tsv', 'subreddits/*.tsv']},
      setup_requires=['numpy', 'scipy'],
      install_requires=['pandas','scikit-learn', 'scipy', 'numpy'],
     )
