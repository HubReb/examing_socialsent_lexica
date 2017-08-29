import hdbscan
import seaborn as sns
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from constants import PATH
from original.subreddit_data import SubredditData

data = SubredditData(PATH)
for v in ['normal', 'minimum', 'maximum', 'all']:
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2).fit(data.sentiments[v])
    sns.distplot(clusterer.outlier_scores_[np.isfinite(clusterer.outlier_scores_)], rug=True)
    plt.savefig(v + '_outliers.png')
