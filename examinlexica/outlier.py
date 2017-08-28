import hdbscan
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from constants import PATH
from original.subreddit_data import SubredditData

data = SubredditData(PATH)
clusterer = hdbscan.HDBSCAN(min_cluster_size=2).fit(data.sentiments['normal'])
#print(clusterer.outlier_scores_)
threshold = pd.Series(clusterer.outlier_scores_).quantile(0.9)
outliers = np.where(clusterer.outlier_scores_ > threshold)[0]
#plt.scatter(*data.T, s=50, linewidth=0, c='gray', alpha=0.25)
plt.scatter(outliers.T, s=50, linewidth=0, c='red', alpha=0.5)

#sns.distplot(clusterer.outlier_scores_[np.isfinite(clusterer.outlier_scores_)], rug=True)
#plt.show()
