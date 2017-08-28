import hdbscan
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from constants import PATH_HISTORICAL_ADJECTIVES
from original.historical_data import HistoricalData

data = HistoricalData(PATH_HISTORICAL_ADJECTIVES)
clusterer = hdbscan.HDBSCAN(min_cluster_size=2).fit(data.sentiments)
print(clusterer.outlier_scores_)
sns.distplot(clusterer.outlier_scores_[np.isfinite(clusterer.outlier_scores_)], rug=True)
plt.show()
