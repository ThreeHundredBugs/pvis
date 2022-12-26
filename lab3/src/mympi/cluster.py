import logging
from os import PathLike

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans

logger = logging.getLogger(__file__)


def prepare_data(csv_file: PathLike | str) -> np.ndarray:
    df = pd.read_csv(csv_file)
    for column in df:
        df[column] *= 1 / df[column].max()

    return df.to_numpy()


def best_kmeans(
    data: np.ndarray,
    clusters_num: int = 8,
    attempts: int = 10,
) -> float:
    best_index = 0

    for _ in range(attempts):
        kmeans = KMeans(n_clusters=clusters_num, n_init=1)
        kmeans.fit(data)
        index = pbm_index(data, kmeans)
        logger.debug(f'PBM-index: {index}')
        best_index = max(index, best_index)

    return best_index


def pbm_index(data: np.ndarray, kmeans: KMeans) -> float:
    """
    https://cran.r-project.org/web/packages/clusterCrit/vignettes/clusterCrit.pdf
    """
    centers = kmeans.cluster_centers_
    global_center = np.mean(centers)

    k = len(centers)
    db = pdist(centers).max(initial=float('-infinity'))
    et = np.linalg.norm(data - global_center, axis=1).sum()
    ew = 0
    for point, cluster in zip(data, kmeans.labels_):
        ew += np.linalg.norm(point - centers[cluster])

    return (et * db / (ew * k)) ** 2
