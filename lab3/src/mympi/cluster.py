from os import PathLike

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans


def prepare_data(csv_file: PathLike | str) -> np.ndarray:
    df = pd.read_csv(csv_file)
    array = df.to_numpy()

    return array


def best_kmeans(data: np.ndarray, clusters_num: int = 2, attempts: int = 10) -> None:
    data = np.array([
        [1, 2, 3, 4, 6],
        [1, 2, 3, 4, 7],
        [1, 42, 3, 14, 5],
        [1, 42, 3, 14, 6],
    ])

    for _ in range(attempts):
        kmeans = KMeans(n_clusters=clusters_num, n_init=1)
        kmeans.fit(data)
        index = pbm_index(data, kmeans)
        print(f'PBM Index: {index}')


def pbm_index(data: np.ndarray, kmeans: KMeans) -> float:
    """
    https://cran.r-project.org/web/packages/clusterCrit/vignettes/clusterCrit.pdf
    """
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    global_center = np.mean(centers)

    k = len(centers)
    db = pdist(centers).max(initial=float('-infinity'))
    et = np.linalg.norm(data - global_center, axis=1).sum()
    ew = 0
    for point, cluster in zip(data, labels):
        print(f'{point} - {centers[cluster]}')
        print('-->', np.linalg.norm(point - centers[cluster]))
        ew += np.linalg.norm(point - centers[cluster])

    print(f'data={data}')
    print(f'centers={centers}')
    print(f'labels={labels}')
    print(f'g={global_center}')
    print(f'k={k}')
    print(f'db={db}')
    print(f'et={et}')
    print(f'ew={ew}')

    return (et * db / (ew * k)) ** 2
