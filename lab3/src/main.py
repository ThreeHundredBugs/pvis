import sys

import numpy as np

from mympi import find_best_clusters

# np.set_printoptions(threshold=sys.maxsize)


def main() -> None:
    find_best_clusters('../DS_2019_public.csv')


if __name__ == '__main__':
    main()
