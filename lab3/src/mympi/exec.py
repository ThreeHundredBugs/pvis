import logging
from os import PathLike

import numpy as np
from mpi4py import MPI

from .cluster import prepare_data, best_kmeans
from .types import Tags, KmeansMetadata


def find_best_clusters(dataset_path: str | PathLike) -> None:
    comm = MPI.COMM_WORLD
    procs = comm.Get_size()
    rank = comm.Get_rank()
    attempts_per_proc = 1
    config_logging(rank)
    logger = logging.getLogger(__file__)

    if rank == 0:
        logger.info(f'Total processes: {procs}')
        data = prepare_data(dataset_path)
        metadata = KmeansMetadata(data.shape, data.dtype)
        requests = []

        for dest_rank in range(1, procs):
            meta_request = comm.isend(metadata, dest_rank, Tags.KMEANS_METADATA)
            data_request = comm.Isend(data, dest_rank, Tags.KMEANS_DATA)
            requests.append(meta_request)
            requests.append(data_request)

        MPI.Request.Waitall(requests)
        logger.info(f'Data sent. [{metadata}]')
        best_kmeans(data, attempts=attempts_per_proc)
    else:
        metadata: KmeansMetadata = comm.recv(source=0, tag=Tags.KMEANS_METADATA)
        logger.info(f'Metadata received. [{metadata}]')

        data = np.empty(metadata.shape, dtype=metadata.dtype)
        comm.Recv(data, source=0, tag=Tags.KMEANS_DATA)
        logger.info(f'Data received. [{data.nbytes} bytes]')


def config_logging(rank: int) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s.%(msecs)03d [{rank}]: %(message)s',
        datefmt='%H:%M:%S'
    )
