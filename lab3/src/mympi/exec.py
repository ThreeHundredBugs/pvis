import logging
from os import PathLike
from time import time_ns

import numpy as np
from mpi4py import MPI

from .cluster import prepare_data, best_kmeans
from .types import Tags, KmeansMetadata

logger = logging.getLogger(__file__)


def find_best_clusters(dataset_path: str | PathLike) -> None:
    time_before = time_ns()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    procs = comm.Get_size()

    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s.%(msecs)03d [{rank}]: %(message)s',
        datefmt='%H:%M:%S'
    )
    global logger
    logger = logging.getLogger(__file__)

    attempts_total = 200
    attempt_per_worker = int(np.ceil(attempts_total / procs))
    attempt_per_master = attempts_total - (attempt_per_worker * (procs - 1))

    if rank == 0:
        logger.info(f'Total attempts: {attempts_total}')
        logger.info(f'Running kmeans with {attempt_per_master} attempts')
        run_master(comm, dataset_path, attempt_per_master)
    else:
        logger.info(f'Running kmeans with {attempt_per_worker} attempts')
        run_worker(comm, attempt_per_worker)

    elapsed = time_ns() - time_before
    logger.info(f'Total time: {elapsed / 1_000_000}ms')


def run_master(comm: MPI.Comm, dataset_path: str | PathLike, attempts: int) -> None:
    procs = comm.Get_size()
    logger.info(f'Total processes: {procs}')
    data = prepare_data(dataset_path)
    metadata = KmeansMetadata(data.shape, data.dtype)
    requests = []

    for dest_rank in range(1, procs):
        meta_request = comm.isend(metadata, dest_rank, Tags.KMEANS_METADATA)
        data_request = comm.Isend(data, dest_rank, Tags.KMEANS_DATA)
        requests.append(meta_request)
        requests.append(data_request)

    logger.info(f'Data sent. [{metadata}]')
    index = best_kmeans(data, attempts=attempts)
    logger.info(f'My PBM-index: {index}')

    MPI.Request.Waitall(requests)

    best_index = comm.allreduce(index, op=MPI.MAX)
    if index == best_index:
        logger.info(f'MY PBM-INDEX IS THE BEST: {index} !!!')
    else:
        logger.info(f"My work is worthless...")


def run_worker(comm: MPI.Comm, attempts: int) -> None:
    metadata: KmeansMetadata = comm.recv(source=0, tag=Tags.KMEANS_METADATA)
    logger.info(f'Metadata received. [{metadata}]')

    data = np.empty(metadata.shape, dtype=metadata.dtype)
    comm.Recv(data, source=0, tag=Tags.KMEANS_DATA)
    logger.info(f'Data received. [{data.nbytes} bytes]')

    index = best_kmeans(data, attempts=attempts)
    logger.info(f'My PBM-index: {index}')

    best_index = comm.allreduce(index, op=MPI.MAX)
    if index == best_index:
        logger.info(f'MY PBM-INDEX IS THE BEST: {index} !!!')
    else:
        logger.info(f"My work is worthless...")
