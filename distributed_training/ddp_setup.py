import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run_distributed(function, world_size):
    mp.spawn(function, args=(world_size,), nprocs=world_size, join=True)


def run_optuna_distributed(function, world_size):
    manager = mp.Manager()
    return_dict = manager.dict()
    mp.spawn(
        function,
        args=(
            world_size,
            return_dict,
        ),
        nprocs=world_size,
        join=True,
    )

    return return_dict["study"]
