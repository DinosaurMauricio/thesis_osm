from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from dataset.dataset import OSMDataset, UCM_dataset

def create_dataloader(
    dataset,
    batch_size: int,
    split: str,
    distributedgpu: bool = False,
    seed: int = None,
    num_workers: int = 0,
    collate_fn:callable=None,
):
    is_train = split == "train"
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        # Set to False for DDP as we shuffle the data with the DistributedSampler
        shuffle=False if distributedgpu or not is_train else True,
        num_workers=num_workers if not distributedgpu else 0,
        persistent_workers=False,
        sampler=(
            DistributedSampler(
                dataset,
                # In case there is no seed fed, 0 is the default seed for DistributedSampler
                # https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
                seed=seed if seed is not None else 0,
                shuffle=True if is_train else False,
            )
            if distributedgpu
            else None
        ),
        collate_fn=collate_fn,
    )
    return dataloader


def load_dataset(
    dataset_name, dataset_config, split, inference, tokenizer, **kwargs
):
    datasets = dict(osm=OSMDataset, ucm=UCM_dataset)

    assert dataset_name in datasets, f"dataset can only be one of {datasets.keys()}"
    
    if tokenizer is None:
        ValueError("tokenizer was not set")

    # should return a dict with keys 'train' and 'val' with the corresponding data
    return datasets[dataset_name](
            path=dataset_config.path,
            tokenizer=tokenizer,
            split=split,
            inference=inference,
            **kwargs,
            )

def load_datasets(dataset_name, dataset_config, inference, tokenizer, splits=("train", "val"), **extra_params):
    return {
        split: load_dataset(
            dataset_name=dataset_name.lower(),
            dataset_config=dataset_config,
            split=split,
            inference=inference,
            tokenizer=tokenizer,
            **extra_params
        )
        for split in splits
    }

def create_dataloaders(datasets_splits, distributedgpu, seed, num_workers=0, collate_fn=None):
    return {
        split: create_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            split=split,
            distributedgpu=distributedgpu,
            seed=seed,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
        for split, (dataset, batch_size) in datasets_splits.items()
    }