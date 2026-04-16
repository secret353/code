import os

from monai.data import CacheDataset, Dataset, ThreadDataLoader

from paper_hyperparams import PAPER_SHARED_HPARAMS, normalize_dataset_name


def get_dataset_assets(dataset_name, stage="point", spatial_size=None):
    normalized_name = normalize_dataset_name(dataset_name)
    if normalized_name == "BTCV":
        from .BTCV import (
            get_transforms,
            train_datalist,
            train_transforms,
            val_datalist,
            val_transforms,
        )
    elif normalized_name == "FLARE2021":
        from .FLARE2021 import (
            get_transforms,
            train_datalist,
            train_transforms,
            val_datalist,
            val_transforms,
        )
    else:
        from .MMWHS import (
            get_transforms,
            train_datalist,
            train_transforms,
            val_datalist,
            val_transforms,
        )

    if stage == "point" and spatial_size is None:
        selected_train_transforms = train_transforms
        selected_val_transforms = val_transforms
    else:
        selected_train_transforms, selected_val_transforms = get_transforms(
            stage=stage, spatial_size=spatial_size
        )

    if not train_datalist or not val_datalist:
        raise RuntimeError(
            f"{normalized_name} datalist is empty. Please run the corresponding "
            "preprocessing script and check the dataset path settings."
        )
    return (
        selected_train_transforms,
        selected_val_transforms,
        list(train_datalist),
        list(val_datalist),
    )


def select_validation_subset(datalist, save_pred_index):
    if save_pred_index is None:
        return datalist
    return [datalist[i] for i in save_pred_index]


def get_loader(
    dataset_name,
    save_dice_csv,
    save_pred,
    save_pred_index,
    stage="point",
    spatial_size=None,
):
    train_transforms, val_transforms, train_datalist, val_datalist = get_dataset_assets(
        dataset_name,
        stage=stage,
        spatial_size=spatial_size,
    )

    if save_dice_csv:
        val_datalist = train_datalist + val_datalist
        train_datalist = train_datalist[:1]
    elif save_pred:
        val_datalist = select_validation_subset(
            train_datalist + val_datalist, save_pred_index
        )
        train_datalist = train_datalist[:1]

    cache_workers = min(8, os.cpu_count() or 1)
    train_ds = CacheDataset(
        data=train_datalist,
        transform=train_transforms,
        cache_rate=1.0,
        num_workers=cache_workers,
    )
    train_loader = ThreadDataLoader(
        train_ds,
        num_workers=0,
        batch_size=PAPER_SHARED_HPARAMS["train_batch_size"],
        shuffle=True,
    )

    val_ds = Dataset(data=val_datalist, transform=val_transforms)
    val_loader = ThreadDataLoader(
        val_ds,
        num_workers=0,
        batch_size=PAPER_SHARED_HPARAMS["val_batch_size"],
    )
    return train_loader, val_loader
