from copy import deepcopy


DATASET_NAME_ALIASES = {
    "MMWHS": "MM-WHS",
    "MM_WHS": "MM-WHS",
}


PAPER_SHARED_HPARAMS = {
    # Explicitly reported in the manuscript.
    "point_roi_size": (96, 96, 96),
    "cross_slice_layers": 4,
    "cross_slice_window": 3,
    "evolution_iters": 60,
    "point_lr": 1e-4,
    "sliding_window_overlap": 0.7,
    "augmentation_flip_prob": 0.10,
    "augmentation_rotation_prob": 0.10,
    "augmentation_intensity_shift_prob": 0.50,
    "augmentation_intensity_shift_offset": 0.10,
    "image_branch_optimizer": "SGD",
    "image_branch_lr": 0.1,
    "image_branch_momentum": 0.9,
    "image_branch_weight_decay": 0.0,
    "point_branch_optimizer": "AdamW",
    # Implementation defaults that are kept configurable because the manuscript
    # does not provide a single fixed value.
    "point_weight_decay": 1e-5,
    "warmup_max_epochs": 50,
    "topology_interval": 10,
    "train_num_samples": 1,
    "train_batch_size": 1,
    "val_batch_size": 1,
}


PAPER_DATASET_CONFIGS = {
    "BTCV": {
        "num_classes": 14,
        "roi_size": (96, 96, 96),
        "topology_threshold": 0.18,
        "lambda_distance": 1.2,
        "lambda_normal": 0.8,
        "lambda_laplacian": 0.2,
    },
    "FLARE2021": {
        "num_classes": 5,
        "roi_size": (96, 96, 96),
        "topology_threshold": 0.16,
        "lambda_distance": 1.0,
        "lambda_normal": 0.8,
        "lambda_laplacian": 0.3,
    },
    "MM-WHS": {
        "num_classes": 5,
        "roi_size": (96, 96, 96),
        "topology_threshold": 0.22,
        "lambda_distance": 1.4,
        "lambda_normal": 1.0,
        "lambda_laplacian": 0.3,
    },
}


def normalize_dataset_name(dataset_name: str) -> str:
    normalized_name = DATASET_NAME_ALIASES.get(dataset_name, dataset_name)
    if normalized_name not in PAPER_DATASET_CONFIGS:
        raise ValueError(
            "Only BTCV, FLARE2021 and MM-WHS are supported, "
            f"but got dataset_name={dataset_name!r}."
        )
    return normalized_name


def get_shared_hparams():
    return deepcopy(PAPER_SHARED_HPARAMS)


def get_dataset_hparams(dataset_name: str):
    normalized_name = normalize_dataset_name(dataset_name)
    config = deepcopy(PAPER_DATASET_CONFIGS[normalized_name])
    config["dataset_name"] = normalized_name
    return config
