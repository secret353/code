import argparse
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from monai.transforms import (
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
)
from paper_hyperparams import PAPER_SHARED_HPARAMS

try:
    import nibabel as nib
except ModuleNotFoundError:
    nib = None


PIXDIM = (1.0, 1.0, 1.0)
BBOX_MARGIN = (16, 16, 16)
SPATIAL_SIZE = PAPER_SHARED_HPARAMS["point_roi_size"]
NUM_SAMPLES = PAPER_SHARED_HPARAMS["train_num_samples"]
TRAIN_CASES = 24

DEFAULT_RAW_DIRS = [
    os.environ.get("BTCV_RAW_DIR"),
    "/root/autodl-tmp/BTCV",
    "/root/autodl-tmp/BTCV/RawData/Training",
    "/root/autodl-tmp/Abdomen/RawData/Training",
]

DEFAULT_PREPROCESSED_DIRS = [
    os.environ.get("BTCV_PREPROCESSED_DIR"),
    "/root/autodl-tmp/BTCV_preprocessed",
]


def _find_existing_dir(candidates: Sequence[Optional[str]]) -> Optional[str]:
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def _get_pair_dirs(data_dir: str) -> Tuple[str, str]:
    pair_candidates = [
        ("imagesTr", "labelsTr"),
        ("img", "label"),
        ("images", "labels"),
    ]
    for image_dir_name, label_dir_name in pair_candidates:
        image_dir = os.path.join(data_dir, image_dir_name)
        label_dir = os.path.join(data_dir, label_dir_name)
        if os.path.isdir(image_dir) and os.path.isdir(label_dir):
            return image_dir, label_dir
    raise FileNotFoundError(
        "Unable to find BTCV image/label folders under '{}'. "
        "Expected one of: imagesTr/labelsTr, img/label, images/labels.".format(data_dir)
    )


def _extract_case_id(filename: str) -> str:
    stem = filename
    for suffix in (".nii.gz", ".nii"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    digits = re.findall(r"\d+", stem)
    return digits[-1] if digits else stem


def _match_label_path(image_name: str, label_dir: str) -> str:
    case_id = _extract_case_id(image_name)
    candidates = [
        os.path.join(label_dir, image_name),
        os.path.join(label_dir, f"label{case_id}.nii.gz"),
        os.path.join(label_dir, f"label{case_id}.nii"),
        os.path.join(label_dir, f"img{case_id}.nii.gz"),
        os.path.join(label_dir, f"img{case_id}.nii"),
        os.path.join(label_dir, f"{case_id}.nii.gz"),
        os.path.join(label_dir, f"{case_id}.nii"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "Cannot find the label file matching image '{}' under '{}'.".format(
            image_name, label_dir
        )
    )


def get_img_label_path(data_dir: str) -> List[Dict[str, str]]:
    image_dir, label_dir = _get_pair_dirs(data_dir)
    datalist: List[Dict[str, str]] = []
    image_names = sorted(
        file_name
        for file_name in os.listdir(image_dir)
        if file_name.endswith(".nii.gz") or file_name.endswith(".nii")
    )
    for image_name in image_names:
        image_path = os.path.join(image_dir, image_name)
        label_path = _match_label_path(image_name, label_dir)
        datalist.append({"image": image_path, "label": label_path})
    return datalist


def split_datalist(
    datalist: Sequence[Dict[str, str]], train_cases: int = TRAIN_CASES
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    sorted_datalist = sorted(datalist, key=lambda item: _extract_case_id(item["image"]))
    train_datalist = list(sorted_datalist[:train_cases])
    val_datalist = list(sorted_datalist[train_cases:])
    return train_datalist, val_datalist


def _tensor_to_numpy(data):
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    return np.asarray(data)


def _save_case(array, affine, output_path: str, dtype=None) -> None:
    if nib is None:
        raise ImportError(
            "nibabel is required for saving preprocessed BTCV volumes. "
            "Please install nibabel before running the offline preprocessing script."
        )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np_array = _tensor_to_numpy(array)
    if dtype is not None:
        np_array = np_array.astype(dtype)
    nib.save(nib.Nifti1Image(np_array, affine), output_path)


def build_preprocess_transforms(
    pixdim: Tuple[float, float, float] = PIXDIM,
    margin: Tuple[int, int, int] = BBOX_MARGIN,
):
    return Compose(
        [
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=pixdim,
                mode=("bilinear", "nearest"),
            ),
            CropForegroundd(
                keys=["image", "label"],
                source_key="label",
                margin=margin,
                allow_smaller=True,
            ),
        ]
    )


def preprocess_btcv_dataset(
    input_dir: str,
    output_dir: str,
    pixdim: Tuple[float, float, float] = PIXDIM,
    margin: Tuple[int, int, int] = BBOX_MARGIN,
) -> None:
    datalist = get_img_label_path(input_dir)
    transforms = build_preprocess_transforms(pixdim=pixdim, margin=margin)

    image_output_dir = os.path.join(output_dir, "imagesTr")
    label_output_dir = os.path.join(output_dir, "labelsTr")
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(label_output_dir, exist_ok=True)

    for index, item in enumerate(datalist, start=1):
        data = transforms(item)
        image_name = os.path.basename(item["image"])
        label_name = os.path.basename(item["label"])

        image_affine = _tensor_to_numpy(data["image_meta_dict"]["affine"])
        label_affine = _tensor_to_numpy(data["label_meta_dict"]["affine"])

        _save_case(
            data["image"][0],
            image_affine,
            os.path.join(image_output_dir, image_name),
            dtype=np.float32,
        )
        _save_case(
            data["label"][0],
            label_affine,
            os.path.join(label_output_dir, label_name),
            dtype=np.uint8,
        )
        print(
            "Preprocessed {}/{}: {}".format(
                index, len(datalist), _extract_case_id(image_name)
            )
        )


def get_default_data_dir(prefer_preprocessed: bool = True) -> Optional[str]:
    if prefer_preprocessed:
        return _find_existing_dir(DEFAULT_PREPROCESSED_DIRS) or _find_existing_dir(
            DEFAULT_RAW_DIRS
        )
    return _find_existing_dir(DEFAULT_RAW_DIRS) or _find_existing_dir(
        DEFAULT_PREPROCESSED_DIRS
    )


def get_default_raw_dir() -> Optional[str]:
    return _find_existing_dir(DEFAULT_RAW_DIRS)


def get_default_preprocessed_dir() -> Optional[str]:
    return _find_existing_dir(DEFAULT_PREPROCESSED_DIRS)


def _get_stage_spatial_size(spatial_size: Optional[Tuple[int, int, int]]):
    return tuple(spatial_size or SPATIAL_SIZE)


def get_transforms(
    stage: str = "point", spatial_size: Optional[Tuple[int, int, int]] = None
):
    if stage not in {"image", "point"}:
        raise ValueError(f"Unsupported training stage: {stage!r}")

    spatial_size = _get_stage_spatial_size(spatial_size)
    common = [
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=PIXDIM, mode=("bilinear", "nearest")),
        CropForegroundd(
            keys=["image", "label"],
            source_key="label",
            margin=BBOX_MARGIN,
            allow_smaller=True,
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
    ]
    train_augments = [
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=PAPER_SHARED_HPARAMS["augmentation_flip_prob"],
        ),
        RandRotate90d(
            keys=["image", "label"],
            prob=PAPER_SHARED_HPARAMS["augmentation_rotation_prob"],
            max_k=3,
        ),
    ]
    if stage == "point":
        train_augments.append(
            RandShiftIntensityd(
                keys=["image"],
                offsets=PAPER_SHARED_HPARAMS["augmentation_intensity_shift_offset"],
                prob=PAPER_SHARED_HPARAMS["augmentation_intensity_shift_prob"],
            )
        )

    train_transforms = Compose(
        common
        + [
            SpatialPadd(keys=["image", "label"], spatial_size=spatial_size),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=spatial_size,
                pos=1,
                neg=1,
                num_samples=NUM_SAMPLES,
                image_key="image",
                image_threshold=0,
            ),
        ]
        + train_augments
    )
    val_transforms = Compose(
        common
        + [
            SpatialPadd(keys=["image", "label"], spatial_size=spatial_size),
            ToTensord(keys=["image", "label"]),
        ]
    )
    return train_transforms, val_transforms


train_transforms, val_transforms = get_transforms(stage="point")

_default_data_dir = get_default_data_dir(prefer_preprocessed=True)
if _default_data_dir is not None:
    data_path = get_img_label_path(_default_data_dir)
    train_datalist, val_datalist = split_datalist(data_path)
else:
    data_path = []
    train_datalist, val_datalist = [], []


def _parse_margin(values: Sequence[int]) -> Tuple[int, int, int]:
    if len(values) == 1:
        return (values[0], values[0], values[0])
    if len(values) == 3:
        return (values[0], values[1], values[2])
    raise ValueError("margin must contain either 1 or 3 integers.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess BTCV with 1 mm isotropic resampling and label-based ROI cropping."
    )
    parser.add_argument(
        "--input-dir",
        default=get_default_raw_dir(),
        help="BTCV raw data directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=get_default_preprocessed_dir()
        or os.path.join(os.getcwd(), "BTCV_preprocessed"),
        help="Directory for the preprocessed BTCV files.",
    )
    parser.add_argument(
        "--margin",
        nargs="+",
        type=int,
        default=list(BBOX_MARGIN),
        help="ROI margin in voxels after resampling. Use one value or three values.",
    )
    args = parser.parse_args()

    if not args.input_dir:
        raise ValueError(
            "No BTCV input directory was found. Please pass --input-dir explicitly."
        )

    preprocess_btcv_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        pixdim=PIXDIM,
        margin=_parse_margin(args.margin),
    )
