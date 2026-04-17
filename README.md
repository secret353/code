

This repository currently supports only:

- `BTCV`
- `FLARE2021`
- `MM-WHS`

The training pipeline follows the paper-style two-stage workflow:

1. Image-branch pretraining: `pretrain_image_branch.py`
2. Dual-branch joint training: `main.py`

## 1. Linux Environment Setup

The commands below assume:

- Linux
- conda
- CUDA 12.1
- PyTorch 2.3.1

```bash
cd /root/autodl-tmp
conda create -n paper3 python=3.10 -y
conda activate paper3

python -m pip install -U pip setuptools wheel

# PyTorch 2.3.1 + CUDA 12.1
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# Core dependencies
pip install monai nibabel trimesh scipy matplotlib pandas openpyxl termcolor iopath open3d

# PyTorch3D
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Needed only for MM-WHS SyN registration
pip install antspyx
```

If you want to stay closer to the paper's topology optimization module, install `PyMesh`.
If `PyMesh` is not installed, the code can still run, but `ATMO` will be skipped automatically.

```bash
sudo apt update
sudo apt install -y git build-essential cmake gcc g++ libeigen3-dev libgmp-dev libmpfr-dev libboost-all-dev libtbb-dev

cd /root/autodl-tmp
git clone https://github.com/PyMesh/PyMesh.git
cd PyMesh
git submodule update --init

cd third_party
python build.py all
cd ..

python setup.py build
python setup.py install
python -c "import pymesh; print('PyMesh OK')"
```

## 2. Enter the Project Directory

```bash
cd /root/autodl-tmp/thesis_codes
```

## 3. Dataset Preprocessing

### BTCV

```bash
export BTCV_RAW_DIR=/root/autodl-tmp/BTCV
export BTCV_PREPROCESSED_DIR=/root/autodl-tmp/BTCV_preprocessed

python data/BTCV.py \
  --input-dir "$BTCV_RAW_DIR" \
  --output-dir "$BTCV_PREPROCESSED_DIR"
```

### FLARE2021

```bash
export FLARE2021_RAW_DIR=/root/autodl-tmp/FLARE2021
export FLARE2021_PREPROCESSED_DIR=/root/autodl-tmp/FLARE2021_preprocessed

python data/FLARE2021.py \
  --input-dir "$FLARE2021_RAW_DIR" \
  --output-dir "$FLARE2021_PREPROCESSED_DIR"
```

### MM-WHS

```bash
export MMWHS_RAW_DIR=/root/autodl-tmp/MM-WHS
export MMWHS_PREPROCESSED_DIR=/root/autodl-tmp/MMWHS_preprocessed

python data/MMWHS.py \
  --input-dir "$MMWHS_RAW_DIR" \
  --output-dir "$MMWHS_PREPROCESSED_DIR"
```

If you want to run paper-style SyN registration before MM-WHS preprocessing:

```bash
python data/MMWHS.py \
  --run-syn-registration \
  --ct-dir /root/autodl-tmp/MM-WHS/ct_train \
  --mr-dir /root/autodl-tmp/MM-WHS/mr_train \
  --output-dir "$MMWHS_PREPROCESSED_DIR"
```

## 4. Stage 1: Image-Branch Pretraining

This stage uses the paper-style image-branch setup:

- optimizer: `SGD`
- initial learning rate: `0.1`
- momentum: `0.9`
- scheduler: `cosine annealing`

Example for `BTCV`:

```bash
cd /root/autodl-tmp/thesis_codes

export DATASET_NAME=BTCV
export MODEL_NAME=UNet
export MAX_EPOCHS=2000
export EVAL_NUM=5

python pretrain_image_branch.py
```

This stage saves two checkpoints by default:

- `best_metric_model_backbone.pth`
- `best_metric_model_image_branch.pth`

For stage 2, the recommended checkpoint is:

```bash
/root/autodl-tmp/thesis_codes/UNet/BTCV/image_branch_pretrain/best_metric_model_backbone.pth
```

## 5. Stage 2: Dual-Branch Joint Training

This stage loads the stage-1 image-branch backbone and runs the dual-branch model with the paper-aligned setup:

- optimizer: `AdamW`
- initial learning rate: `1e-4`
- scheduler: `warmup + cosine`
- mesh evolution iterations: `60`
- validation overlap: `0.7`
- dataset-specific `lambda` and `zeta` are already built into the code

Example for `BTCV`:

```bash
cd /root/autodl-tmp/thesis_codes

export DATASET_NAME=BTCV
export MODEL_NAME=UNet
export MAX_EPOCHS=2000
export EVAL_NUM=5
export MODULE_PRETRAINED_DIR=/root/autodl-tmp/thesis_codes/UNet/BTCV/image_branch_pretrain/best_metric_model_backbone.pth

python main.py
```

For `FLARE2021` and `MM-WHS`, use the same pattern and only change:

- `DATASET_NAME`
- `MODULE_PRETRAINED_DIR`

## 6. Export Predictions or Dice Tables

### Export predictions

```bash
export SAVE_PRED=true
export SAVE_PRED_INDEX=0,1
python main.py
```

### Export Dice table

```bash
export SAVE_DICE_CSV=true
python main.py
```

## 7. Common Environment Variables

### Image-branch pretraining

- `DATASET_NAME`
- `MODEL_NAME`
- `MAX_EPOCHS`
- `EVAL_NUM`
- `PRETRAINED_DIR`
- `OUTPUT_SUBDIR`
- `IMAGE_BRANCH_LR`
- `IMAGE_BRANCH_MOMENTUM`
- `IMAGE_BRANCH_WEIGHT_DECAY`
- `SLIDING_WINDOW_OVERLAP`

### Dual-branch joint training

- `DATASET_NAME`
- `MODEL_NAME`
- `MAX_EPOCHS`
- `EVAL_NUM`
- `MODULE_PRETRAINED_DIR`
- `PRETRAINED_DIR`
- `POINT_LR`
- `POINT_WEIGHT_DECAY`
- `EVOLUTION_ITERS`
- `TOPOLOGY_INTERVAL`
- `SLIDING_WINDOW_OVERLAP`
- `SAVE_PRED`
- `SAVE_PRED_INDEX`
- `SAVE_DICE_CSV`

## 8. Minimal Environment Check

```bash
python - <<'PY'
import torch, monai, nibabel, trimesh, pandas, open3d
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available(), torch.version.cuda)
print("monai:", monai.__version__)
PY
```

## 9. Recommended Run Order

To reproduce the full workflow:

1. Install the environment
2. Preprocess the dataset
3. Run `pretrain_image_branch.py`
4. Run `main.py` with `best_metric_model_backbone.pth`
5. Export predictions or Dice tables if needed

## 10. Notes

- Only `BTCV`, `FLARE2021`, and `MM-WHS` are supported
- `MODULE_PRETRAINED_DIR` should preferably point to `best_metric_model_backbone.pth`
- If `PyMesh` is missing, topology optimization is skipped automatically
- If `antspyx` is missing, do not run the MM-WHS SyN registration command
