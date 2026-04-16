import os

import torch
from monai.config import print_config
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete

from data.dataloader import get_loader, normalize_dataset_name
from image_branch_train import ImageBranchTrainer
from logger import setup_logger
from model import get_model
from paper_hyperparams import get_dataset_hparams, get_shared_hparams
from utils import load_pretrained


dataset_name = normalize_dataset_name(os.environ.get("DATASET_NAME", "BTCV"))
dataset_config = get_dataset_hparams(dataset_name)
shared_hparams = get_shared_hparams()
model_name = os.environ.get("MODEL_NAME", "UNet")
max_epochs = int(os.environ.get("MAX_EPOCHS", 2000))
eval_num = int(os.environ.get("EVAL_NUM", 5))
pretrained_dir = os.environ.get("PRETRAINED_DIR")
output_subdir = os.environ.get("OUTPUT_SUBDIR", "image_branch_pretrain")

in_channels = 1
base_channels = 32
num_classes = dataset_config["num_classes"]
roi_size = dataset_config["roi_size"]
sliding_window_overlap = float(
    os.environ.get(
        "SLIDING_WINDOW_OVERLAP", shared_hparams["sliding_window_overlap"]
    )
)
image_branch_lr = float(
    os.environ.get("IMAGE_BRANCH_LR", shared_hparams["image_branch_lr"])
)
image_branch_momentum = float(
    os.environ.get(
        "IMAGE_BRANCH_MOMENTUM", shared_hparams["image_branch_momentum"]
    )
)
image_branch_weight_decay = float(
    os.environ.get(
        "IMAGE_BRANCH_WEIGHT_DECAY", shared_hparams["image_branch_weight_decay"]
    )
)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_dir = os.path.dirname(__file__)
torch.backends.cudnn.benchmark = True

print_config()
logger = setup_logger(
    output=os.path.join(current_dir, f"{model_name}/{dataset_name}/{output_subdir}"),
    distributed_rank=0,
    name="image_pretrain",
)

model = get_model(
    model_name=model_name,
    in_channels=in_channels,
    out_channels=num_classes,
    roi_size=roi_size,
    feature_channels=base_channels,
).to(device)

if pretrained_dir:
    try:
        load_pretrained(getattr(model, "backbone", model), pretrained_dir, logger, strict=False)
    except Exception as exc:
        logger.error(
            "Failed to load image branch pretrained weights from "
            f"{pretrained_dir} due to {exc}"
        )
        raise
    else:
        logger.info(
            f"Successfully loaded image branch initialization from {pretrained_dir}"
        )

train_loader, val_loader = get_loader(
    dataset_name,
    save_dice_csv=False,
    save_pred=False,
    save_pred_index=None,
    stage="image",
    spatial_size=roi_size,
)

optimizer = torch.optim.SGD(
    getattr(model, "backbone", model).parameters(),
    lr=image_branch_lr,
    momentum=image_branch_momentum,
    weight_decay=image_branch_weight_decay,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=max_epochs
)
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
post_label = AsDiscrete(to_onehot=num_classes)
post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

trainable_params = sum(
    p.numel() for p in getattr(model, "backbone", model).parameters() if p.requires_grad
)
logger.info(f"Image branch trainable parameters: {trainable_params}")
logger.info(
    "Image branch pretraining hyperparameters: "
    f"optimizer={shared_hparams['image_branch_optimizer']}, "
    f"lr={image_branch_lr}, "
    f"momentum={image_branch_momentum}, "
    f"weight_decay={image_branch_weight_decay}, "
    f"roi_size={roi_size}, "
    f"sliding_window_overlap={sliding_window_overlap}"
)

trainer = ImageBranchTrainer(
    model=model,
    optimizer=optimizer,
    loss_function=loss_function,
    train_loader=train_loader,
    val_loader=val_loader,
    scaler=scaler,
    logger=logger,
    eval_num=eval_num,
    max_epochs=max_epochs,
    dataset_name=dataset_name,
    model_name=model_name,
    post_label=post_label,
    post_pred=post_pred,
    dice_metric=dice_metric,
    roi_size=roi_size,
    scheduler=scheduler,
    sliding_window_overlap=sliding_window_overlap,
    output_subdir=output_subdir,
)

best_metric, best_epoch = trainer.train()
logger.info(
    f"image branch pretraining completed, best_metric: {best_metric:.4f} at epoch: {best_epoch}"
)
logger.info(
    "Use MODULE_PRETRAINED_DIR="
    + os.path.join(current_dir, f"{model_name}/{dataset_name}/{output_subdir}/best_metric_model_backbone.pth")
    + " for stage-2 dual-branch training."
)
