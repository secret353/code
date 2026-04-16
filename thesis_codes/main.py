"""
from calendar import c
import os
import torch
from monai.losses import DiceCELoss
from monai.transforms import (
    AsDiscrete,
)

# from monai.visualize import blend_images
from monai.config import print_config
from monai.metrics import DiceMetric

from logger import setup_logger
from utils import load_pretrained

from data.dataloader import get_loader

# from model import get_model

# from pointnet import get_model as get_point_model
from paper3_train import Trainer
from paper3_model import Paper3_model
from paper3_loss import Paper3Loss

# import copy

# ! section: parameters
# dataset_name = "HaNseg"
# dataset_name = "SegRap"
dataset_name = "HaN"
# dataset_name = "hecktor2022"
model_name = "UNet"

max_epochs = 2000
eval_num = 5
global_epoch = 1  # 记录当前epoch
dice_val_best = 0.0
global_epoch_best = 0
num_samples = 1  # transforms中crop采样图片数量
log_name = "train"

save_dice_csv = False
save_pred = False
# need_pretrain = True

save_pred_index = None
# 图像分支的预训练模型地址
current_dir = os.path.dirname(__file__)
module_pretrained_dir = os.path.join(
    current_dir, "UNet/HaN/best_metric_model_0.7114532589912415.pth"
)
# 整个模型的预训练模型地址
pretrained_dir = None


ce_weight = None
in_channels = 1
num_classes = 10
roi_size = (192, 192, 48)

filename = None
if save_dice_csv:
    eval_num = 1
    filename = f"./{model_name}/{dataset_name}/{model_name}_dice.xlsx"
    log_name = "dice_csv"
    max_epochs = 2
if save_pred:
    eval_num = 1
    log_name = "pred"
    max_epochs = 2
    filename = f"./{model_name}/{dataset_name}/"

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print_config()
logger = setup_logger(
    output=os.path.join(current_dir, f"{model_name}/{dataset_name}/"),
    distributed_rank=0,
    name=log_name,
)

# ! section: 加载model
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Paper3_model(
    model_name,
    in_channels,
    32,
    num_classes,
    roi_size,
    # pretrained_dir,
    logger,
)

model = model.to(device)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f"Total parameters count: {pytorch_total_params}")

# ! Load pretrained model weights from pretrained_dir
if pretrained_dir:
    try:
        load_pretrained(model, pretrained_dir, logger, strict=False)
        # load_pretrained(pretrained_model, pretrained_dir, logger, strict=False)
    except Exception as e:
        logger.error(
            "Failed to load pretrained model weights from {} due to {}".format(
                pretrained_dir, e
            )
        )
        raise e
    else:
        logger.info(
            "Successfully loaded pretrained model weights from {}".format(
                pretrained_dir
            )
        )
elif module_pretrained_dir:
    try:
        load_pretrained(model.img_module, module_pretrained_dir, logger, strict=False)
        load_pretrained(
            model.pretrained_img_module, module_pretrained_dir, logger, strict=False
        )
        # load_pretrained(pretrained_model, pretrained_dir, logger, strict=False)
    except Exception as e:
        logger.error(
            "Failed to load image branch pretrained model weights from {} due to {}".format(
                pretrained_dir, e
            )
        )
        raise e
    else:
        logger.info(
            "Successfully loaded image branch pretrained model weights from {}".format(
                pretrained_dir
            )
        )

# ! section: 加载数据
train_loader, val_loader = get_loader(
    dataset_name, save_dice_csv, save_pred, save_pred_index
)

# section optimizer and loss
torch.backends.cudnn.benchmark = True
loss_function = DiceCELoss(to_onehot_y=True, softmax=True, ce_weight=ce_weight)
loss_function = Paper3Loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scaler = torch.cuda.amp.GradScaler()
post_label = AsDiscrete(to_onehot=num_classes)
post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

# section training
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_function=loss_function,
    train_loader=train_loader,
    val_loader=val_loader,
    saler=scaler,
    logger=logger,
    eval_num=eval_num,
    max_epoches=max_epochs,
    dataset_name=dataset_name,
    model_name=model_name,
    post_label=post_label,
    post_pred=post_pred,
    dice_metric=dice_metric,
    roi_size=roi_size,
    filename=filename,
    save_pred=save_pred,
    save_dice_csv=save_dice_csv,
    num_classes=num_classes,
)

while global_epoch < max_epochs:
    global_epoch, dice_val_best, global_epoch_best = trainer.train(
        global_epoch, train_loader, dice_val_best, global_epoch_best
    )

logger.info(
    f"train completed, best_metric: {dice_val_best:.4f} "
    f"at iteration: {global_epoch_best}"
)
"""

import os

import torch
from monai.config import print_config
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete

from data.dataloader import get_loader, normalize_dataset_name
from logger import setup_logger
from paper_hyperparams import get_dataset_hparams, get_shared_hparams
from paper3_loss import Paper3Loss
from paper3_model import Paper3_model
from paper3_train import Trainer
from utils import load_pretrained


def get_bool_env(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y"}


def get_index_list_env(name):
    value = os.environ.get(name)
    if not value:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


dataset_name = normalize_dataset_name(os.environ.get("DATASET_NAME", "BTCV"))
dataset_config = get_dataset_hparams(dataset_name)
shared_hparams = get_shared_hparams()
model_name = os.environ.get("MODEL_NAME", "UNet")
max_epochs = int(os.environ.get("MAX_EPOCHS", 2000))
eval_num = int(os.environ.get("EVAL_NUM", 5))
global_epoch = 1
dice_val_best = 0.0
global_epoch_best = 0
log_name = "train"

save_dice_csv = get_bool_env("SAVE_DICE_CSV", False)
save_pred = get_bool_env("SAVE_PRED", False)
save_pred_index = get_index_list_env("SAVE_PRED_INDEX")

current_dir = os.path.dirname(__file__)
module_pretrained_dir = os.environ.get("MODULE_PRETRAINED_DIR")
pretrained_dir = os.environ.get("PRETRAINED_DIR")

in_channels = 1
base_channels = 32
num_classes = dataset_config["num_classes"]
roi_size = dataset_config["roi_size"]
evolution_iters = int(
    os.environ.get("EVOLUTION_ITERS", shared_hparams["evolution_iters"])
)
topology_interval = int(
    os.environ.get("TOPOLOGY_INTERVAL", shared_hparams["topology_interval"])
)
point_lr = float(os.environ.get("POINT_LR", shared_hparams["point_lr"]))
point_weight_decay = float(
    os.environ.get("POINT_WEIGHT_DECAY", shared_hparams["point_weight_decay"])
)
sliding_window_overlap = float(
    os.environ.get(
        "SLIDING_WINDOW_OVERLAP", shared_hparams["sliding_window_overlap"]
    )
)
filename = None

if save_dice_csv:
    eval_num = 1
    filename = f"./{model_name}/{dataset_name}/{model_name}_dice.xlsx"
    log_name = "dice_csv"
    max_epochs = 2
if save_pred:
    eval_num = 1
    log_name = "pred"
    max_epochs = 2
    filename = f"./{model_name}/{dataset_name}/"

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print_config()
logger = setup_logger(
    output=os.path.join(current_dir, f"{model_name}/{dataset_name}/"),
    distributed_rank=0,
    name=log_name,
)

model = Paper3_model(
    model_name=model_name,
    in_channels=in_channels,
    base_channels=base_channels,
    num_classes=num_classes,
    roi_size=roi_size,
    logger=logger,
    dataset_name=dataset_name,
    evolution_iters=evolution_iters,
    topology_interval=topology_interval,
    enable_topology=True,
).to(device)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f"Total parameters count: {pytorch_total_params}")

if pretrained_dir:
    try:
        load_pretrained(model, pretrained_dir, logger, strict=False)
    except Exception as exc:
        logger.error(
            f"Failed to load pretrained model weights from {pretrained_dir} due to {exc}"
        )
        raise
    else:
        logger.info(
            f"Successfully loaded full pretrained model weights from {pretrained_dir}"
        )
elif module_pretrained_dir:
    try:
        load_pretrained(
            getattr(model.img_module, "backbone", model.img_module),
            module_pretrained_dir,
            logger,
            strict=False,
        )
        load_pretrained(
            getattr(model.pretrained_img_module, "backbone", model.pretrained_img_module),
            module_pretrained_dir,
            logger,
            strict=False,
        )
        model.set_pretrained_init(True)
    except Exception as exc:
        logger.error(
            "Failed to load image branch pretrained model weights from "
            f"{module_pretrained_dir} due to {exc}"
        )
        raise
    else:
        logger.info(
            "Successfully loaded image branch pretrained model weights from "
            f"{module_pretrained_dir}"
        )
else:
    logger.info(
        "No pretrained checkpoint was provided. Coarse meshes will be initialized "
        "from the current image branch predictions."
    )

train_loader, val_loader = get_loader(
    dataset_name, save_dice_csv, save_pred, save_pred_index
)

torch.backends.cudnn.benchmark = True
loss_function = Paper3Loss(dataset_name)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=point_lr, weight_decay=point_weight_decay
)
logger.info(
    "Paper hyperparameters: "
    f"lambda_distance={loss_function.lambda_distance}, "
    f"lambda_normal={loss_function.lambda_normal}, "
    f"lambda_laplacian={loss_function.lambda_laplacian}, "
    f"topology_threshold={model.topology_threshold}, "
    f"evolution_iters={evolution_iters}, "
    f"sliding_window_overlap={sliding_window_overlap}"
)

warmup_epochs = min(shared_hparams["warmup_max_epochs"], max(1, max_epochs // 10))
if max_epochs > warmup_epochs:
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs - warmup_epochs
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )
else:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs
    )

scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
post_label = AsDiscrete(to_onehot=num_classes)
post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_function=loss_function,
    train_loader=train_loader,
    val_loader=val_loader,
    saler=scaler,
    logger=logger,
    eval_num=eval_num,
    max_epoches=max_epochs,
    dataset_name=dataset_name,
    model_name=model_name,
    post_label=post_label,
    post_pred=post_pred,
    dice_metric=dice_metric,
    roi_size=roi_size,
    filename=filename,
    save_pred=save_pred,
    save_dice_csv=save_dice_csv,
    num_classes=num_classes,
    scheduler=scheduler,
    sliding_window_overlap=sliding_window_overlap,
)

while global_epoch <= max_epochs:
    global_epoch, dice_val_best, global_epoch_best = trainer.train(
        global_epoch, train_loader, dice_val_best, global_epoch_best
    )

logger.info(
    f"train completed, best_metric: {dice_val_best:.4f} "
    f"at iteration: {global_epoch_best}"
)
