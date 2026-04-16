import inspect

import torch.nn as nn
from monai.networks.nets import SegResNet, SwinUNETR, UNETR, VNet

from paper_hyperparams import PAPER_SHARED_HPARAMS
from UNet_model import UNet


class ImageBackboneAdapter(nn.Module):
    def __init__(self, backbone: nn.Module, out_channels: int, feature_channels: int):
        super().__init__()
        self.backbone = backbone
        self.feature_channels = feature_channels
        self.forward_signature = inspect.signature(self.backbone.forward)
        self.feature_projection = nn.Sequential(
            nn.Conv3d(out_channels, feature_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(feature_channels, affine=True),
            nn.ReLU(inplace=True),
        )

    def backbone_supports_features(self):
        return "need_feat" in self.forward_signature.parameters

    def forward(self, x, need_feat=True):
        if self.backbone_supports_features():
            return self.backbone(x, need_feat=need_feat)

        logits = self.backbone(x)
        if not need_feat:
            return logits
        return self.feature_projection(logits), logits


def build_backbone(
    model_name,
    in_channels=1,
    out_channels=14,
    roi_size=(96, 96, 96),
    feature_channels=32,
):
    cross_slice_layers = PAPER_SHARED_HPARAMS["cross_slice_layers"]
    cross_slice_window = PAPER_SHARED_HPARAMS["cross_slice_window"]
    if model_name == "UNet":
        return UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(64, 128, 256, 512, 1024),
            strides=(2, 2, 2, 2),
            feature_channels=feature_channels,
            cross_slice_layers=cross_slice_layers,
            cross_slice_window=cross_slice_window,
        )
    if model_name == "VNet":
        return VNet(in_channels=in_channels, out_channels=out_channels)
    if model_name == "SegResNet":
        return SegResNet(
            in_channels=in_channels,
            out_channels=out_channels,
            init_filters=32,
            blocks_down=(1, 2, 2, 2, 2),
            blocks_up=(1, 1, 1, 1),
        )
    if model_name == "UNETR":
        return UNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=roi_size,
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
        )
    if model_name == "SwinUNETR":
        return SwinUNETR(
            img_size=roi_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=48,
            use_checkpoint=True,
        )
    raise ValueError(f"model: {model_name} is not supported yet.")


def get_model(
    model_name,
    in_channels=1,
    out_channels=14,
    roi_size=(96, 96, 96),
    feature_channels=32,
):
    backbone = build_backbone(
        model_name=model_name,
        in_channels=in_channels,
        out_channels=out_channels,
        roi_size=roi_size,
        feature_channels=feature_channels,
    )
    return ImageBackboneAdapter(backbone, out_channels, feature_channels)
