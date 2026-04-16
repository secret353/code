# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import SkipConnection
from monai.utils import alias, deprecated_arg, export

__all__ = ["UNet", "Unet"]


class GlobalContextFunction(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.key_proj = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.value_proj = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, feature_map: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        key = self.key_proj(feature_map).flatten(start_dim=3).permute(0, 2, 3, 1)
        value = self.value_proj(feature_map).flatten(start_dim=3).permute(0, 2, 3, 1)
        scores = torch.tanh((key * query.unsqueeze(2)).sum(dim=-1) + self.bias)
        weights = torch.softmax(scores, dim=-1)
        return torch.sum(weights.unsqueeze(-1) * value, dim=2)


class CrossSliceContextLayer(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(channels)
        self.query_proj = nn.Linear(channels, channels)
        self.global_context = GlobalContextFunction(channels)
        self.relation_norm = nn.LayerNorm(channels)
        self.relation_proj = nn.Linear(channels, channels)
        self.context_scale = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        self.context_shift = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        self.out_norm = nn.InstanceNorm3d(channels, affine=True)
        self.out_act = nn.ReLU(inplace=True)

    def summarize_center(self, feature_map: torch.Tensor) -> torch.Tensor:
        query = feature_map.mean(dim=(-1, -2)).permute(0, 2, 1)
        return self.query_proj(self.query_norm(query))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padded = F.pad(x, (0, 0, 0, 0, 1, 1), mode="replicate")
        left = padded[:, :, :-2]
        center = padded[:, :, 1:-1]
        right = padded[:, :, 2:]

        center_query = self.summarize_center(center)
        r_l = self.global_context(left, center_query)
        r_r = self.global_context(right, center_query)
        r_lc = self.global_context(center, r_l)
        r_cr = self.global_context(center, r_r)

        aggregated = torch.stack([r_l, r_r, r_lc, r_cr], dim=2).mean(dim=2)
        aggregated = self.relation_proj(self.relation_norm(aggregated))
        aggregated = aggregated.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)

        scale = torch.sigmoid(self.context_scale(aggregated))
        shift = self.context_shift(aggregated)
        updated = center + scale * center + shift
        return self.out_act(self.out_norm(updated))


class ChannelFeatureFusion(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.cross_linear = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        self.decoder_linear = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        self.fuse = nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, cross_slice_feat: torch.Tensor, decoder_feat: torch.Tensor
    ) -> torch.Tensor:
        cross_gap = cross_slice_feat.mean(dim=(-1, -2), keepdim=True)
        decoder_gap = decoder_feat.mean(dim=(-1, -2), keepdim=True)
        mask = self.cross_linear(cross_gap) + self.decoder_linear(decoder_gap)
        refined_cross = torch.sigmoid(mask) * cross_slice_feat
        return self.fuse(torch.cat([refined_cross, decoder_feat], dim=1))


@export("monai.networks.nets")
@alias("Unet")
class UNet(nn.Module):
    """
    Enhanced version of UNet which has residual units implemented with the ResidualUnit class.
    The residual part uses a convolution to change the input dimensions to match the output dimensions
    if this is necessary but will use nn.Identity if not.
    Refer to: https://link.springer.com/chapter/10.1007/978-3-030-12029-0_40.

    Each layer of the network has a encode and decode path with a skip connection between them. Data in the encode path
    is downsampled using strided convolutions (if `strides` is given values greater than 1) and in the decode path
    upsampled using strided transpose convolutions. These down or up sampling operations occur at the beginning of each
    block rather than afterwards as is typical in UNet implementations.

    To further explain this consider the first example network given below. This network has 3 layers with strides
    of 2 for each of the middle layers (the last layer is the bottom connection which does not down/up sample). Input
    data to this network is immediately reduced in the spatial dimensions by a factor of 2 by the first convolution of
    the residual unit defining the first layer of the encode part. The last layer of the decode part will upsample its
    input (data from the previous layer concatenated with data from the skip connection) in the first convolution. this
    ensures the final output of the network has the same shape as the input.

    Padding values for the convolutions are chosen to ensure output sizes are even divisors/multiples of the input
    sizes if the `strides` value for a layer is a factor of the input sizes. A typical case is to use `strides` values
    of 2 and inputs that are multiples of powers of 2. An input can thus be downsampled evenly however many times its
    dimensions can be divided by 2, so for the example network inputs would have to have dimensions that are multiples
    of 4. In the second example network given below the input to the bottom layer will have shape (1, 64, 15, 15) for
    an input of shape (1, 1, 240, 240) demonstrating the input being reduced in size spatially by 2**4.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        channels: sequence of channels. Top block first. The length of `channels` should be no less than 2.
        strides: sequence of convolution strides. The length of `stride` should equal to `len(channels) - 1`.
        kernel_size: convolution kernel size, the value(s) should be odd. If sequence,
            its length should equal to dimensions. Defaults to 3.
        up_kernel_size: upsampling convolution kernel size, the value(s) should be odd. If sequence,
            its length should equal to dimensions. Defaults to 3.
        num_res_units: number of residual units. Defaults to 0.
        act: activation type and arguments. Defaults to PReLU.
        norm: feature normalization type and arguments. Defaults to instance norm.
        dropout: dropout ratio. Defaults to no dropout.
        bias: whether to have a bias term in convolution blocks. Defaults to True.
            According to `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_,
            if a conv layer is directly followed by a batch norm layer, bias should be False.
        adn_ordering: a string representing the ordering of activation (A), normalization (N), and dropout (D).
            Defaults to "NDA". See also: :py:class:`monai.networks.blocks.ADN`.

    Examples::

        from monai.networks.nets import UNet

        # 3 layer network with down/upsampling by a factor of 2 at each layer with 2-convolution residual units
        net = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(4, 8, 16),
            strides=(2, 2),
            num_res_units=2
        )

        # 5 layer network with simple convolution/normalization/dropout/activation blocks defining the layers
        net=UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(4, 8, 16, 32, 64),
            strides=(2, 2, 2, 2),
        )

    .. deprecated:: 0.6.0
        ``dimensions`` is deprecated, use ``spatial_dims`` instead.

    Note: The acceptable spatial size of input data depends on the parameters of the network,
        to set appropriate spatial size, please check the tutorial for more details:
        https://github.com/Project-MONAI/tutorials/blob/master/modules/UNet_input_size_constrains.ipynb.
        Typically, when using a stride of 2 in down / up sampling, the output dimensions are either half of the
        input when downsampling, or twice when upsampling. In this case with N numbers of layers in the network,
        the inputs must have spatial dimensions that are all multiples of 2^N.
        Usually, applying `resize`, `pad` or `crop` transforms can help adjust the spatial size of input data.

    """

    @deprecated_arg(
        name="dimensions",
        new_name="spatial_dims",
        since="0.6",
        msg_suffix="Please use `spatial_dims` instead.",
    )
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        feature_channels: int = 32,
        cross_slice_layers: int = 4,
        cross_slice_window: int = 3,
        dimensions: Optional[int] = None,
    ) -> None:

        super().__init__()

        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError(
                "the length of `strides` should equal to `len(channels) - 1`."
            )
        if delta > 0:
            warnings.warn(
                f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used."
            )
        if dimensions is not None:
            spatial_dims = dimensions
        if isinstance(kernel_size, Sequence):
            if len(kernel_size) != spatial_dims:
                raise ValueError(
                    "the length of `kernel_size` should equal to `dimensions`."
                )
        if isinstance(up_kernel_size, Sequence):
            if len(up_kernel_size) != spatial_dims:
                raise ValueError(
                    "the length of `up_kernel_size` should equal to `dimensions`."
                )

        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering
        self.feature_channels = feature_channels
        self.cross_slice_layers = cross_slice_layers
        self.cross_slice_window = cross_slice_window
        self.enable_cross_slice = self.dimensions == 3

        if self.cross_slice_layers < 1:
            raise ValueError("`cross_slice_layers` should be no less than 1.")
        if self.cross_slice_window != 3:
            raise ValueError(
                "The current cross-slice implementation uses the paper's fixed three-slice window."
            )

        def _create_block(
            inc: int,
            outc: int,
            channels: Sequence[int],
            strides: Sequence[int],
            is_top: bool,
        ) -> nn.Module:
            """
            Builds the UNet structure from the bottom up by recursing down to the bottom block, then creating sequential
            blocks containing the downsample path, a skip connection around the previous block, and the upsample path.

            Args:
                inc: number of input channels.
                outc: number of output channels.
                channels: sequence of channels. Top block first.
                strides: convolution stride.
                is_top: True if this is the top block.
            """
            c = channels[0]
            s = strides[0]

            subblock: nn.Module

            if len(channels) > 2:
                subblock = _create_block(
                    c, c, channels[1:], strides[1:], False
                )  # continue recursion down
                upc = c * 2
            else:
                # the next layer is the bottom so stop recursion, create the bottom layer as the sublock for this layer
                subblock = self._get_bottom_layer(c, channels[1])
                upc = c + channels[1]

            down = self._get_down_layer(
                inc, c, s, is_top
            )  # create layer in downsampling path
            up = self._get_up_layer(
                upc, outc, s, is_top
            )  # create layer in upsampling path

            return self._get_connection_block(down, up, subblock)

        self.model = _create_block(
            in_channels, out_channels, self.channels, self.strides, True
        )

        top_skip_channels = self.channels[0] * 2
        projection_stride = tuple(2 for _ in range(self.dimensions))
        self.decoder_projection = Convolution(
            self.dimensions,
            top_skip_channels,
            self.feature_channels,
            strides=projection_stride,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=False,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
        )

        if self.enable_cross_slice:
            self.cross_slice_embed = nn.Sequential(
                nn.Conv3d(
                    top_skip_channels,
                    self.feature_channels,
                    kernel_size=1,
                    bias=False,
                ),
                nn.InstanceNorm3d(self.feature_channels, affine=True),
                nn.ReLU(inplace=True),
            )
            self.cross_slice_blocks = nn.ModuleList(
                [
                    CrossSliceContextLayer(self.feature_channels)
                    for _ in range(self.cross_slice_layers)
                ]
            )
            self.cross_slice_projection = Convolution(
                self.dimensions,
                self.feature_channels,
                self.feature_channels,
                strides=projection_stride,
                kernel_size=self.up_kernel_size,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                conv_only=False,
                is_transposed=True,
                adn_ordering=self.adn_ordering,
            )
            self.feature_fusion = ChannelFeatureFusion(self.feature_channels)
        else:
            self.cross_slice_embed = nn.Identity()
            self.cross_slice_blocks = nn.ModuleList()
            self.cross_slice_projection = nn.Identity()
            self.feature_fusion = nn.Identity()

    def _get_connection_block(
        self, down_path: nn.Module, up_path: nn.Module, subblock: nn.Module
    ) -> nn.Module:
        """
        Returns the block object defining a layer of the UNet structure including the implementation of the skip
        between encoding (down) and decoding (up) sides of the network.

        Args:
            down_path: encoding half of the layer
            up_path: decoding half of the layer
            subblock: block defining the next layer in the network.
        Returns: block for this layer: `nn.Sequential(down_path, SkipConnection(subblock), up_path)`
        """
        return nn.Sequential(down_path, SkipConnection(subblock), up_path)

    def _get_down_layer(
        self, in_channels: int, out_channels: int, strides: int, is_top: bool
    ) -> nn.Module:
        """
        Returns the encoding (down) part of a layer of the network. This typically will downsample data at some point
        in its structure. Its output is used as input to the next layer down and is concatenated with output from the
        next layer to form the input for the decode (up) part of the layer.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        mod: nn.Module
        if self.num_res_units > 0:

            mod = ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                adn_ordering=self.adn_ordering,
            )
            return mod
        mod = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        )
        return mod

    def _get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        """
        Returns the bottom or bottleneck layer at the bottom of the network linking encode to decode halves.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
        """
        return self._get_down_layer(in_channels, out_channels, 1, False)

    def _get_up_layer(
        self, in_channels: int, out_channels: int, strides: int, is_top: bool
    ) -> nn.Module:
        """
        Returns the decoding (up) part of a layer of the network. This typically will upsample data at some point
        in its structure. Its output is used as input to the next layer up.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        conv: Union[Convolution, nn.Sequential]

        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
        )

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_top,
                adn_ordering=self.adn_ordering,
            )
            conv = nn.Sequential(conv, ru)

        return conv

    def _legacy_forward(self, x: torch.Tensor, need_feat=True) -> torch.Tensor:
        return self.forward(x, need_feat=need_feat)
        for i, module in enumerate(self.model):
            x = module(x)
            if i == 1:
                middle_feat = x
        # x = self.model(x)
        # 将middel_feat上采样到x的尺寸
        if need_feat:
            middle_feat = self.middel_feat_conv(middle_feat)
            # middle_feat = nn.functional.interpolate(
            #     middle_feat, size=x.shape[2:], mode="trilinear", align_corners=True
            # )
            return middle_feat, x
        else:
            return x

    def forward(self, x: torch.Tensor, need_feat=True) -> torch.Tensor:
        middle_feat = None
        for i, module in enumerate(self.model):
            x = module(x)
            if i == 1:
                middle_feat = x

        if not need_feat:
            return x

        if middle_feat is None:
            raise RuntimeError(
                "Failed to capture the middle feature for cross-slice modeling."
            )

        decoder_feat = self.decoder_projection(middle_feat)
        if not self.enable_cross_slice:
            return decoder_feat, x

        cross_slice_feat = self.cross_slice_embed(middle_feat)
        for block in self.cross_slice_blocks:
            cross_slice_feat = block(cross_slice_feat)
        cross_slice_feat = self.cross_slice_projection(cross_slice_feat)
        fused_feat = self.feature_fusion(cross_slice_feat, decoder_feat)
        return fused_feat, x


Unet = UNet
