from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.segresnet_block import ResBlock, get_conv_layer, get_upsample_layer
from monai.networks.layers.factories import Dropout
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils import UpsampleMode
from monai.losses import DiceLoss
from myCodeArchives.vffc import Bottleneck
from myCodeArchives.nnAdapts import norm_nd_class


##############################################

### Archived SegResNet ova hea
# class SegResNet(nn.Module):
#     """
#     SegResNet based on `3D MRI brain tumor segmentation using autoencoder regularization
#     <https://arxiv.org/pdf/1810.11654.pdf>`_.
#     The module does not include the variational autoencoder (VAE).
#     The model supports 2D or 3D inputs.

#     Args:
#         spatial_dims: spatial dimension of the input data. Defaults to 3.
#         init_filters: number of output channels for initial convolution layer. Defaults to 8.
#         in_channels: number of input channels for the network. Defaults to 1.
#         out_channels: number of output channels for the network. Defaults to 2.
#         dropout_prob: probability of an element to be zero-ed. Defaults to ``None``.
#         act: activation type and arguments. Defaults to ``RELU``.
#         norm: feature normalization type and arguments. Defaults to ``GROUP``.
#         norm_name: deprecating option for feature normalization type.
#         num_groups: deprecating option for group norm. parameters.
#         use_conv_final: if add a final convolution block to output. Defaults to ``True``.
#         blocks_down: number of down sample blocks in each layer. Defaults to ``[1,2,2,4]``.
#         blocks_up: number of up sample blocks in each layer. Defaults to ``[1,1,1]``.
#         upsample_mode: [``"deconv"``, ``"nontrainable"``, ``"pixelshuffle"``]
#             The mode of upsampling manipulations.
#             Using the ``nontrainable`` modes cannot guarantee the model's reproducibility. Defaults to``nontrainable``.

#             - ``deconv``, uses transposed convolution layers.
#             - ``nontrainable``, uses non-trainable `linear` interpolation.
#             - ``pixelshuffle``, uses :py:class:`monai.networks.blocks.SubpixelUpsample`.

#     """

#     def __init__(
#         self,
#         spatial_dims: int = 3,
#         init_filters: int = 8,
#         in_channels: int = 1,
#         out_channels: int = 2,
#         dropout_prob: float | None = None,
#         act: tuple | str = ("RELU", {"inplace": True}),
#         norm: tuple | str = ("GROUP", {"num_groups": 8}),
#         norm_name: str = "",
#         num_groups: int = 8,
#         use_conv_final: bool = True,
#         blocks_down: tuple = (1, 2, 2, 4),
#         blocks_up: tuple = (1, 1, 1),
#         upsample_mode: UpsampleMode | str = UpsampleMode.NONTRAINABLE,
#     ):
#         super().__init__()

#         if spatial_dims not in (2, 3):
#             raise ValueError("`spatial_dims` can only be 2 or 3.")

#         self.spatial_dims = spatial_dims
#         self.init_filters = init_filters
#         self.in_channels = in_channels
#         self.blocks_down = blocks_down
#         self.blocks_up = blocks_up
#         self.dropout_prob = dropout_prob
#         self.act = act  # input options
#         self.act_mod = get_act_layer(act)
#         if norm_name:
#             if norm_name.lower() != "group":
#                 raise ValueError(f"Deprecating option 'norm_name={norm_name}', please use 'norm' instead.")
#             norm = ("group", {"num_groups": num_groups})
#         self.norm = norm
#         self.upsample_mode = UpsampleMode(upsample_mode)
#         self.use_conv_final = use_conv_final
#         self.convInit = get_conv_layer(spatial_dims, in_channels, init_filters)
#         self.down_layers = self._make_down_layers()
#         self.up_layers, self.up_samples = self._make_up_layers()
#         self.conv_final = self._make_final_conv(out_channels)

#         if dropout_prob is not None:
#             self.dropout = Dropout[Dropout.DROPOUT, spatial_dims](dropout_prob)

#     def _make_down_layers(self):
#         down_layers = nn.ModuleList()
#         blocks_down, spatial_dims, filters, norm = (self.blocks_down, self.spatial_dims, self.init_filters, self.norm)
#         for i, item in enumerate(blocks_down):
#             layer_in_channels = filters * 2**i
#             pre_conv = (
#                 get_conv_layer(spatial_dims, layer_in_channels // 2, layer_in_channels, stride=2)
#                 if i > 0
#                 else nn.Identity()
#             )
#             down_layer = nn.Sequential(
#                 pre_conv, *[ResBlock(spatial_dims, layer_in_channels, norm=norm, act=self.act) for _ in range(item)]
#             )
#             down_layers.append(down_layer)
#         return down_layers

#     def _make_up_layers(self):
#         up_layers, up_samples = nn.ModuleList(), nn.ModuleList()
#         upsample_mode, blocks_up, spatial_dims, filters, norm = (
#             self.upsample_mode,
#             self.blocks_up,
#             self.spatial_dims,
#             self.init_filters,
#             self.norm,
#         )
#         n_up = len(blocks_up)
#         for i in range(n_up):
#             sample_in_channels = filters * 2 ** (n_up - i)
#             up_layers.append(
#                 nn.Sequential(
#                     *[
#                         ResBlock(spatial_dims, sample_in_channels // 2, norm=norm, act=self.act)
#                         for _ in range(blocks_up[i])
#                     ]
#                 )
#             )
#             up_samples.append(
#                 nn.Sequential(
#                     *[
#                         get_conv_layer(spatial_dims, sample_in_channels, sample_in_channels // 2, kernel_size=1),
#                         get_upsample_layer(spatial_dims, sample_in_channels // 2, upsample_mode=upsample_mode),
#                     ]
#                 )
#             )
#         return up_layers, up_samples

#     def _make_final_conv(self, out_channels: int):
#         return nn.Sequential(
#             get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.init_filters),
#             self.act_mod,
#             get_conv_layer(self.spatial_dims, self.init_filters, out_channels, kernel_size=1, bias=True),
#         )
#     # Encoder 
#     def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
#         # Initial Convolution
#         # Increase the number of channels to 32 from 4(input) [8x increase]
#         # Keep the spatial dimension fixed (computationally heavy?)
#         x = self.convInit(x) 
#         # Apply Dropout probability to turn off (x percent) of activations randomly
#         if self.dropout_prob is not None:
#             x = self.dropout(x)

#         down_x = []
#         # consists of 4 layers
#         # Layer 1: Identity - ResBlock. Spatail dimension remaines the same (1, 32, 160, 192, 128)
#         # Layer 2: Conv - ResBlock - ResBlock. Halve the spatial dimension (1, 64, 80, 96, 64) )
#         # Layer 3: Conv - ResBlock - ResBlock. Halve the spatial dimension (1, 128, 40, 48, 32)
#         # Layer 4: Conv - ResBlock - ResBlock - ResBlock - ResBlock. Halve the spatial dimension (1, 256, 20, 24, 16)
#         for down in self.down_layers:
#             x = down(x)
#             down_x.append(x)

#         return x, down_x

#     def decode(self, x: torch.Tensor, down_x: list[torch.Tensor]) -> torch.Tensor:
#         for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
#             x = up(x) + down_x[i + 1]
#             x = upl(x)

#         if self.use_conv_final:
#             x = self.conv_final(x)

#         return x

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x, down_x = self.encode(x)
#         down_x.reverse() 

#         x = self.decode(x, down_x)
#         return x
    

##############################################


# Edited SEGRESNET with the VFFC

class SegResNet(nn.Module):
    """
    SegResNet based on `3D MRI brain tumor segmentation using autoencoder regularization
    <https://arxiv.org/pdf/1810.11654.pdf>`_.
    The module does not include the variational autoencoder (VAE).
    The model supports 2D or 3D inputs.

    Args:
        spatial_dims: spatial dimension of the input data. Defaults to 3.
        init_filters: number of output channels for initial convolution layer. Defaults to 8.
        in_channels: number of input channels for the network. Defaults to 1.
        out_channels: number of output channels for the network. Defaults to 2.
        dropout_prob: probability of an element to be zero-ed. Defaults to ``None``.
        act: activation type and arguments. Defaults to ``RELU``.
        norm: feature normalization type and arguments. Defaults to ``GROUP``.
        norm_name: deprecating option for feature normalization type.
        num_groups: deprecating option for group norm. parameters.
        use_conv_final: if add a final convolution block to output. Defaults to ``True``.
        blocks_down: number of down sample blocks in each layer. Defaults to ``[1,2,2,4]``.
        blocks_up: number of up sample blocks in each layer. Defaults to ``[1,1,1]``.
        upsample_mode: [``"deconv"``, ``"nontrainable"``, ``"pixelshuffle"``]
            The mode of upsampling manipulations.
            Using the ``nontrainable`` modes cannot guarantee the model's reproducibility. Defaults to``nontrainable``.

            - ``deconv``, uses transposed convolution layers.
            - ``nontrainable``, uses non-trainable `linear` interpolation.
            - ``pixelshuffle``, uses :py:class:`monai.networks.blocks.SubpixelUpsample`.

    """

    def __init__(
        self,
        spatial_dims: int = 3,
        init_filters: int = 8,
        in_channels: int = 1,
        out_channels: int = 2,
        dropout_prob: float | None = None,
        act: tuple | str = ("RELU", {"inplace": True}),
        norm: tuple | str = ("GROUP", {"num_groups": 8}),
        norm_name: str = "",
        num_groups: int = 8,
        use_conv_final: bool = True,
        blocks_down: tuple = (1, 2, 2, 4),
        blocks_up: tuple = (1, 1, 1),
        upsample_mode: UpsampleMode | str = UpsampleMode.NONTRAINABLE,
    ):
        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("`spatial_dims` can only be 2 or 3.")

        self.spatial_dims = spatial_dims
        self.init_filters = init_filters
        self.in_channels = in_channels
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up
        self.dropout_prob = dropout_prob
        self.act = act  # input options
        self.act_mod = get_act_layer(act)
        if norm_name:
            if norm_name.lower() != "group":
                raise ValueError(f"Deprecating option 'norm_name={norm_name}', please use 'norm' instead.")
            norm = ("group", {"num_groups": num_groups})
        self.norm = norm
        self.upsample_mode = UpsampleMode(upsample_mode)
        self.use_conv_final = use_conv_final
        self.convInit = get_conv_layer(spatial_dims, in_channels, init_filters)
        self.down_layers = self._make_down_layers()
        self.up_layers, self.up_samples = self._make_up_layers()
        self.conv_final = self._make_final_conv(out_channels)

        if dropout_prob is not None:
            self.dropout = Dropout[Dropout.DROPOUT, spatial_dims](dropout_prob)

    def _make_down_layers(self):
        down_layers = nn.ModuleList()
        blocks_down, spatial_dims, filters, norm = (self.blocks_down, self.spatial_dims, self.init_filters, self.norm)
        for i, item in enumerate(blocks_down):
            layer_in_channels = filters * 2**i
            pre_conv = (
                get_conv_layer(spatial_dims, layer_in_channels // 2, layer_in_channels, stride=2)
                if i > 0
                else nn.Identity()
            )
            down_layer = nn.Sequential(
                pre_conv, *[ResBlock(spatial_dims, layer_in_channels, norm=norm, act=self.act) for _ in range(item)]
            )
            down_layers.append(down_layer)
        return down_layers

    def _make_up_layers(self):
        up_layers, up_samples = nn.ModuleList(), nn.ModuleList()
        upsample_mode, blocks_up, spatial_dims, filters, norm = (
            self.upsample_mode,
            self.blocks_up,
            self.spatial_dims,
            self.init_filters,
            self.norm,
        )
        n_up = len(blocks_up)
        for i in range(n_up):
            sample_in_channels = filters * 2 ** (n_up - i)
            up_layers.append(
                nn.Sequential(
                    *[
                        ResBlock(spatial_dims, sample_in_channels // 2, norm=norm, act=self.act)
                        for _ in range(blocks_up[i])
                    ]
                )
            )
            up_samples.append(
                nn.Sequential(
                    *[
                        get_conv_layer(spatial_dims, sample_in_channels, sample_in_channels // 2, kernel_size=1),
                        get_upsample_layer(spatial_dims, sample_in_channels // 2, upsample_mode=upsample_mode),
                    ]
                )
            )
        return up_layers, up_samples

    def _make_final_conv(self, out_channels: int):
        return nn.Sequential(
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.init_filters),
            self.act_mod,
            get_conv_layer(self.spatial_dims, self.init_filters, out_channels, kernel_size=1, bias=True),
        )
    # Encoder 
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.convInit(x) 
        if self.dropout_prob is not None:
            x = self.dropout(x)

        down_x = []
        # consists of 4 layers
        # Layer 1: Identity - ResBlock. Spatail dimension remaines the same (1, 32, 160, 192, 128)
        # Layer 2: Conv - ResBlock - ResBlock. Halve the spatial dimension (1, 64, 80, 96, 64) )
        # Layer 3: Conv - ResBlock - ResBlock. Halve the spatial dimension (1, 128, 40, 48, 32)
        # Layer 4: Conv - ResBlock - ResBlock - ResBlock - ResBlock. Halve the spatial dimension (1, 256, 20, 24, 16)
        for down in self.down_layers:
            x = down(x)
            down_x.append(x)

        return x, down_x

    def decode(self, x: torch.Tensor, down_x: list[torch.Tensor]) -> torch.Tensor:
        for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
            x = up(x) + down_x[i + 1]
            x = upl(x)

        if self.use_conv_final:
            x = self.conv_final(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, down_x = self.encode(x)
        down_x.reverse() 

        x = self.decode(x, down_x)
        return x


##############################################


# Gradient Reversal Layer Segresnet
class GradReverseFunction(torch.autograd.Function):
    """
    Behaves like identity in the forward pass but reverses gradient in backward pass.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None
    
def grad_reverse(x, alpha = 1.0):
    return GradReverseFunction.apply(x, alpha)


class GRLSegResNet(SegResNet):
    """
    Extends SegResNet with:
    - Domain Classifier head
    - Gradient Reversal in forward pass
    - Returns (seg_loss, domain_loss) so your train loop can sum them
    """

    def __init__(
        self,
        num_domains: int = 3,
        alpha: float = 1,
        *args,
        **kwargs
    ):
        
        """
        Args:
        num domains
        alpha
        all other SegResNet args
        """
        super().__init__(*args, **kwargs)

        self.num_domains = num_domains
        self.alpha = alpha # can override at runtime

        self.domain_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_domains)
        )

        self.seg_loss_fn = DiceLoss(sigmoid = True, to_onehot_y = False)
        self.domain_loss_fn = nn.CrossEntropyLoss()

        norm_layer = norm_nd_class(3)
        self.bottleneck = Bottleneck(
        feats_num_bottleneck = 256,
        norm_layer = norm_layer,
        padding_type = 'reflect',
        resnet_conv_kwargs = {
            "ratio_gin": 0.5,
            "ratio_gout": 0.5,
        },
        inline = True,
        drop_path_rate = 0
    )



    def forward(
        self,
        images: torch.Tensor,
        seg_labels: torch.Tensor | None = None,
        domain_labels: torch.Tensor | None = None,
        alpha: float | None = None
    ):
        """
        Forward pass that yields seg_loss, domain_loss.
        """

        if alpha is None:
            alpha = self.alpha

        x, down_x = self.encode(images)

        # HERE WE USE THE BOTTLENECK AND THEN PASS IT TO BOTH THE DECODER AND THE
        # DOMAIN CLASSIFIER. SEE IF THIS WORKS OR NOT. SO MANY HYPERPARAMETERS THOUGH.

        x = self.bottleneck(x)
    
        feat = x
        down_x.reverse()

        seg_logits = self.decode(x, down_x)

        if seg_labels is None and domain_labels is None:
            return seg_logits

        pooled_feat = F.adaptive_avg_pool3d(feat, (1, 1, 1)).view(feat.size(0), -1)
        reversed_feat = grad_reverse(pooled_feat, alpha = alpha)
        domain_logits = self.domain_head(reversed_feat)

        seg_loss = torch.zeros(1, device = images.device, dtype = torch.float)
        domain_loss = torch.zeros(1, device = images.device, dtype = torch.float)

        if seg_labels is not None:
            seg_loss = self.seg_loss_fn(seg_logits, seg_labels)

        if domain_labels is not None:
            domain_loss = self.domain_loss_fn(domain_logits, domain_labels.long())
        
        return seg_loss, domain_loss