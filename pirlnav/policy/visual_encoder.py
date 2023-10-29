
import numpy as np
import torch
from habitat_baselines.rl.ddppo.policy.running_mean_and_var import RunningMeanAndVar
from torch import nn as nn
from torch.nn import functional as F

from pirlnav.policy.models import resnet_gn as resnet


class VisualEncoder(nn.Module):
    def __init__(
        self,
        image_size: int,
        backbone: str,
        input_channels: int = 3,
        resnet_baseplanes: int = 32,
        resnet_ngroups: int = 32,
        normalize_visual_inputs: bool = True,
        avgpooled_image: bool = False,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.avgpooled_image = avgpooled_image

        if normalize_visual_inputs:
            self.running_mean_and_var: nn.Module = RunningMeanAndVar(input_channels)
        else:
            self.running_mean_and_var = nn.Sequential()

        if "resnet" in backbone:
            make_backbone = getattr(resnet, backbone)
            self.backbone = make_backbone(
                input_channels, resnet_baseplanes, resnet_ngroups
            )

            spatial_size = image_size
            if self.avgpooled_image:
                spatial_size = image_size // 2

            final_spatial = int(spatial_size * self.backbone.final_spatial_compress)
            after_compression_flat_size = 2048
            num_compression_channels = int(
                round(after_compression_flat_size / (final_spatial**2))
            )
            self.compression = nn.Sequential(
                nn.Conv2d(
                    self.backbone.final_channels,
                    num_compression_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(1, num_compression_channels),
                nn.ReLU(True),
            )

            output_shape = (
                num_compression_channels,
                final_spatial,
                final_spatial,
            )
            self.output_size = np.prod(output_shape)
        else:
            raise ValueError("unknown backbone {}".format(backbone))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        if self.avgpooled_image:  # For compatibility with the habitat_baselines implementation
            x = F.avg_pool2d(x, 2)
        x = self.running_mean_and_var(x)
        x = self.backbone(x)
        x = self.compression(x)
        return x
