"""LiteOT-Fuse: MFNet with spatially constrained local optimal transport.

The design deliberately keeps MFNet's lightweight dual encoders and decoder.
At 1/16 scale, thermal tokens are aligned to RGB tokens inside non-overlapping
local windows with a differentiable Sinkhorn solver.  A reliability gate then
balances RGB and aligned thermal features using learned evidence plus an
explicit illumination prior.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .MFNet import ConvBnLeakyRelu2d, MiniInception


class SpatiallyConstrainedLocalOT(nn.Module):
    """Align source features to reference locations with windowed Sinkhorn OT."""

    def __init__(
        self,
        reference_channels: int,
        source_channels: int,
        embedding_dim: int = 32,
        window_size: int = 4,
        sinkhorn_iterations: int = 3,
        epsilon: float = 0.10,
        spatial_weight: float = 0.20,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.sinkhorn_iterations = sinkhorn_iterations
        self.epsilon = epsilon
        self.spatial_weight = spatial_weight

        self.reference_projection = nn.Conv2d(reference_channels, embedding_dim, 1, bias=False)
        self.source_projection = nn.Conv2d(source_channels, embedding_dim, 1, bias=False)
        self.source_output = nn.Conv2d(embedding_dim, source_channels, 1, bias=False)
        # A small residual keeps MFNet initialisation stable while allowing OT
        # projections to receive gradients from the first optimisation step.
        self.residual_scale = nn.Parameter(torch.tensor([0.01]))

        coordinates = torch.stack(
            torch.meshgrid(
                torch.linspace(-1.0, 1.0, window_size),
                torch.linspace(-1.0, 1.0, window_size),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 2)
        spatial_cost = torch.cdist(coordinates, coordinates, p=2).square() / 8.0
        self.register_buffer("spatial_cost", spatial_cost, persistent=False)

    def _to_windows(self, features: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        batch, channels, height, width = features.shape
        window = self.window_size
        padded_height = math.ceil(height / window) * window
        padded_width = math.ceil(width / window) * window
        features = F.pad(features, (0, padded_width - width, 0, padded_height - height))
        windows = features.unfold(2, window, window).unfold(3, window, window)
        # B,C,nH,nW,w,w -> B*nH*nW,w*w,C
        windows = windows.permute(0, 2, 3, 4, 5, 1).contiguous()
        windows = windows.reshape(-1, window * window, channels)
        return windows, (height, width, padded_height, padded_width)

    def _from_windows(
        self, windows: torch.Tensor, shape: tuple[int, int, int, int], batch: int
    ) -> torch.Tensor:
        height, width, padded_height, padded_width = shape
        window = self.window_size
        channels = windows.shape[-1]
        rows, columns = padded_height // window, padded_width // window
        features = windows.reshape(batch, rows, columns, window, window, channels)
        features = features.permute(0, 5, 1, 3, 2, 4).contiguous()
        features = features.reshape(batch, channels, padded_height, padded_width)
        return features[:, :, :height, :width]

    def _sinkhorn(self, cost: torch.Tensor) -> torch.Tensor:
        # Uniform marginals.  Scaling in log space avoids underflow in mixed
        # precision while keeping the tiny 16x16 transport problem inexpensive.
        token_count = cost.shape[-1]
        log_kernel = -cost.float() / self.epsilon
        log_mass = -math.log(token_count)
        log_u = torch.zeros_like(log_kernel[..., 0])
        log_v = torch.zeros_like(log_kernel[..., 0])
        for _ in range(self.sinkhorn_iterations):
            log_u = log_mass - torch.logsumexp(log_kernel + log_v.unsqueeze(-2), dim=-1)
            log_v = log_mass - torch.logsumexp(log_kernel + log_u.unsqueeze(-1), dim=-2)
        return torch.exp(log_kernel + log_u.unsqueeze(-1) + log_v.unsqueeze(-2))

    def forward(self, reference: torch.Tensor, source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        reference_embedding = self.reference_projection(reference)
        source_embedding = self.source_projection(source)
        reference_windows, shape = self._to_windows(reference_embedding)
        source_windows, _ = self._to_windows(source_embedding)

        reference_normalised = F.normalize(reference_windows.float(), dim=-1)
        source_normalised = F.normalize(source_windows.float(), dim=-1)
        appearance_cost = 1.0 - torch.bmm(reference_normalised, source_normalised.transpose(1, 2))
        cost = appearance_cost + self.spatial_weight * self.spatial_cost
        transport = self._sinkhorn(cost)

        # Each row sums to 1/N; multiply by N for a barycentric source feature.
        aligned_windows = torch.bmm(transport, source_windows.float()) * transport.shape[-1]
        aligned_embedding = self._from_windows(aligned_windows, shape, reference.shape[0])
        aligned_embedding = aligned_embedding.to(source.dtype)
        aligned_source = source + torch.tanh(self.residual_scale) * self.source_output(aligned_embedding)
        mean_transport_entropy = -(transport * transport.clamp_min(1e-12).log()).sum((-1, -2)).mean()
        return aligned_source, mean_transport_entropy


class ReliabilityGatedFusion(nn.Module):
    """Modality gate with a learned local score and an RGB illumination prior."""

    def __init__(self, rgb_channels: int, thermal_channels: int, hidden_channels: int = 24) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(rgb_channels + thermal_channels + 2, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, 1, 1),
        )
        # Start close to equal weighting; illumination supplies a mild prior.
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.illumination_strength = nn.Parameter(torch.tensor(0.05))

    def forward(
        self, rgb_feature: torch.Tensor, thermal_feature: torch.Tensor, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rgb = image[:, :3]
        thermal = image[:, 3:4]
        illumination = rgb.mean(1, keepdim=True)
        illumination = F.adaptive_avg_pool2d(illumination, rgb_feature.shape[-2:])
        thermal_local_mean = F.avg_pool2d(thermal, 5, stride=1, padding=2)
        thermal_contrast = (thermal - thermal_local_mean).abs()
        thermal_contrast = F.adaptive_avg_pool2d(thermal_contrast, rgb_feature.shape[-2:])

        evidence = torch.cat((rgb_feature, thermal_feature, illumination, thermal_contrast), dim=1)
        illumination_prior = (illumination - 0.35) * 4.0
        rgb_weight = torch.sigmoid(self.gate(evidence) + self.illumination_strength * illumination_prior)
        # Factor two preserves the expected activation scale around a 0.5 gate.
        fused = torch.cat((2.0 * rgb_weight * rgb_feature, 2.0 * (1.0 - rgb_weight) * thermal_feature), dim=1)
        return fused, rgb_weight


class LiteOTFuse(nn.Module):
    def __init__(
        self,
        n_class: int = 9,
        use_ot: bool = True,
        use_gate: bool = True,
        window_size: int = 4,
        sinkhorn_iterations: int = 3,
        ot_embedding_dim: int = 32,
        spatial_weight: float = 0.20,
    ) -> None:
        super().__init__()
        self.use_ot = use_ot
        self.use_gate = use_gate
        rgb_ch = [16, 48, 48, 96, 96]
        inf_ch = [16, 16, 16, 36, 36]

        self.conv1_rgb = ConvBnLeakyRelu2d(3, rgb_ch[0])
        self.conv2_1_rgb = ConvBnLeakyRelu2d(rgb_ch[0], rgb_ch[1])
        self.conv2_2_rgb = ConvBnLeakyRelu2d(rgb_ch[1], rgb_ch[1])
        self.conv3_1_rgb = ConvBnLeakyRelu2d(rgb_ch[1], rgb_ch[2])
        self.conv3_2_rgb = ConvBnLeakyRelu2d(rgb_ch[2], rgb_ch[2])
        self.conv4_rgb = MiniInception(rgb_ch[2], rgb_ch[3])
        self.conv5_rgb = MiniInception(rgb_ch[3], rgb_ch[4])

        self.conv1_inf = ConvBnLeakyRelu2d(1, inf_ch[0])
        self.conv2_1_inf = ConvBnLeakyRelu2d(inf_ch[0], inf_ch[1])
        self.conv2_2_inf = ConvBnLeakyRelu2d(inf_ch[1], inf_ch[1])
        self.conv3_1_inf = ConvBnLeakyRelu2d(inf_ch[1], inf_ch[2])
        self.conv3_2_inf = ConvBnLeakyRelu2d(inf_ch[2], inf_ch[2])
        self.conv4_inf = MiniInception(inf_ch[2], inf_ch[3])
        self.conv5_inf = MiniInception(inf_ch[3], inf_ch[4])

        self.local_ot = SpatiallyConstrainedLocalOT(
            rgb_ch[4], inf_ch[4], ot_embedding_dim, window_size,
            sinkhorn_iterations, spatial_weight=spatial_weight,
        )
        self.reliability_fusion = ReliabilityGatedFusion(rgb_ch[4], inf_ch[4])

        self.decode4 = ConvBnLeakyRelu2d(rgb_ch[3] + inf_ch[3], rgb_ch[2] + inf_ch[2])
        self.decode3 = ConvBnLeakyRelu2d(rgb_ch[2] + inf_ch[2], rgb_ch[1] + inf_ch[1])
        self.decode2 = ConvBnLeakyRelu2d(rgb_ch[1] + inf_ch[1], rgb_ch[0] + inf_ch[0])
        self.decode1 = ConvBnLeakyRelu2d(rgb_ch[0] + inf_ch[0], n_class)

        self.last_auxiliary: dict[str, torch.Tensor] = {}

    def load_mfnet_backbone(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load every shape-compatible tensor from the released MFNet."""
        own_state = self.state_dict()
        compatible = {key: value for key, value in state_dict.items() if key in own_state and own_state[key].shape == value.shape}
        self.load_state_dict(compatible, strict=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x_rgb, x_inf = image[:, :3], image[:, 3:]

        x_rgb = F.max_pool2d(self.conv1_rgb(x_rgb), 2, 2)
        x_rgb = self.conv2_1_rgb(x_rgb)
        x_rgb_p2 = self.conv2_2_rgb(x_rgb)
        x_rgb = F.max_pool2d(x_rgb_p2, 2, 2)
        x_rgb = self.conv3_1_rgb(x_rgb)
        x_rgb_p3 = self.conv3_2_rgb(x_rgb)
        x_rgb = F.max_pool2d(x_rgb_p3, 2, 2)
        x_rgb_p4 = self.conv4_rgb(x_rgb)
        x_rgb = self.conv5_rgb(F.max_pool2d(x_rgb_p4, 2, 2))

        x_inf = F.max_pool2d(self.conv1_inf(x_inf), 2, 2)
        x_inf = self.conv2_1_inf(x_inf)
        x_inf_p2 = self.conv2_2_inf(x_inf)
        x_inf = F.max_pool2d(x_inf_p2, 2, 2)
        x_inf = self.conv3_1_inf(x_inf)
        x_inf_p3 = self.conv3_2_inf(x_inf)
        x_inf = F.max_pool2d(x_inf_p3, 2, 2)
        x_inf_p4 = self.conv4_inf(x_inf)
        x_inf = self.conv5_inf(F.max_pool2d(x_inf_p4, 2, 2))

        if self.use_ot:
            x_inf, transport_entropy = self.local_ot(x_rgb, x_inf)
        else:
            transport_entropy = x_rgb.new_zeros(())
        if self.use_gate:
            x, rgb_weight = self.reliability_fusion(x_rgb, x_inf, image)
        else:
            x = torch.cat((x_rgb, x_inf), dim=1)
            rgb_weight = x_rgb.new_full((x_rgb.shape[0], 1, *x_rgb.shape[-2:]), 0.5)
        self.last_auxiliary = {
            "transport_entropy": transport_entropy.detach(),
            "mean_rgb_weight": rgb_weight.detach().mean(),
        }

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.decode4(x + torch.cat((x_rgb_p4, x_inf_p4), dim=1))
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.decode3(x + torch.cat((x_rgb_p3, x_inf_p3), dim=1))
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.decode2(x + torch.cat((x_rgb_p2, x_inf_p2), dim=1))
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.decode1(x)
