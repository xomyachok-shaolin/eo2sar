# file: hierarchical_marm_generator.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from marm_generator import AxialAttention  # переиспользуем ваш AxialAttention
# Или можно скопировать AxialAttention/PermuteToFrom сюда напрямую.

class SimpleMARM(nn.Module):
    """
    Упрощённый MARMBlock: 2 ветви (обычная свертка, rot90).
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        b1 = self.conv1(x)
        x2 = torch.rot90(x, 1, (2,3))
        b2 = self.conv2(x2)
        b2 = torch.rot90(b2, -1, (2,3))
        out = (b1 + b2) / 2
        return self.relu(out)

class HierarchicalMARMGenerator(nn.Module):
    """
    Генератор с U-Net skip-connection + иерархическим AxialAttention.
    1) down1 -> axial1
    2) down2 -> axial2
    3) down3 -> axial3
    4) bottleneck (marm + axial)
    5) up1(skip from axial3), up2(skip from axial2), up3(skip from axial1)
    """
    def __init__(self, in_ch=3, out_ch=3, base_ch=64):
        super().__init__()

        # --- ENCODER ---
        self.down1 = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 4, 2, 1),
            nn.InstanceNorm2d(base_ch),
            nn.ReLU(True)
        )
        self.axial1 = AxialAttention(dim=base_ch, heads=4, dim_index=1)

        self.down2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch*2, 4, 2, 1),
            nn.InstanceNorm2d(base_ch*2),
            nn.ReLU(True)
        )
        self.axial2 = AxialAttention(dim=base_ch*2, heads=4, dim_index=1)

        self.down3 = nn.Sequential(
            nn.Conv2d(base_ch*2, base_ch*4, 4, 2, 1),
            nn.InstanceNorm2d(base_ch*4),
            nn.ReLU(True)
        )
        self.axial3 = AxialAttention(dim=base_ch*4, heads=8, dim_index=1)

        # --- BOTTLENECK ---
        self.marm = SimpleMARM(base_ch*4)
        self.axial_bottleneck = AxialAttention(dim=base_ch*4, heads=8, dim_index=1)

        # --- DECODER ---
        # up1 -> combine skip from axial3
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_ch*4*2, base_ch*2, 3, 1, 1),
            nn.InstanceNorm2d(base_ch*2),
            nn.ReLU(True)
        )
        self.axial_u1 = AxialAttention(dim=base_ch*2, heads=4, dim_index=1)

        # up2 -> combine skip from axial2
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_ch*2*2, base_ch, 3, 1, 1),
            nn.InstanceNorm2d(base_ch),
            nn.ReLU(True)
        )
        self.axial_u2 = AxialAttention(dim=base_ch, heads=4, dim_index=1)

        # up3 -> combine skip from axial1
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_ch*2, base_ch, 3, 1, 1),
            nn.InstanceNorm2d(base_ch),
            nn.ReLU(True)
        )
        # можно еще axial_u3
        self.final = nn.Conv2d(base_ch, out_ch, 7, 1, 3)

    def forward(self, x):
        # x shape: [N,3,H,W]
        # ENCODER
        d1 = self.down1(x)        # [N, base_ch, H/2, W/2]
        d1_attn = self.axial1(d1) # residual
        d1_out = d1 + d1_attn

        d2 = self.down2(d1_out)            # [N, base_ch*2, H/4, W/4]
        d2_attn = self.axial2(d2)
        d2_out = d2 + d2_attn

        d3 = self.down3(d2_out)            # [N, base_ch*4, H/8, W/8]
        d3_attn = self.axial3(d3)
        d3_out = d3 + d3_attn

        # BOTTLENECK
        b = self.marm(d3_out)              # MARM
        b_attn = self.axial_bottleneck(b)
        b_out = b + b_attn

        # DECODER
        # up1, skip with d3_out
        x_up1_in = torch.cat([b_out, d3_out], dim=1)  # concat channels
        u1 = self.up1(x_up1_in)                       # [N, base_ch*2, H/4, W/4]
        u1_attn = self.axial_u1(u1)
        u1_out = u1 + u1_attn

        # up2, skip with d2_out
        x_up2_in = torch.cat([u1_out, d2_out], dim=1) # [N, base_ch*2 + base_ch*2, ...]
        u2 = self.up2(x_up2_in)                       # => [N, base_ch, H/2, W/2]
        u2_attn = self.axial_u2(u2)
        u2_out = u2 + u2_attn

        # up3, skip with d1_out
        x_up3_in = torch.cat([u2_out, d1_out], dim=1) # => [N, base_ch + base_ch, ...] = [N, base_ch*2, H, W]
        u3 = self.up3(x_up3_in)                       # => [N, base_ch, H, W]

        out = self.final(u3)                          # => [N, out_ch, H, W]
        out = torch.tanh(out)

        return out
