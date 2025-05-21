import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimAM(nn.Module):
    """
    SimAM: Simple, Parameter-Free Attention Module
    https://arxiv.org/abs/2012.13492
    """
    def __init__(self, lambda_val=1e-4):
        super().__init__()
        self.lambda_val = lambda_val

    def forward(self, x):
        x_sq = x * x
        x_sum = x_sq.sum(dim=[2,3], keepdim=True)
        # E = 1 - x^2/(x^2 + S + eps)
        E = 1 - x_sq / (x_sq + x_sum + self.lambda_val)
        return x * torch.sigmoid(E)

class ECA(nn.Module):
    """
    ECA: Efficient Channel Attention
    https://arxiv.org/abs/1910.03151
    """
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log(channels,2)+b)/gamma))
        k_size = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)

    def forward(self, x):
        # x: [N,C,H,W]
        y = self.avg_pool(x)  # [N,C,1,1]
        y = y.squeeze(-1).transpose(-1,-2)  # [N,C,1] -> [N,1,C] (?)
        y = self.conv(y)  # [N,1,C]
        y = y.transpose(-1,-2).unsqueeze(-1)  # обратно -> [N,C,1,1]
        y = torch.sigmoid(y)
        return x * y

class AttnDiscriminator(nn.Module):
    """
    Discriminator with:
    - PatchGAN idea (strided convs)
    - SimAM module for spatial attention
    - ECA module for channel attention
    """
    def __init__(self, in_ch=3, base_ch=64):
        super().__init__()

        # block1
        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2,True),
            SimAM(),
            ECA(base_ch)
        )
        # block2
        self.block2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch*2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2,True),
            SimAM(),
            ECA(base_ch*2)
        )
        # block3
        self.block3 = nn.Sequential(
            nn.Conv2d(base_ch*2, base_ch*4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2,True),
            SimAM(),
            ECA(base_ch*4)
        )
        # final
        self.final = nn.Conv2d(base_ch*4, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        """
        Выдаём карту (N,1,H/8,W/8), которая говорит "реальное/подделка"
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.final(x)
        return x
