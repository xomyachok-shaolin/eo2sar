import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== КЛАССЫ ДЛЯ АКСИАЛЬНОГО ВНИМАНИЯ =====
# Данный код включает Ваш пример axial-attention: AxialPositionalEmbedding, SelfAttention, AxialAttention и вспом. классы
# + вспомогательная логика для перестановок.

from operator import itemgetter

def exists(val):
    return val is not None

def map_el_ind(arr, ind):
    return list(map(itemgetter(ind), arr))

def sort_and_return_indices(arr):
    indices = [i for i in range(len(arr))]
    arr = zip(arr, indices)
    arr = sorted(arr)
    return map_el_ind(arr, 0), map_el_ind(arr, 1)

def calculate_permutations(num_dimensions, emb_dim):
    total_dimensions = num_dimensions + 2
    emb_dim = emb_dim if emb_dim > 0 else (emb_dim + total_dimensions)
    axial_dims = [ind for ind in range(1, total_dimensions) if ind != emb_dim]
    permutations = []
    for axial_dim in axial_dims:
        last_two_dims = [axial_dim, emb_dim]
        dims_rest = set(range(0, total_dimensions)) - set(last_two_dims)
        permutation = [*dims_rest, *last_two_dims]
        permutations.append(permutation)
    return permutations

class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class Sequential(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks

    def forward(self, x):
        for f, g in self.blocks:
            x = x + f(x)
            x = x + g(x)
        return x

# Модуль для перестановки, чтобы применить AxialAttention по одной из осей (H или W) 
class PermuteToFrom(nn.Module):
    def __init__(self, permutation, fn):
        super().__init__()
        self.fn = fn
        _, inv_permutation = sort_and_return_indices(permutation)
        self.permutation = permutation
        self.inv_permutation = inv_permutation

    def forward(self, x, **kwargs):
        axial = x.permute(*self.permutation).contiguous()
        shape = axial.shape
        *_, t, d = shape
        axial = axial.reshape(-1, t, d)
        axial = self.fn(axial, **kwargs)
        axial = axial.reshape(*shape)
        axial = axial.permute(*self.inv_permutation).contiguous()
        return axial

class SelfAttention(nn.Module):
    def __init__(self, dim, heads, dim_heads=None):
        super().__init__()
        if dim_heads is None:
            self.dim_heads = dim // heads
        else:
            self.dim_heads = dim_heads
        dim_hidden = self.dim_heads * heads
        self.heads = heads
        self.to_q = nn.Linear(dim, dim_hidden, bias=False)
        self.to_kv = nn.Linear(dim, 2 * dim_hidden, bias=False)
        self.to_out = nn.Linear(dim_hidden, dim)

    def forward(self, x, kv=None):
        if kv is None:
            kv = x
        q = self.to_q(x)
        kv = self.to_kv(kv)
        k, v = kv.chunk(2, dim=-1)
        b, t, d = q.shape
        h = self.heads
        e = self.dim_heads

        def merge_heads(tensor):
            return tensor.reshape(b, -1, h, e).transpose(1, 2).reshape(b * h, -1, e)

        q = merge_heads(q)
        k = merge_heads(k)
        v = merge_heads(v)

        dots = torch.einsum('bie,bje->bij', q, k) * (e ** -0.5)
        dots = dots.softmax(dim=-1)
        out = torch.einsum('bij,bje->bie', dots, v)

        out = out.reshape(b, h, -1, e).transpose(1, 2).reshape(b, -1, d)
        out = self.to_out(out)
        return out

class AxialAttention(nn.Module):
    def __init__(self, dim, num_dimensions=2, heads=8, dim_heads=None, dim_index=-1, sum_axial_out=True):
        super().__init__()
        assert (dim % heads) == 0, 'hidden dimension must be divisible by number of heads'
        self.dim = dim
        self.total_dimensions = num_dimensions + 2
        if dim_index < 0:
            self.dim_index = dim_index + self.total_dimensions
        else:
            self.dim_index = dim_index
        attentions = []
        permutations = calculate_permutations(num_dimensions, self.dim_index)
        for permutation in permutations:
            attentions.append(PermuteToFrom(permutation, SelfAttention(dim, heads, dim_heads)))
        self.axial_attentions = nn.ModuleList(attentions)
        self.sum_axial_out = sum_axial_out

    def forward(self, x):
        assert len(x.shape) == self.total_dimensions, f'Input tensor must have {self.total_dimensions} dims'
        assert x.shape[self.dim_index] == self.dim, f'Input tensor does not match the required dimension {self.dim}'
        if self.sum_axial_out:
            outputs = [axial_attn(x) for axial_attn in self.axial_attentions]
            return sum(outputs)
        out = x
        for axial_attn in self.axial_attentions:
            out = axial_attn(out)
        return out

# ===== MARM BLOCK =====

class MARMBlock(nn.Module):
    """
    Пример блока MARM (Multiscale Axial Residual Module).
    Допустим, мы делаем 4 ветви:
    1) Обычная свёртка
    2) Перестановка axes (C <-> H)
    3) Поворот на 90 градусов
    4) Перестановка (C <-> W)
    И после этого усредняем их результаты.
    """
    def __init__(self, channels):
        super().__init__()
        self.branch1_conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.branch2_conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.branch3_conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # branch1: обычная свертка
        b1 = self.branch1_conv(x)
   
        # branch2: поворот на 90 градусов
        x2 = torch.rot90(x, 1, dims=(2,3))
        b2 = self.branch2_conv(x2)
        b2 = torch.rot90(b2, -1, dims=(2,3))

        # branch3: flip по H, к примеру
        x3 = torch.flip(x, dims=(2,))
        b3 = self.branch3_conv(x3)
        b3 = torch.flip(b3, dims=(2,))

        # branch4: просто та же карта (identity) или другой вариант
        b4 = x  # или ещё одна conv

        out = (b1 + b2 + b3 + b4) / 4.0
        out = self.relu(out)
        return out


# ===== Сборка: MARM + Axial Attention + Encoder/Decoder =====

class MARMGenerator(nn.Module):
    """
    Генератор, состоящий из:
    - Encoder (Conv -> Downsample)
    - MARMBlock (multiscale axial residual)
    - AxialAttention
    - Decoder (Upsample -> Conv)
    Выход: изображение SAR
    """
    def __init__(self, in_ch=3, out_ch=3, base_ch=64):
        super().__init__()

        # === ENCODER ===
        self.down1 = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(base_ch),
            nn.ReLU(True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch*2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(base_ch*2),
            nn.ReLU(True)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(base_ch*2, base_ch*4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(base_ch*4),
            nn.ReLU(True)
        )

        # === MARM BLOCK ===
        self.marm = MARMBlock(base_ch*4)

        # === AXIAL ATTENTION ===
        # Применим AxialAttention (размерность = base_ch*4)
        self.axial = AxialAttention(
            dim=base_ch*4,
            heads=8,
            dim_heads=None,
            dim_index=1,         # <--- этот параметр
            sum_axial_out=True
        )

        # === DECODER ===
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_ch*4, base_ch*2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(base_ch*2),
            nn.ReLU(True)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_ch*2, base_ch, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(base_ch),
            nn.ReLU(True)
        )
        self.final = nn.Conv2d(base_ch, out_ch, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        # ENCODER
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)

        # MARM
        x = self.marm(x)

        # AXIAL
        x_attn = self.axial(x)
        # можно слить (residual) или просто заменить:
        x = x + x_attn  # небольшая остаточная связь

        # DECODER
        x = self.up1(x)
        x = self.up2(x)
        x = self.final(x)
        x = torch.tanh(x)
        return x
