# sibi_islr/models/dsta_slr.py and dsta_slr_network.py are adapted from DSTA-SLR:
#   https://github.com/hulianyuyy/DSTA-SLR, commit e7e5ee225511488039a4f68ab3a146cf8f01312d
#   Hu et al., "Dynamic Spatial-Temporal Aggregation for Skeleton-Aware Sign Language Recognition",
#   LREC-COLING 2024 (arXiv:2403.12519).
# DSTA-SLR is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0),
# https://creativecommons.org/licenses/by-nc/4.0/. This modified version is distributed under the same license, for
# non-commercial use only. The MIT license of the rest of this repository does not apply to these two files.
#
# Modifications by the SIBI-ISLR Benchmark authors (2026):
# - copied from model/fstgan.py, model/attention.py, model/dropSke.py, model/dropT.py, graph/sign_27.py, and
#   graph/tools.py into one file (now split into dsta_slr.py and
#   dsta_slr_network.py); only the joint stream is kept;
# - in_channels = 2 (xy input; the HRNet score channel is dropped) and the number of classes is an argument;
# - parameters are created on the module's device (the original forces device="cuda");
# - classes not used by Model (unit_tan, global_tan, Edge_conv, MHSA, ...) are not copied;
# - einops.rearrange is replaced by the same permute/reshape (tok, untok), and timm's trunc_normal_ by
#   torch.nn.init.trunc_normal_ (same algorithm);
# - graph/sign_27.py and graph/tools.py are reduced to EDGES and graph_a() (labeling mode "spatial" only);
# - DropBlock_Ske keeps only the branch used for 27 points, without its warning; console prints are removed;
# - new input code (prepare, make_batch) that follows feeders/feeder.py; comments rewritten.
"""DSTA-SLR network (graph, layers and Model), adapted from the official repository (moved from
dsta_slr.py; see the license header above)."""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- graph/sign_27.py + graph/tools.py ----

EDGES = [(i - 5, j - 5) for i, j in [(5, 6), (5, 7), (6, 8), (8, 10), (7, 9), (9, 11), (12, 13), (12, 14), (12, 16),
                                       (12, 18), (12, 20), (14, 15), (16, 17), (18, 19), (20, 21), (22, 23), (22, 24),
                                       (22, 26), (22, 28), (22, 30), (24, 25), (26, 27), (28, 29), (30, 31), (10, 12),
                                       (11, 22)]]


def graph_a(n=27):
    """labeling_mode 'spatial': (3, n, n) = identity, inward, outward; inward/outward divided by their column sums."""
    def mat(link):
        A = np.zeros((n, n))
        for i, j in link:
            A[j, i] = 1
        return A

    def norm(A):
        d = A.sum(0)
        return A / np.where(d > 0, d, 1)
    return np.stack((mat([(i, i) for i in range(n)]), norm(mat(EDGES)), norm(mat([(j, i) for i, j in EDGES]))))


# ---- model/attention.py ----

def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode="fan_out")
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def tok(x):
    """rearrange 'b c t v -> (b t) v c'."""
    n, c, t, v = x.shape
    return x.permute(0, 2, 3, 1).reshape(n * t, v, c)


def untok(x, n):
    """rearrange '(b t) v c -> b c t v'."""
    bt, v, c = x.shape
    return x.reshape(n, bt // n, v, c).permute(0, 3, 1, 2)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, padding=padding, groups=dim_in, stride=stride, bias=bias),
            nn.Conv2d(dim_out, dim_out, kernel_size=1, bias=bias))

    def forward(self, x):
        return self.net(x)


# ---- model/dropSke.py, model/dropT.py ----

class DropBlock_Ske(nn.Module):
    """DropGraph: a point is dropped together with its neighbours in A."""
    def __init__(self, num_point, block_size=7):
        super().__init__()
        self.keep_prob = 0.0
        self.block_size = block_size
        self.num_point = num_point

    def forward(self, input, keep_prob, A):
        self.keep_prob = keep_prob
        if not self.training or self.keep_prob == 1:
            return input
        n, c, t, v = input.size()
        input_abs = torch.mean(torch.mean(torch.abs(input), dim=2), dim=1).detach()
        input_abs = input_abs / torch.sum(input_abs) * input_abs.numel()
        gamma = (1.0 - self.keep_prob) / (1 + 1.92)        # the repo's "else" branch for 27 points, without warnings.warn
        M_seed = torch.bernoulli(torch.clamp(input_abs * gamma, max=1.0)).to(device=input.device, dtype=input.dtype)
        M = torch.matmul(M_seed, A)
        M[M > 0.001] = 1.0
        M[M < 0.5] = 0.0
        mask = (1 - M).view(n, 1, 1, self.num_point)
        return input * mask * mask.numel() / mask.sum()


class DropBlockT_1d(nn.Module):
    def __init__(self, block_size=7):
        super().__init__()
        self.keep_prob = 0.0
        self.block_size = block_size

    def forward(self, input, keep_prob):
        self.keep_prob = keep_prob
        if not self.training or self.keep_prob == 1:
            return input
        n, c, t, v = input.size()
        input_abs = torch.mean(torch.mean(torch.abs(input), dim=3), dim=1).detach()
        input_abs = (input_abs / torch.sum(input_abs) * input_abs.numel()).view(n, 1, t)
        gamma = (1.0 - self.keep_prob) / self.block_size
        input1 = input.permute(0, 1, 3, 2).contiguous().view(n, c * v, t)
        M = torch.bernoulli(torch.clamp(input_abs * gamma, max=1.0)).repeat(1, c * v, 1)
        Msum = F.max_pool1d(M, kernel_size=[self.block_size], stride=1, padding=self.block_size // 2)
        mask = (1 - Msum).to(device=input.device, dtype=input.dtype)
        return (input1 * mask * mask.numel() / mask.sum()).view(n, c, v, t).permute(0, 1, 3, 2)


# ---- model/fstgan.py ----

class Edge_feature_conv(nn.Module):
    """Super nodes: E = 6 virtual features exchanged with every point."""
    def __init__(self, in_channels, num_joint=27, edge_num=6):
        super().__init__()
        self.num_joint = num_joint
        self.edge_num = edge_num
        self.edge_features = nn.Parameter(torch.randn(in_channels, edge_num), requires_grad=True)
        nn.init.trunc_normal_(self.edge_features, std=0.02)
        self.norm1 = nn.LayerNorm(in_channels)
        self.transform = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.alphas = nn.Parameter(torch.ones(1, 1, edge_num, 1), requires_grad=True)

    def forward(self, x):
        n, c, t, v = x.shape
        x = self.transform(untok(self.norm1(tok(x)), n))
        attention = torch.tanh(torch.einsum("nctv,ce->ntev", x, self.edge_features)) / c * self.alphas
        return torch.einsum("ce,ntev->nctv", self.edge_features, attention)


class unit_san(nn.Module):
    """Graph correlation (dynamic A per sample, 8 subsets) + super nodes + a learnable full graph per channel."""
    def __init__(self, in_channels, out_channels, A, groups, num_point, num_subset=3, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_point = num_point
        self.groups = groups
        num_subset = 8                                     # the repo overrides the num_subset argument
        self.num_subset = num_subset
        self.inter_channels = out_channels // num_subset
        self.attention0s = nn.Parameter(
            torch.ones(1, num_subset, self.inter_channels, num_point, num_point) / num_point, requires_grad=True)
        if in_channels != out_channels:
            self.down = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
        else:
            self.down = lambda x: x
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        for m in self.modules():                           # repo order: only the modules created above
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        self.in_nets = nn.Conv2d(out_channels, 2 * num_subset * self.inter_channels, 1, bias=True)
        self.alphas = nn.Parameter(torch.ones(1, num_subset, 1, 1), requires_grad=True)
        self.bn0 = nn.BatchNorm2d(self.inter_channels * num_subset)
        self.tan = nn.Tanh()
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.mlp = Mlp(in_features=out_channels, hidden_features=int(out_channels * 4), act_layer=nn.GELU, drop=0.0)
        self.Linear_weight = nn.Parameter(torch.zeros(in_channels, self.inter_channels * num_subset), requires_grad=True)
        nn.init.normal_(self.Linear_weight, 0, math.sqrt(0.5 / (self.inter_channels * num_subset)))
        self.Linear_bias = nn.Parameter(torch.zeros(1, self.inter_channels * num_subset, 1, 1), requires_grad=True)
        nn.init.constant_(self.Linear_bias, 1e-6)
        self.Edge_conv = Edge_feature_conv(out_channels)
        self.edge_weight = nn.Parameter(torch.zeros(1))

    def forward(self, x0):
        x = torch.einsum("nctw,cd->ndtw", x0, self.Linear_weight).contiguous()
        x = x + self.Linear_bias
        x = self.bn0(x)
        n, c, t, v = x.size()
        x_resi = x
        edge_features = self.Edge_conv(x) * self.edge_weight
        x = untok(self.norm1(tok(x)), n)
        q, k = torch.chunk(self.in_nets(x).view(n, 2 * self.num_subset, self.inter_channels, t, v), 2, dim=1)
        attention = self.tan(torch.einsum("nkctu,nkctv->nkuv", q, k) / (self.inter_channels * t)) * self.alphas
        x = x.view(n, self.num_subset, -1, t, v)
        x = torch.einsum("nkctv,nkvw->nkctw", x, attention).view(n, -1, t, v) + x_resi
        x_resi = x
        x = x_resi + untok(self.mlp(self.norm2(tok(x))), n) + edge_features
        n, kc, t, v = x.size()
        x = x.view(n, self.num_subset, kc // self.num_subset, t, v)
        x = torch.einsum("nkctv,nkcvw->nkctw", x, self.attention0s.repeat(n, 1, 1, 1, 1)).view(n, -1, t, v)
        x = self.bn(x)
        x += self.down(x0)
        return self.relu(x)


class unit_tcn_skip(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1):
        super().__init__()
        self.pool = DepthWiseConv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=0, stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        bn_init(self.bn, 1)

    def forward(self, x):
        return self.bn(self.pool(x))


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, num_point=25, block_size=41):
        super().__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()                              # created but unused, as in the repo
        conv_init(self.conv)
        bn_init(self.bn, 1)
        self.dropS = DropBlock_Ske(num_point=num_point)
        self.dropT = DropBlockT_1d(block_size=block_size)

    def forward(self, x, keep_prob, A):
        return self.dropT(self.dropS(self.bn(self.conv(x)), keep_prob, A), keep_prob)


class MSTCN(nn.Module):
    """PTCN: temporal convolution branches with kernels 5 and 7, concatenated over channels."""
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 7), stride=1, num_point=27, block_size=41):
        super().__init__()
        self.num_branches = len(kernel_sizes)
        assert out_channels % self.num_branches == 0
        branch_channels = out_channels // self.num_branches
        self.branches = nn.ModuleList([unit_tcn(in_channels, branch_channels, kernel_size=k, stride=stride,
                                                num_point=num_point, block_size=block_size) for k in kernel_sizes])

    def forward(self, x, keep_prob, A):
        return torch.cat([b(x, keep_prob, A) for b in self.branches], dim=1)


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, A, groups, num_point, block_size, stride=1, residual=True,
                 use_grpe=True, is_first=False, window_size=120, **kwargs):
        super().__init__()
        tmp_c = out_channels if is_first else in_channels
        self.san = unit_san(in_channels, tmp_c, A, groups, num_point, is_first=is_first, use_grpe=use_grpe, **kwargs)
        self.tcn = MSTCN(tmp_c, out_channels, stride=1, num_point=num_point)
        self.tcn2 = MSTCN(out_channels, out_channels, stride=1, num_point=num_point)
        self.tcn3 = MSTCN(out_channels, out_channels, stride=stride, num_point=num_point)
        self.relu = nn.ReLU()
        self.attention = True                              # STC module of SAM-SLR-v2 (the repo prints "Attention Enabled!")
        self.sigmoid = nn.Sigmoid()
        self.conv_ta = nn.Conv1d(tmp_c, 1, 9, padding=4)
        nn.init.constant_(self.conv_ta.weight, 0)
        nn.init.constant_(self.conv_ta.bias, 0)
        ker_jpt = num_point - 1 if not num_point % 2 else num_point
        self.conv_sa = nn.Conv1d(tmp_c, 1, ker_jpt, padding=(ker_jpt - 1) // 2)
        nn.init.xavier_normal_(self.conv_sa.weight)
        nn.init.constant_(self.conv_sa.bias, 0)
        rr = 2
        self.fc1c = nn.Linear(tmp_c, tmp_c // rr)
        self.fc2c = nn.Linear(tmp_c // rr, tmp_c)
        nn.init.kaiming_normal_(self.fc1c.weight)
        nn.init.constant_(self.fc1c.bias, 0)
        nn.init.constant_(self.fc2c.weight, 0)
        nn.init.constant_(self.fc2c.bias, 0)
        self.A = nn.Parameter(torch.tensor(np.sum(np.reshape(A.astype(np.float32), [3, num_point, num_point]), axis=0),
                                           dtype=torch.float32), requires_grad=False)
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn_skip(in_channels, out_channels, kernel_size=1, stride=stride)
        self.dropSke = DropBlock_Ske(num_point=num_point)
        self.dropT_skip = DropBlockT_1d(block_size=block_size)

    def forward(self, x, keep_prob):
        y = self.san(x)
        se1 = self.sigmoid(self.conv_sa(y.mean(-2)))       # spatial attention
        y = y * se1.unsqueeze(-2) + y
        se1 = self.sigmoid(self.conv_ta(y.mean(-1)))       # temporal attention
        y = y * se1.unsqueeze(-1) + y
        se2 = self.sigmoid(self.fc2c(self.relu(self.fc1c(y.mean(-1).mean(-1)))))   # channel attention
        y = y * se2.unsqueeze(-1).unsqueeze(-1) + y
        y = self.tcn3(self.tcn2(self.tcn(y, keep_prob, self.A), keep_prob, self.A), keep_prob, self.A)
        x_skip = self.dropT_skip(self.dropSke(self.residual(x), keep_prob, self.A), keep_prob)
        return self.relu(y + x_skip)


class Model(nn.Module):
    def __init__(self, num_class, in_channels=2, num_point=27, num_person=1, groups=16, block_size=41, inner_dim=64,
                 depth=4, drop_layers=2, window_size=120):
        super().__init__()
        A = graph_a(num_point)
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.drop_layers = depth - drop_layers             # DropGraph only in blocks 2 and 3
        e = [2 ** i for i in range(depth)]
        self.layers = nn.ModuleList([
            Block(in_channels, inner_dim, A, groups, num_point, block_size, residual=False, window_size=window_size,
                  i=i, is_first=True) if i == 0 else
            Block(inner_dim * e[i - 1], inner_dim * e[i], A, groups, num_point, block_size, stride=e[i] // e[i - 1],
                  residual=True, window_size=window_size // e[i], i=i)
            for i in range(depth)])
        self.fc = nn.Linear(inner_dim * e[-1], num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2.0 / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x, keep_prob=0.9):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        for u, blk in enumerate(self.layers):
            x = blk(x, 1.0 if u < self.drop_layers else keep_prob)
        x = x.reshape(N, M, x.size(1), -1).mean(3).mean(1)
        return self.fc(x)
