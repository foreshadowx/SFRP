import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

# 定义函数index_points，用于根据索引从点数据中采样
# 输入：
#     points：输入点数据，[B, N, C]
#     idx：采样索引数据，[B, S]
# 输出：
#     new_points：采样后的点数据，[B, S, C]
def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


    
class FRPS(nn.Module):
    def __init__(self, npts_ds,
                 in_channel: int,
                 query_num: int):
        super(FRPS, self).__init__()
        self.npts_ds = npts_ds
        self.query_num = query_num
        self.q_conv = nn.Conv1d(in_channel, in_channel, 1, bias=False)
        self.k_conv = nn.Conv1d(in_channel, in_channel, 1, bias=False)
        self.v_conv = nn.Conv1d(in_channel, in_channel, 1, bias=False)
        self.con1 = nn.Conv1d(in_channel * self.query_num, in_channel, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, xyz, pcd):
        # pcd -> (B, 128, 1024)
        # xyz -> (B, 3, 1024)
        xyz = xyz.permute(0, 2, 1)
        q = self.q_conv(pcd)  # (B, C, N) -> (B, C, N)
        k = self.k_conv(pcd)  # (B, C, N) -> (B, C, N)
        v = self.v_conv(pcd)  # (B, C, N) -> (B, C, N)
        energy = rearrange(q, 'B C N -> B N C').contiguous() @ k  # (B, N, C) @ (B, C, N) -> (B, N, N)
        scale_factor = math.sqrt(q.shape[-2])
        attention = self.softmax(energy / scale_factor)  # (B, N, N) -> (B, N, N)
        selection = torch.sum(attention, dim=-2)  # (B, N, N) -> (B, N)
        self.idx = selection.topk(self.npts_ds, dim=-1)[1]  # (B, N) -> (B, M)
        # 根据索引取出对应行和列的值
        se_attention = torch.gather(attention, dim=1,
                                    index=self.idx.unsqueeze(-1).repeat(1, 1, self.idx.shape[1]))
        se_v = index_points(v.permute(0, 2, 1), self.idx)  # (B, M, C)
        v = se_attention @ se_v  # (B, M, M) @ (B, M, C) -> (B, M, C)
        out = rearrange(v, 'B M C -> B C M').contiguous()  # (B, M, C) -> (B, C, M)
        new_xyz = index_points(xyz, self.idx)  # (B, M, C)

        return new_xyz.permute(0, 2, 1), out

class HybirdRear(nn.Module):
    def __init__(self, in_channel, pool_types=['avg', 'max']):
        super(HybirdRear, self).__init__()
        self.pool_types = pool_types
        self.mlp = nn.Conv1d(in_channel, in_channel, 1)

    def forward(self, x):
        # x -> (B, C, N)
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = x.mean(dim=-1, keepdim=True)[0]
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = x.max(dim=-1, keepdim=True)[0]
                channel_att_raw = self.mlp(max_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum)
        return x * scale