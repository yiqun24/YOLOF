from functools import partial
import torch.nn as nn
import torch
# a = torch.randn(1, 4, 2, 2)
# b = torch.randn(1, 2, 2, 2)
# print(a)
# print(b)
# print(b.repeat(1, 2))

# a = torch.randn([2, 3, 2])
# print(a.repeat(1, 2))

# T1 = torch.tensor([[1, 2, 3],
#                    [4, 5, 6],
#                    [7, 8, 9]])
# # 假设是时间步T2
# print(T1[:, 2])
# T2 = torch.tensor([[10, 20, 30],
#                    [40, 50, 60],
#                    [70, 80, 90]])
# print(torch.stack([T1, T2], dim=1))
# print(torch.stack([T1, T2], dim=1).shape)
# print(torch.cat([T1, T2], dim=1))
# print((torch.cat([T1, T2], dim=1)).shape)

H, W = 5, 5
# print(torch.arange(H))
# anchor_y, anchor_x = torch.meshgrid([torch.arange(H), torch.arange(W)], indexing='xy')
# a = torch.randn(5, 4)
# b = torch.randn(5, 5, 4)
# c = a * b
# d = a[None, ...] * b
# e = a[..., :2] * b[..., 2:]
# print(e)
# print(e.shape)
# f = b[..., :2] + e
# print(f.shape)
# print(c)
# print(d)

a = torch.randn(3, 3)

