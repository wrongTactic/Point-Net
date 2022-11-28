#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #
    # batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


class DGCNN(nn.Module):
    def __init__(self, args):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.size_encoder)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.size_encoder, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))


    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        x = x.view(batch_size, self.args.size_encoder*2)
        return x

class Decoder(nn.Module):
    ''' Just a lightweight Fully Connected decoder:
    '''

    def __init__(self, args):
        super(Decoder, self).__init__()
        self.num_points = args.num_points
        self.fc1 = nn.Linear(args.size_encoder, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.fc5 = nn.Linear(1024, self.num_points * 3)
        #self.dp = nn.Dropout(p=args.dropout)
        self.th = nn.Tanh()

    def forward(self, x):
        batch_size = x.size()[0]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        #x = self.dp(x)
        x = self.th(self.fc5(x))
        x = x.view(batch_size, self.num_points, 3)
        return x


class PyramidDecoder(nn.Module):
    ''' Point pyramid decoder from PF_Net:
    '''

    def __init__(self, args):
        super(PyramidDecoder, self).__init__()
        self.num_points = args.num_points
        self.fc1 = nn.Linear(args.size_encoder*2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)

        self.fc1_1 = nn.Linear(1024, 256 * self.num_points)
        self.fc2_1 = nn.Linear(512, 128 * 256)
        self.fc3_1 = nn.Linear(256, 128 * 3)

        self.conv1_1 = torch.nn.Conv1d(self.num_points, self.num_points, 1)
        self.conv1_2 = torch.nn.Conv1d(self.num_points, 512, 1)
        self.conv1_3 = torch.nn.Conv1d(512, int((self.num_points * 3) / 256), 1)
        self.conv2_1 = torch.nn.Conv1d(256, 6, 1)


    def forward(self, x):
        x_1 = F.relu(self.fc1(x))  # 1024
        x_2 = F.relu(self.fc2(x_1))  # 512
        x_3 = F.relu(self.fc3(x_2))  # 256

        pc1_feat = self.fc3_1(x_3)
        pc1_xyz = pc1_feat.reshape(-1, 128, 3)  # 128x3 [center1 ,coarse sampling] final!

        pc2_feat = F.relu(self.fc2_1(x_2))
        pc2_feat = pc2_feat.reshape(-1, 256, 128)
        pc2_xyz = self.conv2_1(pc2_feat)  # 256x128 -> 6x128 [center2, fine sampling]

        pc3_feat = F.relu(self.fc1_1(x_1))
        pc3_feat = pc3_feat.reshape(-1, self.num_points, 256)
        pc3_feat = F.relu(self.conv1_1(pc3_feat))  # 1024x256 -> 1024x256
        pc3_feat = F.relu(self.conv1_2(pc3_feat))  # 1024x256 -> 512x256
        pc3_xyz = self.conv1_3(pc3_feat)  # 512x256 -> 12x256 complete

        pc1_xyz_expand = torch.unsqueeze(pc1_xyz, 2)  # 128x1x3
        pc2_xyz = pc2_xyz.transpose(1, 2)   # 128x6
        pc2_xyz = pc2_xyz.reshape(-1, 128, 2, 3)  # 128x2x3
        pc2_xyz = pc1_xyz_expand + pc2_xyz
        pc2_xyz = pc2_xyz.reshape(-1, 256, 3)  # 128x2x3 -> 256x3 final!

        pc2_xyz_expand = torch.unsqueeze(pc2_xyz, 2)  # 256x1x3
        pc3_xyz = pc3_xyz.transpose(1, 2)  # 256x12
        pc3_xyz = pc3_xyz.reshape(-1, 256, int(self.num_points / 256), 3)  # 256x4x3
        pc3_xyz = pc2_xyz_expand + pc3_xyz
        pc3_xyz = pc3_xyz.reshape(-1, self.num_points, 3)  #1024x3 final!

        return pc1_xyz, pc2_xyz, pc3_xyz  # center1 ,center2 ,complete


class DGCNN_AutoEncoder(nn.Module):
    '''
  Complete AutoEncoder Model:
  Given an input point cloud X:
      - Step 1: encode the point cloud X into a latent low-dimensional code
      - Step 2: Starting from the code geneate a representation Y as close as possible to the original input X


  '''

    def __init__(self, args):
        super(DGCNN_AutoEncoder, self).__init__()
        #print("PointNet AE Init - num_points (# generated): %d" % num_points)

        # Encoder Definition
        self.encoder = DGCNN(args=args)


        # Decoder Definition
        self.decoder = PyramidDecoder(args=args) if args.type_decoder == "pyramid" else Decoder(args=args)

    def forward(self, x):
        BS, N, dim = x.size()
        #print(x.size())
        assert dim == 3, f"Fail: expecting 3 (x-y-z) as last tensor dimension! Found {dim}"

        #  Refactoring batch for 'PointNetfeat' processing
        x = x.permute(0, 2, 1)  # [BS, N, 3] => [BS, 3, N]

        # Encoding
        code = self.encoder(x)  # [BS, 3, N] => [BS, size_encoder]

        # Decoding
        decoded = self.decoder(code)

        return decoded #either a pointcloud [BS, num_points, 3] or a tuple of 3 pointclouds 3 x [BS, 3, num_points]

        return decoded



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=0.999, help="decay rate for second moment")
    parser.add_argument("--size_encoder", type=int, default=7, help="How long to wait after last time val loss improved.")
    parser.add_argument("--dropout", type=int, default=0, help="How long to wait after last time val loss improved.")
    opt = parser.parse_args()
    model = DGCNN(opt)
    model.forward(torch.rand((32, 3, 1024)))
