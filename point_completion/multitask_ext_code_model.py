from pointnet.pointnet_model import PointNetfeat
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np
from utils.FPS import farthest_point_sample, index_points
import matplotlib.pyplot as plt


class Convlayer(nn.Module):
    def __init__(self, point_scales, globalfeat=True):
        super(Convlayer, self).__init__()
        self.point_scales = point_scales
        self.global_input = globalfeat
        self.conv1 = torch.nn.Conv2d(1, 64, (1, 3))
        self.conv2 = torch.nn.Conv2d(64, 64, 1)
        self.conv3 = torch.nn.Conv2d(64, 128, 1)
        self.conv4 = torch.nn.Conv2d(128, 256, 1)
        self.conv5 = torch.nn.Conv2d(256, 512, 1)
        self.conv6 = torch.nn.Conv2d(512, 1024, 1)
        self.maxpool = torch.nn.MaxPool2d((self.point_scales, 1), 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn6 = nn.BatchNorm2d(1024)
        self.counter = 0

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x_128 = F.relu(self.bn3(self.conv3(x)))
        x_256 = F.relu(self.bn4(self.conv4(x_128)))
        x_512 = F.relu(self.bn5(self.conv5(x_256)))
        x_1024 = F.relu(self.bn6(self.conv6(x_512)))
        x_128_squeezed = torch.squeeze(x_128, dim=-1)
        x_128 = torch.squeeze(self.maxpool(x_128), 2)
        x_256 = torch.squeeze(self.maxpool(x_256), 2)
        x_512 = torch.squeeze(self.maxpool(x_512), 2)
        x_1024 = torch.squeeze(self.maxpool(x_1024), 2)
        L = [x_1024, x_512, x_256, x_128]
        x = torch.cat(L, 1)
        if self.global_input:
            return x
        return x, x_128_squeezed


class Latentfeature(nn.Module):
    def __init__(self, num_scales, each_scales_size, point_scales_list):
        super(Latentfeature, self).__init__()
        self.num_scales = num_scales
        self.each_scales_size = each_scales_size
        self.point_scales_list = point_scales_list
        self.Convlayers1 = nn.ModuleList(
            [Convlayer(point_scales=self.point_scales_list[0], globalfeat=False) for i in range(self.each_scales_size)])
        self.Convlayers2 = nn.ModuleList(
            [Convlayer(point_scales=self.point_scales_list[1]) for i in range(self.each_scales_size)])
        self.Convlayers3 = nn.ModuleList(
            [Convlayer(point_scales=self.point_scales_list[2]) for i in range(self.each_scales_size)])
        self.conv1 = torch.nn.Conv1d(3, 1, 1)
        self.bn1 = nn.BatchNorm1d(1)
        self.count = 0

    def forward(self, x):
        outs = []
        for i in range(self.each_scales_size):
            app, points_features = self.Convlayers1[i](x[0])
            outs.append(app)
        for j in range(self.each_scales_size):
            outs.append(self.Convlayers2[j](x[1]))
        for k in range(self.each_scales_size):
            outs.append(self.Convlayers3[k](x[2]))
        latentfeature = torch.cat(outs, 2)
        latentfeature = latentfeature.transpose(1, 2)
        latentfeature = F.relu(self.bn1(self.conv1(latentfeature)))
        latentfeature = torch.squeeze(latentfeature, 1)
        segfeatures = latentfeature.view(-1, 1920, 1).repeat(1, 1, self.point_scales_list[0])
        segfeatures = torch.cat([points_features, segfeatures], 1)  # BS, 2048, 2048-512

        return latentfeature, segfeatures


# DECODER FOR PART SEGMENTATION
# from https://github.com/fxia22/pointnet.pytorch/blob/f0c2430b0b1529e3f76fb5d6cd6ca14be763d975/pointnet/model.py
class PointNetDenseCls(nn.Module):
    def __init__(self, k=2, input_dim=1088):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.input_dim = input_dim
        self.conv0 = torch.nn.Conv1d(2048, 1088, 1)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn0 = nn.BatchNorm1d(1088)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        if self.input_dim == 2048:
            x = F.relu(self.bn0(self.conv0(x)))
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        x = F.log_softmax(x.view(-1, self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x


class PointPyramidDecoder(nn.Module):
    def __init__(self, input_dimension=1920, crop_point_num=512):
        super(PointPyramidDecoder, self).__init__()
        self.crop_point_num = crop_point_num
        self.fc1 = nn.Linear(input_dimension, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)

        self.fc1_1 = nn.Linear(1024, 128 * 512)
        self.fc2_1 = nn.Linear(512, 64 * 128)  # nn.Linear(512,64*256) !
        self.fc3_1 = nn.Linear(256, 64 * 3)

        self.conv1_1 = torch.nn.Conv1d(512, 512, 1)  # torch.nn.Conv1d(256,256,1) !
        self.conv1_2 = torch.nn.Conv1d(512, 256, 1)
        self.conv1_3 = torch.nn.Conv1d(256, int((self.crop_point_num * 3) / 128), 1)
        self.conv2_1 = torch.nn.Conv1d(128, 6, 1)  # torch.nn.Conv1d(256,12,1) !

    def forward(self, x):
        x_1 = F.relu(self.fc1(x))  # 1024
        x_2 = F.relu(self.fc2(x_1))  # 512
        x_3 = F.relu(self.fc3(x_2))  # 256

        pc1_feat = self.fc3_1(x_3)
        pc1_xyz = pc1_feat.reshape(-1, 64, 3)  # 64x3 center1

        pc2_feat = F.relu(self.fc2_1(x_2))
        pc2_feat = pc2_feat.reshape(-1, 128, 64)
        pc2_xyz = self.conv2_1(pc2_feat)  # 6x64 center2

        pc3_feat = F.relu(self.fc1_1(x_1))
        pc3_feat = pc3_feat.reshape(-1, 512, 128)
        pc3_feat = F.relu(self.conv1_1(pc3_feat))
        pc3_feat = F.relu(self.conv1_2(pc3_feat))
        pc3_xyz = self.conv1_3(pc3_feat)  # 12x128 fine

        pc1_xyz_expand = torch.unsqueeze(pc1_xyz, 2)
        pc2_xyz = pc2_xyz.transpose(1, 2)
        pc2_xyz = pc2_xyz.reshape(-1, 64, 2, 3)
        pc2_xyz = pc1_xyz_expand + pc2_xyz
        pc2_xyz = pc2_xyz.reshape(-1, 128, 3)

        pc2_xyz_expand = torch.unsqueeze(pc2_xyz, 2)
        pc3_xyz = pc3_xyz.transpose(1, 2)
        pc3_xyz = pc3_xyz.reshape(-1, 128, int(self.crop_point_num / 128), 3)
        pc3_xyz = pc2_xyz_expand + pc3_xyz
        pc3_xyz = pc3_xyz.reshape(-1, self.crop_point_num, 3)

        return pc1_xyz, pc2_xyz, pc3_xyz  # center1 ,center2 ,fine


class OnionNet(nn.Module):
    def __init__(self, num_scales=3, each_scales_size=1, point_scales_list=[1536, 768, 384],
                 crop_point_num=512, num_classes=50, num_spheres=10, option=True):
        super(OnionNet, self).__init__()

        self.point_scales_list = point_scales_list
        self.num_classes = num_classes
        self.num_spheres = num_spheres
        self.ext_code_size = self.num_spheres * self.num_classes
        self.option = option
        self.radius_tensor = self.init_radius()
        self.encoder = Latentfeature(num_scales, each_scales_size, self.point_scales_list)
        self.seg_decoder_input_size = 2048
        if self.option:
            self.fc1 = torch.nn.Linear(self.ext_code_size, self.ext_code_size * 2)
            self.fc2 = torch.nn.Linear(self.ext_code_size * 2, self.ext_code_size * 2)
            self.fc3 = torch.nn.Linear(self.ext_code_size * 2, self.ext_code_size)
        self.pc_decoder_input_size = 1920 + self.ext_code_size

        # Decoder for segmentation
        self.seg_decoder = PointNetDenseCls(k=num_classes, input_dim=self.seg_decoder_input_size)

        # Decoder for point completion
        self.pc_decoder = PointPyramidDecoder(input_dimension=self.pc_decoder_input_size, crop_point_num=crop_point_num)

    def forward(self, x):
        original_x = x
        x1_index = farthest_point_sample(x, self.point_scales_list[1], RAN=True)
        x1 = index_points(x, x1_index)
        x1 = x1.cuda()
        x2_index = farthest_point_sample(x, self.point_scales_list[2], RAN=False)
        x2 = index_points(x, x2_index)
        x2 = x2.cuda()
        x = [x, x1, x2]

        x, x_seg = self.encoder(x)

        pred = self.seg_decoder(x_seg)
        prediction_seg = pred.cuda()
        prediction_seg = prediction_seg.view(-1, self.num_classes)
        pred_choice = prediction_seg.data.max(1)[1].cuda()
        onion_feat = self.spherical_features(original_x, pred_choice).cuda()
        if self.option:
            onion_feat = F.relu(self.fc1(onion_feat))
            onion_feat = F.relu(self.fc2(onion_feat))
            onion_feat = F.relu(self.fc3(onion_feat))
        x = torch.cat((x, onion_feat), dim=1)

        decoded_x = self.pc_decoder(x)
        return decoded_x, pred

    def init_radius(self, constant_area=False):
        self.r_max = np.sqrt(3)
        if constant_area:
            coeffs = np.array([np.power(2, i / 3) for i in range(self.num_spheres)])
            r1 = self.r_max / coeffs[-1]
            radius_array = r1 * coeffs
        else:
            radius_array = self.r_max * np.arange(1, self.num_spheres + 1) / self.num_spheres
        return torch.tensor(radius_array).view(radius_array.shape[0], 1, 1).cuda()

    def spherical_features(self, x, pred):
        batch_size = x.size(0)
        num_radius = self.radius_tensor.size(0)
        pred = pred.view(batch_size, x.size(1), 1)
        x_and_class = torch.cat((x, pred), dim=-1).cuda()
        distances_from_origin = torch.sqrt(torch.sum(x ** 2, dim=-1)).cuda()
        # point_belongs_to: each point is associated to a specific sphere (from 0 up to num_spheres-1)
        # distances_from_origin = distances_from_origin.repeat(num_radius, 1, 1) - self.radius_tensor
        dfo_m_r = (self.radius_tensor - distances_from_origin.repeat(num_radius, 1, 1)).cuda()
        dfo_m_r[dfo_m_r < 0] = self.r_max
        point_belongs_to = torch.min(dfo_m_r, dim=0)[1].cuda()
        id_and_pred = torch.cat((point_belongs_to.view(batch_size, -1, 1), pred.view(batch_size, -1, 1)), dim=-1)
        feat_batch = None
        for id_pc in range(batch_size):
            feat = None
            previous_sphere_feat = 0
            for id_sphere in range(self.num_spheres):
                current_sphere_feat = torch.bincount(
                    id_and_pred[id_pc, point_belongs_to[id_pc, :] == id_sphere, :][:, -1].to(torch.int).cuda(),
                    minlength=self.num_classes)
                current_sphere_feat = current_sphere_feat + previous_sphere_feat
                previous_sphere_feat = current_sphere_feat
                # tot_frequency: num points inside id_sphere (and the nested ones)
                tot_frequency = torch.sum(current_sphere_feat)
                normalized_current_sphere_feat = current_sphere_feat / tot_frequency.item() if tot_frequency.item() != 0 else current_sphere_feat
                normalized_current_sphere_feat = normalized_current_sphere_feat.view(1, self.num_classes)
                feat = torch.cat((feat, normalized_current_sphere_feat), dim=-1) if feat is not None else normalized_current_sphere_feat
            feat = feat.view(1, self.num_spheres*self.num_classes)
            feat_batch = torch.cat((feat_batch, feat), dim=0) if feat_batch is not None else feat
        return feat_batch


if __name__ == "__main__":
    input1 = torch.randn(64, 2048, 3)
    input2 = torch.randn(64, 512, 3)
    input3 = torch.randn(64, 256, 3)
    input_ = [input1, input2, input3]
    netG = OnionNet(3, 1, [2048, 512, 256], 1024)
    output = netG(input_)
    print(output)
