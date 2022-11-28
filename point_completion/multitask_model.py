from pointnet.pointnet_model import PointNetfeat
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from utils.FPS import farthest_point_sample, index_points


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
        if self.counter == 0:
            print(x_128_squeezed.shape)
            self.counter = 1
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
        if self.count == 0:
            print(segfeatures.shape, points_features.shape)
            self.count = 1
        segfeatures = torch.cat([points_features, segfeatures], 1) # BS, 2048, 2048-512

        return latentfeature, segfeatures


# DECODER FOR PART SEGMENTATION
# from https://github.com/fxia22/pointnet.pytorch/blob/f0c2430b0b1529e3f76fb5d6cd6ca14be763d975/pointnet/model.py
class PointNetDenseCls(nn.Module):
    def __init__(self, k = 2, input_dim=1088):
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
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
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


class MultiTaskCompletionNet(nn.Module):
    def __init__(self, num_scales=3, each_scales_size=1, point_scales_list=[1536, 768, 384]\
                 , crop_point_num=512, num_points=2048, num_classes=50, pfnet_encoder = True):
        super(MultiTaskCompletionNet, self).__init__()

        self.point_scales_list = point_scales_list
        #  Encoder Definition
        self.encoder = Latentfeature(num_scales, each_scales_size, self.point_scales_list)\
            if pfnet_encoder else PointNetfeat(global_feat=False)
        self.seg_decoder_input_size = 2048 if pfnet_encoder else 1088
        self.pc_decoder_input_size = 1920 if pfnet_encoder else 1024
        self.pfnet_encoder = pfnet_encoder
        # Decoder for segmentation
        self.seg_decoder = PointNetDenseCls(k=num_classes, input_dim = self.seg_decoder_input_size)

        # Decoder for point completion
        self.pc_decoder = PointPyramidDecoder(input_dimension=self.pc_decoder_input_size, crop_point_num=crop_point_num)

    def forward(self, x):
        if self.pfnet_encoder:
            x1_index = farthest_point_sample(x, self.point_scales_list[1], RAN=True)
            x1 = index_points(x, x1_index)
            x1 = x1.cuda()
            x2_index = farthest_point_sample(x, self.point_scales_list[2], RAN=False)
            x2 = index_points(x, x2_index)
            x2 = x2.cuda()
            x = [x, x1, x2]
        else:
            x = x.permute(0, 2, 1)  # [BS, N, 3] => [BS, 3, N]
        x, x_seg = self.encoder(x)
        prediction_seg = self.seg_decoder(x_seg)   #BS, 2048-512, 27 -> BS, 2048-512, 1 ->
        decoded_x = self.pc_decoder(x)
        return decoded_x, prediction_seg


if __name__=="__main__":
    input1 = torch.randn(64, 2048, 3)
    input2 = torch.randn(64, 512, 3)
    input3 = torch.randn(64, 256, 3)
    input_ = [input1, input2, input3]
    netG = MultiTaskCompletionNet(3, 1, [2048, 512, 256], 1024)
    output = netG(input_)
    print(output)