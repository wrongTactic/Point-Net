#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@File: dataset.py
@Time: 2020/1/2 10:26 AM
"""

import os
import torch
import json
import h5py
from glob import glob
import numpy as np
import torch.utils.data as data

shapenetpart_cat2id = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4,
                       'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9,
                       'motorbike': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}
shapenetpart_seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
shapenetpart_seg_start_index = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]


class ShapeNetPart(data.Dataset):
    def __init__(self, root, class_choice=None,
                 num_points=2048, split='train', load_name=True, load_file=True,
                 segmentation=False):

        assert num_points <= 2048

        assert split.lower() in ['train', 'test', 'val', 'trainval', 'all']

        self.root = root
        self.class_choice = ["airplane", "car", "chair", "lamp", "motorbike", "mug", "table"] \
            if class_choice is None else [class_choice]
        self.num_points = num_points
        self.split = split
        self.load_name = load_name
        self.load_file = load_file
        self.segmentation = segmentation

        self.path_h5py_all = []
        self.path_name_all = []
        self.path_file_all = []

        if self.split in ['train', 'trainval', 'all']:
            self.get_path('train')
        if self.split in ['val', 'trainval', 'all']:
            self.get_path('val')
        if self.split in ['test', 'all']:
            self.get_path('test')

        self.path_h5py_all.sort()
        self.data, self.label, self.seg = self.load_h5py(self.path_h5py_all)
        if self.load_name or self.class_choice != None:
            self.path_name_all.sort()
            self.name = self.load_json(self.path_name_all)  # load label name

        if self.load_file:
            self.path_file_all.sort()
            self.file = np.array(self.load_json(self.path_file_all))  # load file name

        if self.class_choice != None:
            if len(self.class_choice) == 1:
                indices = np.array(np.array(self.name) == self.class_choice).squeeze()
            else:
                indices = np.array(np.sum(np.array(self.name).reshape(-1,1) == self.class_choice, axis=-1, dtype=np.bool8)).squeeze()
            self.data = self.data[indices]
            self.label = self.label[indices]
            if self.segmentation:
                self.seg = self.seg[indices]
                id_choice = [shapenetpart_cat2id[classs] for classs in self.class_choice]
                self.seg_num_all = np.sum(np.array(shapenetpart_seg_num)[id_choice])
                self.seg_start_index = np.array(shapenetpart_seg_start_index)[id_choice]
                self.seg_num_class = np.array(shapenetpart_seg_num)[id_choice]
                self.class_idx = np.array(id_choice)
                offset = 0
                self.map_class_offset = dict()
                for idx, classs in zip(self.class_idx, self.class_choice):
                    self.map_class_offset[classs] = offset
                    offset += shapenetpart_seg_num[idx]
            if self.load_file:
                self.file = self.file[indices]
        elif self.segmentation:
            self.seg_num_all = 50
            self.seg_start_index = 0

    def get_path(self, type):
        path_h5py = os.path.join(self.root, '*%s*.h5' % type)
        self.path_h5py_all += glob(path_h5py)
        if self.load_name:
            path_json = os.path.join(self.root, '%s*_id2name.json' % type)
            self.path_name_all += glob(path_json)
        if self.load_file:
            path_json = os.path.join(self.root, '%s*_id2file.json' % type)
            self.path_file_all += glob(path_json)
        return

    def load_h5py(self, path):
        all_data = None
        all_label = None
        all_seg = None
        for h5_name in path:
            f = h5py.File(h5_name, 'r+')
            data = np.array(f['data'][:].astype('float32'))
            label = np.array(f['label'][:].astype('int64'))
            if self.segmentation:
                seg = np.array(f['seg'][:].astype('int64'))
            f.close()
            all_data = data if all_data is None else np.concatenate([all_data, data])
            all_label = label if all_label is None else np.concatenate([all_label, label])
            if self.segmentation:
                all_seg = seg if all_seg is None else np.concatenate([all_seg, seg])
        return all_data, all_label, all_seg

    def load_json(self, path):
        all_data = []
        for json_name in path:
            j = open(json_name, 'r+')
            data = json.load(j)
            all_data += data
        return all_data

    def __getitem__(self, item):
        point_set = self.data[item][:self.num_points]
        label = self.label[item]
        if self.load_name:
            name = self.name[item]  # get label name
        if self.load_file:
            file = self.file[item]  # get file name

        # convert numpy array to pytorch Tensor
        point_set = torch.from_numpy(point_set)
        label = torch.from_numpy(np.array([label]).astype(np.int64))
        label = label.squeeze(0)

        if self.segmentation:
            seg = self.seg[item]
            offset = 0
            for idx in range(label.item()):
                if idx in self.class_idx:
                    offset += shapenetpart_seg_num[idx]
            start_index = shapenetpart_seg_start_index[label.item()]
            func_map = np.vectorize(lambda x: x-start_index+offset)
            seg = torch.from_numpy(func_map(seg))
            return point_set, seg
        else:
            return point_set

    def __len__(self):
        return self.data.shape[0]

