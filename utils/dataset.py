from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import json


class ShapeNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 class_choice=None,
                 split='train',
                 data_augmentation=False,
                 set_size=1):  # set_size: float: between 0 and 1 (percentage dataset)
        self.set_size = set_size if 0 < set_size <= 1 else 1
        self.npoints = npoints
        self.root = root
        # download Shapenet dataset command here and save in colab in /Shapenet
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.data_augmentation = data_augmentation

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                # e.g.: cat['Airplane'] = 02691156
                self.cat[ls[0]] = ls[1]
        # print(self.cat)
        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.id2cat = {v: k for k, v in self.cat.items()}

        self.meta = {}
        splitfile = os.path.join(self.root, 'train_test_split', 'shuffled_{}_file_list.json'.format(split))
        # from IPython import embed; embed()
        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            self.meta[item] = []

        for file in filelist:
            _, category, uuid = file.split('/')
            if category in self.cat.values():
                # randomly select (set_size*100)% point clouds from the original ones
                val = np.random.uniform(0, 1)
                if val <= set_size:  # e.g.: if set_size=1, then val<=1 will be always true
                                     # (then take 100% point clouds)
                    self.meta[self.id2cat[category]].append((os.path.join(self.root, category, 'points', uuid + '.pts')))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        # if split == 'train':
        #     print(self.classes)
        #     #Note: len(filelist) contains the info about the point clouds of the other classes (not our 7!)
        #     print(f"Total point clouds selected: {sum([len(array) for _, array in self.meta.items()])}/{len(filelist)}")

    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        choice = np.random.choice(len(point_set), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]
        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        point_set = torch.from_numpy(point_set)
        # Uncomment if label is needed
        # cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        return point_set

    def __len__(self):
        return len(self.datapath)

    def get_point_cloud_by_category(self, category, index=0):
        if category not in self.meta.keys():
            print(f"Category {category} not found.")
            print(f"Choose one of these: {self.meta.keys()}")
            return None
        if index >= len(self.meta[category]):
            print(f"Taking index 0 instead of {index}")
            index = 0
        file_path = self.meta[category][index]
        point_set = np.loadtxt(file_path).astype(np.float32)
        choice = np.random.choice(len(point_set), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]
        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale
        point_set = torch.from_numpy(point_set)
        return point_set

    def get_categories(self):
        return self.meta.keys()
