import os
import numpy as np
import warnings
import pickle
import h5py
import torch
from torch.utils.data import Dataset
from pathlib import Path
import open3d as o3d
from ..build import DATASETS
warnings.filterwarnings('ignore')

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def _get_data_files(list_filename):
    with open(list_filename) as f:
        # return [line.rstrip()[5:] for line in f]
        return [line.strip()+'.pcd' for line in f]

def _load_data_file(name):
    # print(name)
    pcd = o3d.io.read_point_cloud(name, format='pcd')
    data = np.array(pcd.points)
    # label =
    # return data, label
    return data

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))  #centroids
    distance = np.ones((N,)) * 1e10
    #随机生成一个数
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


@DATASETS.register_module()
class FG3DPoint_Car(Dataset):
    classes = ['armored',
                'atv',
                'bus',
                'cabriolet',
                'coupe',
                'formula',
                'jeep',
                'limousine',
                'microbus',
                'muscle',
                'pickup',
                'racer',
                'retro',
                'scooter',
                'sedan',
                'sports',
                'suv',
                'tricycle',
                'truck',
                'wagon']

    # class_choice可选 Airplane，Car，Chair, All

    def __init__(self,
                 num_points=1024,
                 data_dir="./data/FG3DPoint/Car_10000",
                 split='train',
                 transform=None
                 ):
        self.npoints = num_points #sample point
        # self.uniform = args.use_uniform_sample # Using Sampling, default: true
        self.transform = transform
        self.split = split
        # self.use_normals = args.use_normals #Using normal information, default: False
        # self.data_path = args.data_path
        if split == 'train':
            self.train = True
        else:
            self.train = False

        self.data_dir = data_dir
        # self.data_dir = os.path.abspath(r'D:\Desktop\pointnet\Deeplearning_Pytorch\data\FG3D\Airplane')
    # 类别映射
        self.catfile = os.path.join(self.data_dir, 'subcategories.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

    # print(self.data_dir)
        if self.train:
            self.files = _get_data_files(os.path.join(self.data_dir, 'train_file.txt'))
        else:
            self.files = _get_data_files(os.path.join(self.data_dir, 'test_file.txt'))

        point_list, label_list = [], []
        for f in self.files:
            # print(f)
            points = _load_data_file(os.path.join(self.data_dir, f)).astype(np.float32)
            labels = f.rpartition('_')[0]
            # cls = self.classes[self.datapath[index][0]]
            point_list.append(np.expand_dims(points, 0))
            labels = np.array([self.classes[labels]]).astype(int)
            label_list.append(np.expand_dims(labels, 0))
        self.points = np.concatenate(point_list, 0)
        self.labels = np.concatenate(label_list, 0)

    def __len__(self):
        return self.points.shape[0]

    def _get_item(self, index):
        # pt_idxs = np.arange(0, self.points.shape[1])  # 2048
        # if self.train:
        #     np.random.shuffle(pt_idxs)
        # current_points = self.points[index, pt_idxs].copy()
        label = self.labels[index].astype(np.int32)
        if self.uniform:  # Using the sampling method FPS
            # current_points = farthest_point_sample(current_points, self.npoints)
            current_points = farthest_point_sample(self.points[index], self.npoints)
        # else:
        #     # Using the sampling method Random
        #     choice = np.random.choice(len(current_points), self.npoints, replace=True)
        #     current_points = current_points[choice, :]

        # current_points = pc_normalize(current_points) #FG3D datasets have been speciﬁed,so there is no need to formalize
        current_points = pc_normalize(current_points)
        return current_points, label[0]

    def __getitem__(self, index):
        # return self._get_item(index)  # (B, N, 3)
        pointcloud = farthest_point_sample(self.points[index], self.npoints)
        pointcloud = pc_normalize(pointcloud)
        # pointcloud = self.points[index][:self.num_points]
        label = self.labels[index]

        if self.split == 'train':
            np.random.shuffle(pointcloud)
        data = {'pos': pointcloud,
                'y': label
                }
        if self.transform is not None:
            data = self.transform(data)

        if 'heights' in data.keys():
            data['x'] = torch.cat((data['pos'], data['heights']), dim=1)
        else:
            data['x'] = data['pos']
        return data

    @property
    def num_classes(self):
        return np.max(self.labels) + 1
