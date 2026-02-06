"""
Project: Advancing Surgical VQA with Scene Graph Knowledge
-----
Copyright (c) University of Strasbourg, All Rights Reserved.

遍历图像数据集，使用ResNet-18提取视觉特征，并将这些特征保存为HDF5文件。
"""
import os
import sys
import h5py
import argparse

import torch
import torchvision.models
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
from glob import glob

import torch
from torch import nn
from torchvision import models


class FeatureExtractor(nn.Module):
    def __init__(self, patch_size=4):
        super(FeatureExtractor, self).__init__()
        # visual feature extraction
        self.img_feature_extractor = models.resnet18(pretrained=True)

        # 只需空间特征图，不需要最后的全连接层和池化层
        self.img_feature_extractor = torch.nn.Sequential(
            *(list(self.img_feature_extractor.children())[:-2])
        )
        # 自适应平均池化，调整输出特征图的空间尺寸
        self.resize_dim = nn.AdaptiveAvgPool2d((patch_size, patch_size))

    def forward(self, img):
        outputs = self.resize_dim(self.img_feature_extractor(img))
        return outputs


# input data and IO folder location
filenames = []
seq = [
    "1",
    "2",
    "3",
    "4",
    "6",
    "7",
    "8",
    "9",
    "10",
    "13",
    "14",
    "15",
    "16",
    "18",
    "20",
    "5",
    "11",
    "12",
    "17",
    "19",
    "26",
    "27",
    "31",
    "21",
    "22",
    "23",
    "24",
    "25",
    "28",
    "29",
    "30",
    "32",
    "33",
    "34",
    "35",
    "36",
    "37",
    "38",
    "39",
    "40",
]
folder_head = "/media/mobarak/data/lalith_dataset/Cholec80-VQA/cropped_image/"
folder_tail = "/*.png"
for curr_seq in seq:
    filenames = filenames + glob(folder_head + str(curr_seq) + folder_tail)

new_filenames = []
for filename in filenames:
    frame_num = int(filename.split("/")[-1].split(".")[0])
    # sample every 25th frame
    if frame_num % 25 == 0:
        new_filenames.append(filename)

transform = transforms.Compose(
    [
        transforms.Resize((300, 256)),
        transforms.ToTensor(),
        # resnet-18 在ImageNet上训练，因此输入数据也应该进行相同的标准化处理
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# arguments
parser = argparse.ArgumentParser(description="feature extractor")
parser.add_argument("--patch_size", type=int, default=1, help="")
args = parser.parse_args()


# declare fearure extraction model
feature_network = FeatureExtractor(patch_size=args.patch_size)

# Set data parallel based on GPU
num_gpu = torch.cuda.device_count()
if num_gpu > 0:
    device_ids = np.arange(num_gpu).tolist()
    feature_network = nn.DataParallel(feature_network, device_ids=device_ids)

# Use Cuda
feature_network = feature_network.cuda()
feature_network.eval()

for img_loc in new_filenames:
    # get visual features
    print(img_loc)
    img = Image.open(img_loc)
    img = transform(img)
    img = torch.unsqueeze(img, 0) # add batch dimension
    with torch.no_grad():
        visual_features = feature_network(img)
        visual_features = torch.flatten(visual_features, start_dim=2)
        visual_features = visual_features.permute((0, 2, 1)) # (1, num_patches, feature_dim)
        visual_features = visual_features.squeeze(0)
        visual_features = visual_features.data.cpu().numpy()

    # save extracted features
    img_loc = img_loc.split("/")
    save_dir = "/" + os.path.join(
        img_loc[0],
        img_loc[1],
        img_loc[2],
        img_loc[3],
        img_loc[4],
        img_loc[5],
        img_loc[6],
        img_loc[7],
        "vqa/img_features",
        (str(args.patch_size) + "x" + str(args.patch_size)),
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # save to file
    hdf5_file = h5py.File(
        os.path.join(save_dir, "{}.hdf5".format(img_loc[-1].split(".")[0])), "w"
    )
    print(os.path.join(save_dir, "{}.hdf5".format(img_loc[-1].split(".")[0])))
    hdf5_file.create_dataset("visual_features", data=visual_features)
    hdf5_file.close()
    print("save_dir: ", save_dir, " | visual_features: ", visual_features.shape)
