"""
Project: Advancing Surgical VQA with Scene Graph Knowledge
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
与feat_extract_visual 脚本相似，都是从图像中提取视觉特征。但不再是提取整个图像的网格特征，而是提取特定于图像中每个对象（Object）的特征。
这种方法提取的特征包含了每个物体的视觉外观、位置信息和类别信息，是一种更结构化、信息密度更高的特征表示，非常适合定位、需要理解物体间关系和属性的复杂视觉问答（VQA）任务。
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
import json
from torchvision.ops import roi_pool, roi_align
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    def __init__(
        self,
    ):
        super(FeatureExtractor, self).__init__()
        # visual feature extraction
        self.img_feature_extractor = models.resnet18(pretrained=True)
        self.img_feature_extractor = torch.nn.Sequential(
            *(list(self.img_feature_extractor.children())[:-2])
        )
        self.max_bbox = 20

    def forward(self, img, boxes, classes):
        # 处理没有物体的情况
        if len(boxes[0]) == 0:
            final_outputs = torch.zeros((self.max_bbox, 530)).cuda()
            return final_outputs
        # 整张图的特征图
        outputs = self.img_feature_extractor(img)
        # outputs = roi_pool(outputs, boxes, spatial_scale=0.031, output_size=1).squeeze(-1).squeeze(-1)
        outputs = (
            # roi_align 根据bbox信息从全局特征图裁剪每个物体对应的特征，output_size=1表示每个物体的特征被压缩为1x1的空间维度
            roi_align(outputs, boxes, spatial_scale=0.031, output_size=1)
            .squeeze(-1)
            .squeeze(-1)
            # squeeze降维后得到每个物体的512维特征向量
        )
        # 归一化bbox坐标，转为相对坐标来消除图像尺寸的影响
        boxes_norm = torch.FloatTensor(
            [[i[0] / 860, i[1] / 480, i[2] / 860, i[3] / 480] for i in boxes[0]]
        ).cuda()
        # 拼接类别信息、归一化后的bbox坐标和视觉特征，总维度为14+4+512=530
        outputs = torch.cat([classes, boxes_norm, outputs], 1)
        # 如果物体数量少于max_bbox，则用零向量进行填充
        padd_outputs = torch.zeros((self.max_bbox - len(outputs), 530)).cuda()
        final_outputs = torch.cat([outputs, padd_outputs])

        return final_outputs


# input data and IO folder location
filenames = []
seq = [
    "VID73",
    "VID40",
    "VID62",
    "VID42",
    "VID29",
    "VID56",
    "VID50",
    "VID78",
    "VID66",
    "VID13",
    "VID52",
    "VID06",
    "VID36",
    "VID05",
    "VID12",
    "VID26",
    "VID68",
    "VID32",
    "VID49",
    "VID65",
    "VID47",
    "VID04",
    "VID23",
    "VID79",
    "VID51",
    "VID10",
    "VID57",
    "VID75",
    "VID25",
    "VID14",
    "VID15",
    "VID08",
    "VID80",
    "VID27",
    "VID70",
    "VID18",
    "VID48",
    "VID01",
    "VID35",
    "VID31",
    "VID22",
    "VID74",
    "VID02",
    "VID60",
    "VID43",
]

folder_head = "/media/camma/DATA/SurgicalVQA/CholecT50-challenge-train/data/"  # fill out with your your path

txt_folder = "./dataset_construction/data/qa_txt/"
folder_tail = "/*.txt"

for curr_seq in seq:
    filenames = filenames + glob(txt_folder + str(curr_seq) + folder_tail)

new_filenames = []
for i in filenames:
    frame_num = i.split(".txt")[0].split("/")[-1]
    name_head = "/".join(i.split("/")[:-1])
    name = name_head + "/%06d" % int(frame_num) + ".png"
    name = name.replace(txt_folder, folder_head)
    new_filenames.append(name)


transform = transforms.Compose(
    [
        transforms.Resize((480, 860)),  # 240,430
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# declare fearure extraction model
feature_network = FeatureExtractor()

# Set data parallel based on GPU
num_gpu = torch.cuda.device_count()
if num_gpu > 0:
    device_ids = np.arange(num_gpu).tolist()
    feature_network = nn.DataParallel(feature_network, device_ids=device_ids)

# Use Cuda
feature_network = feature_network.cuda()
feature_network.eval()

scene_graph_root = "./dataset_construction/data/scene_graph"


# labels
object_dict = {
    "abdominal_wall_cavity": 2,
    "cystic_duct": 5,
    "cystic_plate": 0,
    "gallbladder": 1,
    "gut": 6,
    "liver": 4,
    "omentum": 3,
    "bipolar": 7,
    "clipper": 8,
    "grasper": 9,
    "hook": 10,
    "irrigator": 11,
    "scissors": 12,
    "specimenbag": 13,
}


from tqdm import tqdm

for img_loc in tqdm(new_filenames):
    # 加载需要处理图像的场景图
    VID = img_loc.split("/")[-2] + "_"
    frame_name = str(int(img_loc.split("/")[-1].replace(".png", ""))) + ".json"
    with open(os.path.join(scene_graph_root, VID + frame_name), "r") as f:
        scene_graph = json.load(f)
    assert len(scene_graph["scenes"]) == 1

    # 解析物体框和类别
    boxes = []
    classes = []
    for o in scene_graph["scenes"][0]["objects"]:
        boxes.append(o["bbox"])
        classes_vector = torch.zeros(14)
        classes_vector[object_dict[o["component"]]] = 1
        classes.append(classes_vector.unsqueeze(0))
    boxes = [torch.Tensor(boxes) * 2] # 由于图像被放大了2倍，因此bbox坐标也相应放大
    assert len(boxes[0]) == len(classes)

    img = Image.open(img_loc)
    img = transform(img)

    if len(classes) > 0:
        classes = torch.cat(classes, 0)

    img = torch.unsqueeze(img, 0)
    with torch.no_grad():
        visual_features = feature_network(img, boxes, classes)
        # visual_features (L, D)
        visual_features = visual_features.data.cpu().numpy()
    # 最终得到现状为 (20, 530)的特征矩阵

    # save extracted features
    img_loc = img_loc.split("/")
    save_dir = os.path.join(
        "./dataset/roi_coord/" + img_loc[7], "vqa/img_features", "roi"
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
