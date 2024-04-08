from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import cv2
import os, torch
import dlib
import torch.nn as nn
from collections import OrderedDict
torch.set_printoptions(precision=3, edgeitems=14, linewidth=350)
# 加载人脸检测与关键点定位
#http://dlib.net/python/index.html#dlib_pybind11.get_frontal_face_detector
detector = dlib.get_frontal_face_detector()
#http://dlib.net/python/index.html#dlib_pybind11.shape_predictor
criticPoints = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#对于68个检测点，将人脸的几个关键点排列成有序，便于后面的遍历
shape_predictor_68_face_landmark=OrderedDict([
    ('mouth',(48,68)),
    ('right_eyebrow',(17,22)),
    ('left_eye_brow',(22,27)),
    ('right_eye',(36,42)),
    ('left_eye',(42,48)),
    ('nose',(27,36)),
    ('jaw',(0,17))
])
def predict2Np(predict):
    # 创建68*2关键点的二维空数组[(x1,y1),(x2,y2)……]
    dims=np.zeros(shape=(predict.num_parts,2),dtype=int)
    #遍历人脸的每个关键点获取二维坐标
    length=predict.num_parts
    for i in range(0,length):
        dims[i]=(predict.part(i).x, predict.part(i).y)
    return dims
def dimsPoints(detected,frame):
    for (step,locate) in enumerate(detected):
        #对获取的人脸框再进行人脸关键点检测
        #获取68个关键点的坐标值
        dims=criticPoints(frame,locate)

        #将得到的坐标值转换为二维
        dims=predict2Np(dims)
    return dims
# 数据集
def dlibdata(path_on0, path_apex0):
    print(path_on0)
    print(path_apex0)
    image_on0 = cv2.imread(path_on0)

    image_apex0 = cv2.imread(path_apex0)

    detected_on0 = detector(image_on0)
    detected_apex0 = detector(image_apex0)

    # print(path_on0)
    dim_on0 = dimsPoints(detected_on0, image_on0)
    dim_apex0 = dimsPoints(detected_apex0, image_apex0)
    dim_on0 = torch.from_numpy(dim_on0)
    dim_apex0 = torch.from_numpy(dim_apex0)

    # p=2就是计算欧氏距离，p=1就是曼哈顿距离，例如上面的例子，距离是1.
    pdist = nn.PairwiseDistance(p=2)
    # 计算各自每一行之间的欧式距离。
    output = pdist(dim_on0, dim_apex0)
    # 将apex帧与onset帧地标相减
    X = dim_apex0 - dim_on0
    # 将大于0的元素替换为1，小于0的元素替换为-1
    X[X < 0] = -1
    X[X > 0] = 1

    # 获取张量的shape
    dim0, dim1 = X.shape
    X1 = np.ones((68, 2))
    t = torch.tensor(X1)

    # 遍历张量
    for i in range(dim0):
        for j in range(dim1):
            X1[i][j] = X[i][j] * output[i]
    X2 = dim_apex0 + X1
    # 计算地标之间欧氏距离和梯度
    rslt = []

    for itemx, x in enumerate(X2):
        for itemy, y in enumerate(dim_on0):
            d = pdist(x, y)
            if x[0] - y[0] != 0:
                z = (x[1] - y[1]) / (x[0] - y[0])
            else:
                z = 0
            rslt.append(d)
            rslt.append(z)

    # 对特征向量进行归一化
    # F = MaxMinNormalization(F)
    rslt = torch.tensor(rslt)
    rslt = rslt.type(torch.FloatTensor)

    return rslt
def make_dataset(image_list, labels):
    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
        print(images)

    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], val.split()[1], val.split()[2], val.split()[3],
                   np.array([int(la) for la in val.split()[4:]])) for val in image_list]

      else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]

    return images


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

class ImageList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path_on, path_apex, path_on2, path_apex2, target = self.imgs[index]
        path_on = path_on.replace("\\", '/')
        path_apex = path_apex.replace("\\", '/')

        image_on = self.loader(path_on)
        image_apex = self.loader(path_apex)

        # image_on = cv2.imread(path_on)
        # image_apex = cv2.imread(path_apex)
        # image_on = image_on[:, :, ::-1]  # BGR to RGB
        # image_apex = image_apex[:, :, ::-1]
        # rslt = dlibdata(path_on2, path_apex2)
        if self.transform is not None:
            img_on = self.transform(image_on)
            img_apex = self.transform(image_apex)
        if self.target_transform is not None:
            target = self.target_transform(target)
        rslt = [0]
        rslt = torch.tensor(rslt)
        rslt = rslt.type(torch.FloatTensor)

        return img_on, img_apex, rslt, target

    def __len__(self):
        return len(self.imgs)

class ImageValueList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None,
                 loader=rgb_loader):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.values = [1.0] * len(imgs)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def set_values(self, values):
        self.values = values

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)