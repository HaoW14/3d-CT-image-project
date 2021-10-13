import SimpleITK as sitk
import numpy as np
from scipy import ndimage
import torch.nn.functional as F
import torch
from torchvision import transforms as T

def Z_score_norm3D(image):
    image_mean = np.mean(image)
    image_std = np.std(image)
    image = (image - image_mean)/image_std
    return image

def cut_off(image,min,max):
    image[image<min] = min
    image[image > max] = max
    return image



def sitk_read_raw(img_path): # 读取3D图像并rescale（因为一般医学图像并不是标准的[1,1,1]scale）
    nda = sitk.ReadImage(img_path)
    if nda is None:
        raise TypeError("input img is None!!!")
    nda = sitk.GetArrayFromImage(nda)  # channel first
    return nda

def to_onehot(label,class_num): #s*h*w  or h * w
    x = np.zeros((class_num,label.shape[0],label.shape[1],label.shape[2]))
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            for k in range(label.shape[2]):
                x[int(label[i,j,k]),i,j,k] = 1
    return x   #  c* h * w  or c* s* h * w


import random
def random_crop_2d(img, label, crop_size):
    random_x_max = img.shape[0] - crop_size[0]
    random_y_max = img.shape[1] - crop_size[1]

    if random_x_max < 0 or random_y_max < 0:
        return None

    x_random = random.randint(0, random_x_max)
    y_random = random.randint(0, random_y_max)

    crop_img = img[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1]]
    crop_label = label[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1]]

    return crop_img, crop_label


def random_crop_3d(img, label, crop_size):

    random_x_max = img.shape[0] - crop_size[0]
    random_y_max = img.shape[1] - crop_size[1]
    random_z_max = img.shape[2] - crop_size[2]

    if random_x_max < 0 or random_y_max < 0 or random_z_max < 0: #不能比裁剪的更小
        return None

    x_random = random.randint(0, random_x_max)
    y_random = random.randint(0, random_y_max)
    z_random = random.randint(0, random_z_max)

    crop_img = img[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1], z_random:z_random + crop_size[2]]
    crop_label = label[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1], z_random:z_random + crop_size[2]]
    return crop_img, crop_label


def load_file_name_list(file_path):
    file_name_list = []
    with open(file_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline().strip()  # 整行读取数据
            if not lines:
                break
                pass
            file_name_list.append(lines)
            pass
    return file_name_list

def adjust_learning_rate(optimizer, epoch, args):
    """Poly Strategy"""
    power = 2 #power>1是凹函数，即学习率下降的越来越慢
    lr = args.lr * (1 - (epoch / 200)** power) #//表示整除
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr