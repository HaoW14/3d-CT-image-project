import torch.nn.functional as F
from utils.common import *
from scipy import ndimage
import numpy as np
from torchvision import transforms as T
import torch,os
from torch.utils.data import Dataset, DataLoader

class Lits_DataSet(Dataset):
    def __init__(self, crop_size, dataset_path,mode=None):
        self.crop_size = crop_size
        self.dataset_path = dataset_path
        self.class_num = 2

        if mode=='train':
            self.filename_list = load_file_name_list(os.path.join(dataset_path,'train_name_list.txt'))
        elif mode =='val':
            self.filename_list = load_file_name_list(os.path.join(dataset_path, 'val_name_list.txt'))
        else:
            raise TypeError('Dataset mode error!!! ')


    def __getitem__(self, index):
        data, target = self.get_data(crop_size=self.crop_size, filename=self.filename_list[index]) #
        data, target = torch.from_numpy(data), torch.from_numpy(target)

        data= torch.unsqueeze(data, dim=0)  #1 *s * h * w

        #bound = ndimage.distance_transform_edt(target)  # 得到distance map
        #bound = np.trunc(bound)  # 取整
        return data, target

    def __len__(self):
        return len(self.filename_list)

    def get_data(self, crop_size, filename): #读取文件
        data_np = sitk_read_raw(self.dataset_path + '/data/' + filename,)
        #data_np=Z_score_norm3D(data_np)   #归一化
        label_np = sitk_read_raw(self.dataset_path + '/label/' + filename,)  #.replace('volume', 'segmentation')
        #sub_img, sub_label = random_crop_3d(data_np, label_np, crop_size)  # 裁剪
        return data_np, label_np

# 测试代码
def main():
    fixd_path  = '../fixed_data'

    train_set = Lits_DataSet([16, 64, 64],fixd_path,mode='val')
    train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=4)

    for data, mask in train_loader:
        print(data.shape, mask.shape)
if __name__ == '__main__':
    main()
