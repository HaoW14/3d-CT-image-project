import config
import random
from preprocess.myLib import *
from scipy import ndimage
import torchvision.transforms as transforms
np.set_printoptions(threshold=np.inf)

class LITS_fix:
    def __init__(self, raw_dataset_path,fixed_dataset_path):
        self.raw_root_path = raw_dataset_path
        self.fixed_path = fixed_dataset_path

        if not os.path.exists(self.fixed_path):    # 创建保存目录
            os.makedirs(self.fixed_path+'data')
            os.makedirs(self.fixed_path+'label')

        self.fix_data()                            # 对原始图像进行修剪并保存
        self.write_train_val_test_name_list()      # 创建索引txt文件

    def fix_data(self):
        args = config.args
        target_spacing = args.resize_scale
        crop_size = args.crop_size

        print('the raw dataset total numbers of samples is :',len(os.listdir(self.raw_root_path)) )
        for ct_file in os.listdir(self.raw_root_path):
            print(ct_file)
            # 将CT和金标准入读内存
            ct = sitk.ReadImage(os.path.join(self.raw_root_path + ct_file, 'imaging'), sitk.sitkInt16)
            seg = sitk.ReadImage(os.path.join(self.raw_root_path + ct_file, 'segmentation'), sitk.sitkInt8)

            ct_array = sitk.GetArrayFromImage(ct)
            seg_array = sitk.GetArrayFromImage(seg)
            seg_array[seg_array == 2] = 1  # 将3分类变成2分类

            ct_array = ct_array.transpose(2, 1, 0) #表示转置，X轴为0，y轴为1
            seg_array = seg_array.transpose(2, 1, 0)
            #注意下面的aray都要保证已经转成了 slice * h * w
            ct_array, seg_array = spacing_interpolation(ct_array, seg_array, ct.GetSpacing(), target_spacing)

            ct_array = window_transform(ct_array, seg_array, labelindex=1)

            num = sliding_window_crop(ct_array, seg_array, crop_size, self.fixed_path, ct_file)

            print(num)

            '''
            ctSpace =ct.GetSpacing()

            scale_vector = (
                ctSpace[0] / target_resolution[0],
                ctSpace[1] / target_resolution[1], 
                ctSpace[2] / target_resolution[2], )
            ct_array = ndimage.zoom(ct_array, scale_vector, order=3)
            seg_array = ndimage.zoom(seg_array, scale_vector, order=0)  #seg一定得用order=0，近邻差值

            ct_array = (ct_array - 101) / 76.9   #归一化
            print(ct_array.shape)
            
            print(ct_array.shape)


            # 数组转成图片，兵保存
            new_ct = sitk.GetImageFromArray(ct_array)
            new_seg = sitk.GetImageFromArray(seg_array)
            sitk.WriteImage(new_ct, os.path.join(self.fixed_path + 'data/', ct_file.replace('case', 'volume') +'.nii'))
            sitk.WriteImage(new_seg,os.path.join(self.fixed_path + 'label/', ct_file.replace('case', 'segmentation') + '.nii'))
            '''


    def write_train_val_test_name_list(self):
        data_name_list = os.listdir(self.fixed_path + "/" + "data")
        data_num = len(data_name_list)
        print('the fixed dataset total numbers of samples is :', data_num)
        random.shuffle(data_name_list)  #洗牌

        train_rate = 0.8
        val_rate = 0.2

        assert val_rate+train_rate == 1.0
        train_name_list = data_name_list[0:int(data_num*train_rate)]
        val_name_list = data_name_list[int(data_num*train_rate):int(data_num*(train_rate + val_rate))]

        self.write_name_list(train_name_list, "train_name_list.txt")
        self.write_name_list(val_name_list, "val_name_list.txt")

    def write_name_list(self, name_list, file_name):
        f = open(self.fixed_path + file_name, 'w')
        for i in range(len(name_list)):
            f.write(str(name_list[i]) + "\n")
        f.close()

    def ImageResample(self,sitk_image, is_label = False):
        '''
        sitk_image:
        new_spacing: x,y,z
        is_label: if True, using Interpolator `sitk.sitkNearestNeighbor`
        '''
        new_spacing = [ 3.22, 1.62, 1.62]
        size = np.array(sitk_image.GetSize())
        spacing = np.array(sitk_image.GetSpacing())
        new_space = np.array(new_spacing)
        new_size = size * spacing / new_space
        new_spacing_refine = size * spacing / new_size
        new_spacing_refine = [float(s) for s in new_spacing_refine]
        new_size = [int(s) for s in new_size]

        resample = sitk.ResampleImageFilter()
        resample.SetOutputDirection(sitk_image.GetDirection())
        resample.SetOutputOrigin(sitk_image.GetOrigin())
        resample.SetSize(new_size)
        resample.SetOutputSpacing(new_spacing_refine)

        if is_label:
            resample.SetInterpolator(sitk.sitkNearestNeighbor)
            #sitk_image = sitk.BinaryFillhole(sitk_image)   #去除孔洞
        else:
            resample.SetInterpolator(sitk.sitkLinear)

        newimage = resample.Execute(sitk_image)
        return newimage


def main():
    raw_dataset_path = '../row_data/'
    fixed_dataset_path = '../fixed_data/'
    
    LITS_fix(raw_dataset_path,fixed_dataset_path)

if __name__ == '__main__':
    main()
