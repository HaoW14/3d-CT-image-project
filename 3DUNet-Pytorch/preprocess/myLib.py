import os
import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage
import warnings
warnings.simplefilter("ignore")

#调整窗宽，除去过大过小的像素
def window_transform(ct_array, seg_array, labelindex = 1):
    """
    return: trucated image according to window center and window width
    and Z-score norm
    """
    seg_object = seg_array >= labelindex
    ct_object = ct_array * seg_object

    ct_max,ct_min = ct_object.max(),ct_object.min()
    windowCenter,windowWidth = (ct_max + ct_min)/2, ct_max - ct_min

    minWindow = float(windowCenter) - 0.5 * float(windowWidth)
    maxWindow = float(windowCenter) + 0.5 * float(windowWidth)
    ct_array[ct_array < minWindow] = minWindow
    ct_array[ct_array > maxWindow] = maxWindow

    ct_mean = np.mean(ct_array)
    ct_std = np.std(ct_array)
    ct_array = (ct_array - ct_mean) / ct_std
    return ct_array
#统一space
def spacing_interpolation(ct_array,seg_array,original_spacing,target_spacing):  # 注意: ct_array: slice * h * w
    ct_array = ndimage.zoom(ct_array, (original_spacing[0] / target_spacing[0],
                                       original_spacing[1] / target_spacing[1],
                                        original_spacing[2] / target_spacing[2]),order=3)
    # 对金标准插值不应该使用高级插值方式，这样会破坏边界部分,检查数据输出很重要！！！
    # 使用order=1可确保zoomed seg unique = [0,1,2]
    seg_array = ndimage.zoom(seg_array, (original_spacing[0] / target_spacing[0],
                                       original_spacing[1] / target_spacing[1],
                                        original_spacing[2] / target_spacing[2]), order=0)
    #print('zoomed seg unique:', np.unique(seg_array))  #可以检查seg插值对不对

    return ct_array, seg_array
#取存在目标的slice
def get_mask_effective(ct_array, seg_array,labelindex = 1):
    seg_array = seg_array >= labelindex
    z = np.any(seg_array, axis=(1, 2))  # z表示存在标签为1的slice   np.all是做与操作，判断是否全为1
    start_slice, end_slice = np.where(z)[0][[0, -1]]  # 返回第一个跟最后一个
    ct_array = ct_array[max(0, start_slice - 5):min(seg_array.shape[0], end_slice + 5), :, :]  # 去掉了两边没有肾脏的部分
    seg_array = seg_array[max(0, start_slice - 5):min(seg_array.shape[0], end_slice + 5), :, :]
    return ct_array, seg_array
#滑动窗口裁剪
def sliding_window_crop(ct_array,seg_array,crop_size,saved_path,original_volname):
    """
    get croped images and save
    :param saved_idx:
    :param origin:
    :param direction:
    :param xyz_thickness:
    :param savedct_path: 保存的ct目录路径
    :param savedseg_path:  保存的seg目录路径
    :param original_volname: 原图像名（便于给裁剪后的图像命名）
    :param original_segname: 原图像名（便于给裁剪后的图像命名）
    :return:  the number of croped_image
    """

    Stride = [ ele * 2 // 3 for ele in crop_size] #步长为裁剪图像的2/3
    num_z = (ct_array.shape[0]-crop_size[0])//Stride[0] + 1 #当前维度可以根据步长分成的份数
    num_x = (ct_array.shape[1]-crop_size[1])//Stride[1] + 1
    num_y = (ct_array.shape[2]-crop_size[2])//Stride[2] + 1
    idx = 0
    for i in range(len(ct_array.shape)):
        if (ct_array.shape[i]-crop_size[i]) < 0:  #存在一个维度小于裁剪的大小，就直接输出0
            return idx
    savedct_path = os.path.join(saved_path,'data')  # 原目录进去一层，分成 data跟label两个目录
    savedseg_path = os.path.join(saved_path,'label')

    '''
    if os.path.exists(savedct_path)|os.path.exists(savedseg_path):
        shutil.rmtree(savedct_path) #递归地删除文件
        shutil.rmtree(savedseg_path)
    os.mkdir(savedct_path)  #创建目录
    os.mkdir(savedseg_path)
    '''

    for z in range(num_z):
        for x in range(num_x):
            for y in range(num_y):
                seg_block = seg_array[z*Stride[0]:z*Stride[0]+crop_size[0],
                            x*Stride[1]:x*Stride[1]+crop_size[1],y*Stride[2]:y*Stride[2]+crop_size[2]]
                ct_block = ct_array[z * Stride[0]:z * Stride[0] + crop_size[0],
                           x * Stride[1]:x * Stride[1] + crop_size[1], y * Stride[2]:y * Stride[2] + crop_size[2]]
                if not seg_block.any():      #如果全是0，就不要这一张裁剪的图
                    continue
                saved_ctlocation = os.path.join(savedct_path, original_volname + '_'+str(idx) +'.nii')
                saved_seglocation = os.path.join(savedseg_path, original_volname + '_'+str(idx)+'.nii') #刻意为之，让两个目录下的文件名是相同的

                newct = sitk.GetImageFromArray(ct_block)
                newseg = sitk.GetImageFromArray(seg_block)
                sitk.WriteImage(newct, saved_ctlocation)
                sitk.WriteImage(newseg, saved_seglocation)
                idx = idx + 1
    return  idx
