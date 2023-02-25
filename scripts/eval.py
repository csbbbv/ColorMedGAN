import glob,os
import cv2
import imutils
from skimage.metrics import structural_similarity ,peak_signal_noise_ratio,variation_of_information
import time
from skimage import filters, img_as_ubyte

import numpy as np
import sklearn.metrics as skm
 
 
def hxx_forward(x, y):
    return skm.mutual_info_score(x, y)
 
def hxx(x, y):
    size = x.shape[-1]
    px = np.histogram(x, 256, (0, 255))[0] / size
    py = np.histogram(y, 256, (0, 255))[0] / size
    hx = - np.sum(px * np.log(px + 1e-8))
    hy = - np.sum(py * np.log(py + 1e-8))
 
    hxy = np.histogram2d(x, y, 256, [[0, 255], [0, 255]])[0]
    hxy /= (1.0 * size)
    hxy = - np.sum(hxy * np.log(hxy + 1e-8))
 
    r = hx + hy - hxy
    return r

def ssmi(img1,img2):
    src = cv2.imread(img1)
    img = cv2.imread(img2)
    grayA = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ssmis = structural_similarity(grayA, grayB, win_size=101, full=True)
    return ssmis

def psnr(img1,img2):
    src = cv2.imread(img1)
    img = cv2.imread(img2)
    grayA = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    psnrs = peak_signal_noise_ratio(grayA, grayB)
    return psnrs

def entrophy(img1,img2):
    src = cv2.imread(img1)
    img = cv2.imread(img2)
    grayA = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    entrophys = variation_of_information(grayA, grayB)
    return abs(entrophys[0]-entrophys[1])
def mutul_info(img1,img2):

    img1 = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)
    
    x = np.reshape(img1, -1)
    y = np.reshape(img2, -1)
    
    return hxx_forward(x, y)


# if __name__=='__main__':
exp_name = 'oasis_color_v15_concat_discri_network(upsample)_bidirectG'
img_dir = os.path.join('/home/user/workspace/shaobo/colorMedCycleGAN/results/',exp_name+'_results','test')
ori_list = glob.glob(img_dir+'/ori*.png')
mean_ssmi,mean_psnr,mean_info,mean_var = 0,0,0,0
for ori in ori_list:
    name = ori.split('/')[-1]
    nums = name[3:5]
    if nums[1]=='.':
        nums = nums[0]
    pred_img = img_dir+'/pred'+nums+'.png'
    ssmis,_ = ssmi(ori,pred_img)
    psnrs = psnr(ori,pred_img)
    mutul_infos = mutul_info(ori,pred_img)
    info_vars = entrophy(ori,pred_img)
    mean_ssmi+=ssmis
    mean_psnr+=psnrs
    mean_info+=mutul_infos
    mean_var+=info_vars
    print('num:{0},ssmi:{1},psnr:{2},info:{3},info_vars:{4};'.format(nums,ssmis,psnrs,mutul_infos,info_vars))
total_num = len(ori_list)
mean_info,mean_psnr,mean_ssmi,mean_var = mean_info/total_num,mean_psnr/total_num,mean_ssmi/total_num,mean_var/total_num
print('ssmi:{0},psnr:{1},info:{2},mean_var{3}.'.format(mean_ssmi,mean_psnr,mean_info,mean_var))

# img_dir = 'data/fakecolor/A/train'
# compare_dir = 'data/fakecolor/B/train'
# ori_list = glob.glob(img_dir+'/*.jpg')
# mean_ssmi,mean_psnr,mean_info,mean_var = 0,0,0,0
# nums =0
# for ori in ori_list:
#     name = ori.split('/')[-1]
#     pred_img = compare_dir+'/'+name
#     ssmis,_ = ssmi(ori,pred_img)
#     psnrs = psnr(ori,pred_img)
#     mutul_infos = mutul_info(ori,pred_img)
#     info_vars = entrophy(ori,pred_img)
#     mean_ssmi+=ssmis
#     mean_psnr+=psnrs
#     mean_info+=mutul_infos
#     mean_var+=info_vars
#     print('num:{0},ssmi:{1},psnr:{2},info:{3},info_vars:{4};'.format(nums,ssmis,psnrs,mutul_infos,info_vars))
#     nums += 1
# total_num = len(ori_list)
# mean_info,mean_psnr,mean_ssmi,mean_var = mean_info/total_num,mean_psnr/total_num,mean_ssmi/total_num,mean_var/total_num
# print('ssmi:{0},psnr:{1},info:{2},mean_var{3}.'.format(mean_ssmi,mean_psnr,mean_info,mean_var))

