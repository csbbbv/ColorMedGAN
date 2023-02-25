import numpy as np
import os  # 遍历文件夹
import nibabel as nib  # nii格式一般都会用到这个包
import imageio  # 转换成图像
from nibabel.viewers import OrthoSlicer3D
import cv2
from PIL import Image
def nii_to_image(filepath, imgfile):
    filenames = os.listdir(filepath)  # 读取nii文件夹
    slice_trans = []

    for f in filenames:
        # 开始读取nii文件
        img_path = os.path.join(filepath, f)
        img = nib.load(img_path)  # 读取nii
        img_fdata = img.get_fdata()
        fname = f.replace('.nii', '')  # 去掉nii的后缀名
        img_f_path = os.path.join(imgfile, fname)
        # 创建nii对应的图像的文件夹
        if not os.path.exists(img_f_path):
            os.mkdir(img_f_path)  # 新建文件夹

        # 开始转换为图像
        (x, y, z,_) = img.shape
        for i in range(z):  # z是图像的序列
            silce = img_fdata[i,:, :,:3]  # 选择哪个方向的切片都可以
            imageio.imwrite(os.path.join(img_f_path, '{}.png'.format(i)), silce)
            # 保存图像

def nii_to_image_from_files(filepath):
    file_lists = os.listdir(filepath)  # 读取nii文件夹
    for file in file_lists:
        if os.path.isdir(os.path.join(filepath,file)):
            filename = file.split('/')[-1]
            nii_path = os.path.join(filepath,file,'orig.nii.gz')
            label4_path = os.path.join(filepath,file,'seg4.nii.gz')
            label35_path = os.path.join(filepath,file,'seg35.nii.gz')
            img = nib.load(nii_path)
            # label4 = nib.load(label4_path)
            # label35 = nib.load(label35_path)
            img_fdata = img.get_fdata()
            # label4_fdata = label4.get_fdata()
            # label35_fdata = label35.get_fdata()
            (x,y,z) = img.shape
            for i in range(z):  # z是图像的序列
                if i >= 70 and i<= 190:
                    silce = img_fdata[:,i, :]*255  # 选择哪个方向的切片都可以
                    # imageio.save(os.path.join('/media/user/8f28289d-aa45-46ab-b9e4-a7d4d9f08b7e/shaobo/oasis1/norm', '{}_{}.png'.format(filename,i)), silce)
                    cv2.imwrite(os.path.join('/media/agent001/8f28289d-aa45-46ab-b9e4-a7d4d9f08b7e/shaobo/oasis1/orig', '{}_{}.png'.format(filename,i)), silce)
                    # silce = label4_fdata[:,i, :]  # 选择哪个方向的切片都可以
                    # cv2.imwrite(os.path.join('/media/user/8f28289d-aa45-46ab-b9e4-a7d4d9f08b7e/shaobo//oasis1/label4', '{}_{}.png'.format(filename,i)), silce)
                    # silce = label35_fdata[:,i, :]  # 选择哪个方向的切片都可以
                    # cv2.imwrite(os.path.join('/media/user/8f28289d-aa45-46ab-b9e4-a7d4d9f08b7e/shaobo//oasis1/label35', '{}_{}.png'.format(filename,i)), silce)


def rotate(img,savepath):
    im_name = img.split('/')[-1]
    im = Image.open(img)
    im_rotate = im.rotate(-90)
    im_rotate.save(os.path.join(savepath,im_name))

def resize(img,savepath):
    im_name = img.split('/')[-1]
    im = Image.open(img)
    im_rotate = im.resize((128,128))
    im_rotate.save(os.path.join(savepath,im_name))

if __name__ == '__main__':
    # f = './img'  # 輸入路徑
    # i = './img'  # 輸出路徑
    # nii_to_image(f, i)


    # nii_to_image_from_files('/media/agent001/8f28289d-aa45-46ab-b9e4-a7d4d9f08b7e/shaobo/neurite-oasis.v1.0/')


    # example_filename = r'/media/agent001/8f28289d-aa45-46ab-b9e4-a7d4d9f08b7e/shaobo/neurite-oasis.v1.0/OASIS_OAS1_0001_MR1/norm.nii.gz'
    # img = nib.load(example_filename)
    # img_fdata = img.get_fdata()
    # # OrthoSlicer3D(img.dataobj).show()
    # (x,y,z) = img.shape
    # for i in range(z):  # z是图像的序列
    #     if i >= 70 and i<= 190:
    #         silce = img_fdata[:,i, :]  # 选择哪个方向的切片都可以
    #         imageio.imwrite(os.path.join('data/test_nii', '{}_norm.png'.format(i)), silce)


    img_dir = "/media/agent001/8f28289d-aa45-46ab-b9e4-a7d4d9f08b7e/shaobo/visibleKoreanRmbgBrain/"
    savepath = "/media/agent001/8f28289d-aa45-46ab-b9e4-a7d4d9f08b7e/shaobo/visibleKoreanRmbgBrain/"
    import glob
    img_list = glob.glob(img_dir+'/*.png')
    cnt = 0
    for img in img_list:
        rotate(img,savepath)
        # resize(img,savepath)
        print('rotated saved {0}'.format(cnt))
        cnt+=1

    # import cv2
    # img = cv2.imread("/home/user/workspace/shaobo/colorCycleGAN/data/oasis1/label35/OASIS_OAS1_0001_MR1_175.png",0)
    # lis = np.unique(img)
    # print('1')