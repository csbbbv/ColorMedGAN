import cv2
import numpy as np
from numpy.core.fromnumeric import repeat
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class EdgeLoss:
        def __init__(self,size,batch):
            self.size = size
            self.batch = batch
        #     self.img2 = img2
        def toEdge(self,label_arr):
                # label_arr = np.squeeze(np.transpose(label_arr,(2,3,1,0)))
                
                # edge_arr = cv2.Canny(label_arr, 0, 255)
                edge_arr = cv2.Sobel(label_arr,cv2.CV_8U , 1, 1)
                return edge_arr
        def functional_conv2d(self,im):
                # sobel_kernels = np.zeros((1,3,3,3))
                
                # conv_op = nn.Conv2d(1, 1, 3, bias=False)
                # nn.Conv2d()
                # 定义sobel算子参数
                im = im.reshape(self.batch,1,self.size,self.size)
                edge_detect = torch.FloatTensor(self.batch,1, self.size,self.size)
                sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
                # 将sobel算子转换为适配卷积操作的卷积核
                sobel_kernel = sobel_kernel.reshape((1,1, 3, 3))
                # new_img = 
                # sobel_kernel = np.expand_dims(sobel_kernel,0).repeat(4,axis=0)
                # 给卷积操作的卷积核赋值
                weight = Variable(torch.from_numpy(sobel_kernel)).cuda()
                # 对图像进行卷积操作
                edge_detect= F.conv2d(Variable(im),weight,padding=1)
                # edge_detect[1] = conv_op(Variable(im[1]))
                # edge_detect[2] = conv_op(Variable(im[2]))
                # edge_detect[3] = conv_op(Variable(im[3]))
                # 将输出转换为图片格式
                edge_detect = edge_detect.squeeze().detach()#.cpu().numpy()
                return edge_detect


        # def create3DsobelFilter(self):
        #         num_1, num_2, num_3 = np.zeros((3,3))
        #         num_1 = [[1., 2., 1.],
        #                 [2., 4., 2.],
        #                 [1., 2., 1.]]
        #         num_2 = [[0., 0., 0.],
        #                 [0., 0., 0.],
        #                 [0., 0., 0.]]
        #         num_3 = [[-1., -2., -1.],
        #                 [-2., -4., -2.],
        #                 [-1., -2., -1.]]
        #         # sobelFilter = np.zeros((3,1,3,3,3))
        #         sobelFilter = np.zeros((1,3,3,3))
        #         sobelFilter[0,0,:,:] = num_1
        #         sobelFilter[0,1,:,:] = num_2
        #         sobelFilter[0,2,:,:] = num_3
        #         # sobelFilter[1,0,:,0,:] = num_1
        #         # sobelFilter[1,0,:,1,:] = num_2
        #         # sobelFilter[1,0,:,2,:] = num_3
        #         # sobelFilter[2,0,:,:,0] = num_1
        #         # sobelFilter[2,0,:,:,1] = num_2
        #         # sobelFilter[2,0,:,:,2] = num_3
        #         return Variable(torch.from_numpy(sobelFilter).type(torch.cuda.FloatTensor))
        # def functional_conv3d(self,input):
        #         # pad = nn.ConstantPad3d((1,1,1,1,1,1),-1)
        #         kernel = self.create3DsobelFilter()
        #         # act = nn.Tanh()
        #         # paded = pad(input)
        #         # fake_sobel = F.conv3d(paded, kernel, padding = 0, groups = 1)/4
        #         edge_detect = F.conv3d(input,kernel)
        #         # n,c,h,w,l = fake_sobel.size()
        #         fake = torch.norm(fake_sobel,2,1,True)/c*3
        #         fake_out = act(fake)*2-1
        #         return fake_out


        def ssim(self,img1, img2):
                C1 = (0.01 * 255) ** 2
                C2 = (0.03 * 255) ** 2
                img1 = img1.astype(np.float64)
                img2 = img2.astype(np.float64)
                kernel = cv2.getGaussianKernel(11, 1.5)
                window = np.outer(kernel, kernel.transpose())
                mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
                mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
                mu1_sq = mu1 ** 2
                mu2_sq = mu2 ** 2
                mu1_mu2 = mu1 * mu2
                sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
                sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
                sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
                ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                        (sigma1_sq + sigma2_sq + C2))
                return ssim_map.mean()
 
 
        def calculate_ssim(self):
                '''calculate SSIM
                the same outputs as MATLAB's
                img1, img2: [0, 255]
                '''
                img1 = self.toEdge(self.img1)
                img2 = self.toEdge(self.img2)

                if not img1.shape == img2.shape:
                        raise ValueError('Input images must have the same dimensions.')
                if img1.ndim == 2:
                        return self.ssim(img1, img2)
                elif img1.ndim == 3:
                        if img1.shape[2] == 3:
                                ssims = []
                                for i in range(3):
                                        ssims.append(self.ssim(img1, img2))
                                return np.array(ssims).mean()
                        elif img1.shape[2] == 1:
                                return 1 - self.ssim(np.squeeze(img1), np.squeeze(img2))
                else:
                        raise ValueError('Wrong input image dimensions.')
        
        # def cal_lap_mse(self):


def Attention_edge(l_edge,img,size,batch_size):
        l_edge_new = l_edge.reshape(batch_size,1,size,size)
        img[:,0,:,:] += img[:,0,:,:] * l_edge
