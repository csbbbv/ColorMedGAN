'''
修改框架，让判别器判别VGG风格和边缘信息
'''


import argparse
import os
import time
import matplotlib.pyplot as plt
import pickle
from torchvision import transforms, models
import torch
import itertools
import numpy as np
import torch.nn as nn
import utils
import glob
import network as net
import edge_loss as eloss
# from util import tensor2im
from collections import OrderedDict
from torchvision.utils import save_image
from model import VGGEncoder, Decoder
from style_swap import style_swap
import cv2
from PIL import Image
from phash import phash,hmdistance
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--name', required=False, default='oasis_color_v15_concat_discri_network(upsample)_bidirectG',  help='project name')
parser.add_argument('--src_data', required=False, default='oasisMRI/A',  help='src data path')
parser.add_argument('--tgt_data', required=False, default='oasisMRI/B',  help='tgt data path')
parser.add_argument('--in_ngc', type=int, default=3, help='input channel for generator')
parser.add_argument('--out_ngc', type=int, default=3, help='output channel for generator')
parser.add_argument('--in_ndc', type=int, default=4, help='input channel for discriminator')
parser.add_argument('--out_ndc', type=int, default=1, help='output channel for discriminator')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=32)
parser.add_argument('--input_size', type=int, default=256, help='input size')
parser.add_argument('--train_epoch', type=int, default=8)
parser.add_argument('--lrD', type=float, default=0.0001, help='learning rate, default=0.0002')
parser.add_argument('--lrG', type=float, default=0.0001, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
parser.add_argument('--if_edge_loss', type=bool, default=True, help='use edge loss or not ')
parser.add_argument('--if_color_loss', type=bool, default=True, help='use color loss or not ')
parser.add_argument('--extra description', type=str, default=True, help='with seg feature,with edge loss')
parser.add_argument('--model_state_path', type=str, default='model_state.pth',
                        help='save directory for result and loss')
parser.add_argument('--patch_size', '-p', type=int, default=3,
                        help='Size of extracted patches from style features')
parser.add_argument('--style', '-s', type=str, default='style/OASIS_OAS1_0152_MR1_101.png',
                help='Style image path e.g. image.jpg')
args = parser.parse_args()

print('------------ Options -------------')
for k, v in sorted(vars(args).items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')



# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# results save path
if not os.path.isdir(os.path.join('results',args.name + '_results', 'Colorization')):
    os.makedirs(os.path.join('results',args.name + '_results', 'Colorization'))


argsDict = args.__dict__
with open(os.path.join('results',args.name + '_results','setting.txt'), 'w') as f:
    f.writelines('------------------ start ------------------' + '\n')
    for eachArg, value in argsDict.items():
        f.writelines(eachArg + ' : ' + str(value) + '\n')
    f.writelines('------------------- end -------------------')


transform = transforms.Compose([
         transforms.Resize((args.input_size, args.input_size)),
        #  transforms.CenterCrop(128),
        #  transforms.RandomHorizontalFlip(p=0.5),
        #  transforms.RandomVerticalFlip(p=0.5),
        #  transforms.RandomRotation(180, resample=False, expand=False, center=None),
        #  transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
        #  transforms.RandomErasing(),
         transforms.ToTensor(),
         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
 ])

train_loader_src = utils.data_load(os.path.join('/media/user/8f28289d-aa45-46ab-b9e4-a7d4d9f08b7e/shaobo/colorization_tmp_oasis/data/', args.src_data), 'train', transform, args.batch_size, shuffle=True, drop_last=True)
train_loader_tgt = utils.data_load(os.path.join('/media/user/8f28289d-aa45-46ab-b9e4-a7d4d9f08b7e/shaobo/colorization_tmp_oasis/data/', args.tgt_data), 'train', transform, args.batch_size, shuffle=True, drop_last=True)
test_loader_src = utils.data_load(os.path.join('/media/user/8f28289d-aa45-46ab-b9e4-a7d4d9f08b7e/shaobo/colorization_tmp_oasis/data/', args.src_data), 'test', transform, 1, shuffle=True, drop_last=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



#################################  style transfer   ############################################
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

trans = transforms.Compose([transforms.ToTensor(),
                            normalize])

def denorm(tensor, device):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res

def save_image_tensor2cv2(input_tensor, filename):
    """
    将tensor保存为cv2格式
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.copy()
    # 到cpu
    input_tensor = input_tensor
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    # 去掉批次维度
    input_tensor = input_tensor.squeeze()
    # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为cv2
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    # RGB转BRG
    input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, input_tensor)


def image_tensor2cv2(input_tensor):
    """
    将tensor保存为cv2格式
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone()
    # 到cpu
    input_tensor = input_tensor.detach()
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    # 去掉批次维度
    input_tensor = input_tensor.squeeze()
    # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为cv2
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).cpu().detach().numpy()
    # RGB转BRG
    input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2GRAY)
    # cv2.imwrite(filename, input_tensor)
    return input_tensor

def save_image_tensor2cv2_gray(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为cv2格式
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    # assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = torch.tensor(input_tensor).cuda()
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    # 去掉批次维度
    # input_tensor = input_tensor.squeeze()
    # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为cv2
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).type(torch.uint8).numpy()
    # RGB转BRG
    # input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, input_tensor)


def image_tensor2cv2_gray(input_tensor: torch.Tensor):
    """
    将tensor保存为cv2格式
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    # assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = torch.tensor(input_tensor).cuda()
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    # 去掉批次维度
    # input_tensor = input_tensor.squeeze()
    # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为cv2
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).type(torch.uint8).numpy()
    # input_tensor = np.resize(input_tensor[0][0],(128,128))
    # RGB转BRG
    # input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(filename, input_tensor)
    return input_tensor
# set model
e = VGGEncoder().to(device)
d = Decoder()
d.load_state_dict(torch.load(args.model_state_path))
d = d.to(device)


################################################################################################

#############################segmentation branch###############################
seg_net = net.Baseline(num_classes=5)
state_dict = torch.load("segmentation/checkpoint/oasis_v1_orig/unet-dice_-2021-07-02 22:41:57/30.pkl")
new_state_dict = OrderedDict()
# new_state_dict = {}
for k, v in state_dict.items(): # k为module.xxx.weight, v为权重
    name = k[7:] # 截取`module.`后面的xxx.weight
    new_state_dict[name] = v
seg_net.load_state_dict(new_state_dict)
seg_net.cuda()
seg_net.eval()
###############################################################################
##############  create img hash ####################################
conference_code,conference_path = [],[]

C_list = os.listdir('./conference_images')
for path in C_list:
    img = Image.open(os.path.join('./conference_images',path))
    img = img.convert('RGB')
    img = transform(img)
    img = img.detach().to(torch.device('cpu')).squeeze().permute(1, 2, 0).type(torch.uint8).numpy()

    hash_code = phash(img)
    conference_code.append(hash_code)
    conference_path.append(path)
print("phash dict created!")

###############################################################################

# A2BG = net.generator(args.in_ngc, args.out_ngc, args.ngf)
# A2BG = net.generator_seg_branch(args.in_ngc, args.out_ngc, args.ngf)

A2BG = net.generator_seg_attention_upsample(args.in_ngc, args.out_ngc, args.ngf)

B2AG = net.generator_seg_attention_upsample(args.in_ngc, args.out_ngc, args.ngf)

AD = net.discriminator(args.in_ndc, args.out_ndc, args.ndf)
BD = net.discriminator(args.in_ndc, args.out_ndc, args.ndf)

print('---------- Networks initialized -------------')
utils.print_network(A2BG)
utils.print_network(AD)
print('-----------------------------------------------')

vgg16 = models.vgg16(pretrained=True)
vgg16 = net.VGG(vgg16.features[:23]).to(device)

A2BG = nn.DataParallel(A2BG)
B2AG = nn.DataParallel(B2AG)
AD = nn.DataParallel(AD)
BD = nn.DataParallel(BD)

A2BG.to(device)
B2AG.to(device)
AD.to(device)
BD.to(device)

A2BG.train()
B2AG.train()
AD.train()
BD.train()

# loss
criterion_GAN = nn.MSELoss().to(device)
criterion_cycle = nn.L1Loss().to(device)
criterion_identity = nn.L1Loss().to(device)
criterion_color = nn.L1Loss().to(device)
criterion_edge = nn.L1Loss().to(device)
# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(filter(lambda p: p.requires_grad,A2BG.parameters()), filter(lambda p: p.requires_grad,B2AG.parameters())),lr=args.lrG, betas=(args.beta1, args.beta2))
optimizer_D_A = torch.optim.Adam(filter(lambda p: p.requires_grad,AD.parameters()), lr=args.lrD, betas=(args.beta1, args.beta2))
optimizer_D_B = torch.optim.Adam(filter(lambda p: p.requires_grad,BD.parameters()), lr=args.lrD, betas=(args.beta1, args.beta2))

lr_scheduler_G = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer_G, milestones=[args.train_epoch // 2, args.train_epoch // 4 * 3], gamma=0.1)
lr_scheduler_D_A = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer_D_A, milestones=[args.train_epoch // 2, args.train_epoch // 4 * 3], gamma=0.1)
lr_scheduler_D_B = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer_D_B, milestones=[args.train_epoch // 2, args.train_epoch // 4 * 3], gamma=0.1)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor

target_real = torch.ones(args.batch_size, 1, args.input_size // 4, args.input_size // 4).to(device)
target_fake = torch.zeros(args.batch_size, 1, args.input_size // 4, args.input_size // 4).to(device)

fake_A_buffer = utils.ReplayBuffer()
fake_B_buffer = utils.ReplayBuffer()

torch.backends.cudnn.benchmark = True
train_hist = {}
train_hist['G_loss'] = []
train_hist['G_identity_loss'] = []
train_hist['G_GAN_loss'] = []
train_hist['G_cycle_loss'] = []
train_hist['G_Color_loss'] = []
train_hist['D_loss'] = []
train_hist['per_epoch_time'] = []
train_hist['total_time'] = []

print('training start!')
start_time = time.time()

for epoch in range(args.train_epoch):
    epoch_start_time = time.time()
    G_losses = []
    G_identity_losses = []
    G_GAN_losses = []
    G_cycle_losses = []
    G_Color_losses = []
    D_losses = []
    EdgeLossAB = []
    EdgeLossBA = []

    for (x, _), (y, _)in zip(train_loader_src, train_loader_tgt):
        x, y= x.to(device), y.to(device)


        '''
        seg_net = Baseline(num_classes=5)
        state_dict = torch.load("segmentation/checkpoint/oasis_v1_orig/unet-dice_-1625227792.6709714.pkl")
        new_state_dict = OrderedDict()
        # new_state_dict = {}
        for k, v in state_dict.items(): # k为module.xxx.weight, v为权重
            name = k[7:] # 截取`module.`后面的xxx.weight
            new_state_dict[name] = v
        # load params
        # net = XXXnet()
        # if isinstance(state_dict, torch.nn.DataParallel):
        #        state_dict = state_dict.module
        # state_dict.load_state_dict(torch.load("/home/user/workspace/shaobo/colorCycleGAN/segmentation/checkpoint/oasis_v1_orig/unet-dice_-1625045476.347268.pth"))
        self.seg_net.load_state_dict(new_state_dict)
        self.seg_net.cuda()
        self.seg_net.eval()
        seg_ft1,seg_ft2 = self.seg_net(input_seg)
        '''


        # Set model input
        real_A = x
        real_B = y
        RA_Gray = real_A[:, 0, :, :].view(args.batch_size, 1, args.input_size, args.input_size)
        # RB_Gray = utils.Gray(real_B, args.input_size, args.batch_size)

        '''
        styleSwap for RGB image to MR image
        '''
        RB_Gray = utils.Gray(real_B, args.input_size, args.batch_size)
        rb_gray_hash = real_B.clone().detach().to(torch.device('cpu')).squeeze().permute(1, 2, 0).type(torch.uint8).numpy()
        hash_rb_real = phash(rb_gray_hash)
        min_code = 10000
        idx = -1
        for i in range(len(conference_code)):
            pcode = hmdistance(conference_code[i],hash_rb_real)
            if pcode < min_code:
                idx = i
                min_code = pcode
        style_img = conference_path[idx]
        s = Image.open(os.path.join('./conference_images',style_img)).convert('RGB')
        # s = Image.open(args.style).convert('RGB')
        s = transform(s)
        s = s.detach().to(torch.device('cpu')).squeeze().permute(1, 2, 0).type(torch.uint8).numpy()

        s_tensor = trans(s).unsqueeze(0).to(device) 
        with torch.no_grad():
            cf = e(real_B)
            sf = e(s_tensor)
            style_swap_res = style_swap(cf, sf, args.patch_size, 1)
            out = d(style_swap_res)
        out_denorm = denorm(out, device)
        RB_Gray = utils.Gray(out_denorm, args.input_size, args.batch_size)
        


        
        Aseg_ft1,Aseg_ft2 = seg_net(RA_Gray)
        Bseg_ft1,Bseg_ft2 = seg_net(RB_Gray)
        ra_gray = utils.Gray(real_A,args.input_size, args.batch_size)
        rb_gray = utils.Gray(real_B,args.input_size, args.batch_size)
        # for i in range(128):
        #     Length = len(os.listdir('seg_featuremap/'))
        #     if Length < 12800:
        #         save_image_tensor2cv2_gray(np.concatenate((Aseg_ft1[0][i].cpu().detach().numpy(),cv2.resize(image_tensor2cv2_gray(RA_Gray)[0].reshape(256,256),(128,128))),axis=1),'seg_featuremap/Aseg_ft1_{0}.jpg'.format(str(i)))
        #         save_image_tensor2cv2_gray(np.concatenate((Aseg_ft2[0][i].cpu().detach().numpy(),cv2.resize(image_tensor2cv2_gray(RA_Gray)[0].reshape(256,256),(128,128))),axis=1),'seg_featuremap/Aseg_ft2_{0}.jpg'.format(str(i)))
        #         save_image_tensor2cv2_gray(np.concatenate((Bseg_ft1[0][i].cpu().detach().numpy(),cv2.resize(image_tensor2cv2(real_B).reshape(256,256),(128,128))),axis=1),'seg_featuremap/Bseg_fea1_{0}.jpg'.format(str(i)))
        #         save_image_tensor2cv2_gray(np.concatenate((Bseg_ft2[0][i].cpu().detach().numpy(),cv2.resize(image_tensor2cv2(real_B).reshape(256,256),(128,128))),axis=1),"seg_featuremap/Bseg_ft2_{0}.jpg".format(str(i)))
        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()
        # Identity loss

        #gray
        RB_Gray = utils.Gray(real_B, args.input_size, args.batch_size)


        # G_A2B(B) should equal B if real B is fed
        same_B = A2BG(real_B,Bseg_ft1,Bseg_ft2)
        real_B_feature=vgg16(real_B)
        same_B_feature=vgg16(same_B)
        loss_identity_B = criterion_identity(same_B_feature[2], real_B_feature[2]) * 5.0
        # loss_identity_B = criterion_identity(same_B, real_B) * 5.0


        # G_B2A(A) should equal A if real A is fed
        same_A = B2AG(real_A,Bseg_ft1,Bseg_ft2)
        real_A_feature=vgg16(real_A)
        same_A_feature=vgg16(same_A)
        loss_identity_A = criterion_identity(same_A_feature[2], real_A_feature[2]) * 5.0
        # loss_identity_A = criterion_identity(same_A, real_A) * 5.0

        loss_G_identity = loss_identity_A + loss_identity_B
        G_identity_losses.append(loss_G_identity.item())
        train_hist['G_identity_loss'].append(loss_G_identity.item())

        ###################################
        # GAN loss
        #A2B
        fake_B = A2BG(real_A,Aseg_ft1,Aseg_ft2)
        FB_Gray = utils.Gray(fake_B,args.input_size,args.batch_size)
        Edge = eloss.EdgeLoss(args.input_size,args.batch_size)
        lp_fakeB = Edge.functional_conv2d(FB_Gray)
        edge_feature_fb = lp_fakeB.reshape(args.batch_size,1,256,256)
        S_I_feature_fb = torch.cat([fake_B,edge_feature_fb],dim=1)
        # S_I_feature_fb = fake_B + torch.sigmoid(edge_feature_fb) * fake_B
        # pred_fake = BD(S_I_feature_fb)
        pred_fake = BD(S_I_feature_fb)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

        # color loss
        # RA_Gray=utils.Gray(real_A,args.input_size,args.batch_size)
        RA_Gray = real_A[:, 0, :, :]
        
        # with torch.no_grad():
        #     cf = e(fake_B)
        #     sf = e(s_tensor)
        #     style_swap_res = style_swap(cf, sf, args.patch_size, 1)
        #     out = d(style_swap_res)
        # out_denorm = denorm(out, device)
        # FB_Gray = utils.Gray(out_denorm, args.input_size, args.batch_size)
        color_loss_A2B = criterion_color(FB_Gray,RA_Gray) * 10.0



        #B2A
        fake_A = B2AG(real_B,Bseg_ft1,Bseg_ft2)
        FA_Gray = fake_A[:, 0, :, :]
        lp_fakeA = Edge.functional_conv2d(FA_Gray)
        edge_feature_fa = lp_fakeA.reshape(args.batch_size,1,256,256)
        S_I_feature_fa = torch.cat([fake_A,edge_feature_fa],dim=1)
        # S_I_feature_fa = fake_A + fake_A * torch.sigmoid(edge_feature_fa)
        # pred_fake = AD(S_I_feature_fa)
        pred_fake = AD(S_I_feature_fa)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

        # color loss
        
        # FA_Gray = utils.Gray(fake_A, args.input_size, args.batch_size)

        # with torch.no_grad():
        #     cf = e(real_B)
        #     sf = e(s_tensor)
        #     style_swap_res = style_swap(cf, sf, args.patch_size, 1)
        #     out = d(style_swap_res)
        # out_denorm = denorm(out, device)
        # RB_Gray = utils.Gray(out_denorm, args.input_size, args.batch_size)

        
        color_loss_B2A = criterion_color(FA_Gray, RB_Gray) * 10.0

        loss_G_GAN = loss_GAN_A2B + loss_GAN_B2A
        G_GAN_losses.append(loss_G_GAN.item())
        train_hist['G_GAN_loss'].append(loss_G_GAN.item())

        loss_Color = color_loss_B2A + color_loss_A2B
        G_Color_losses.append(loss_Color.item())
        train_hist['G_Color_loss'].append(loss_Color.item())
            ################edge loss###################
        
        

        lp_realA = Edge.functional_conv2d(RA_Gray)
        lp_realB = Edge.functional_conv2d(RB_Gray)
        elossAB = criterion_edge(lp_fakeB,lp_realA) * 10.0
        elossBA = criterion_edge(lp_fakeA,lp_realB) * 10.0
        EdgeLossAB.append(elossAB)
        EdgeLossBA.append(elossBA)
        elosses = elossAB + elossBA
        # elosses *= 0.2
       
        #################################################

        # Cycle loss
        recovered_A = B2AG(fake_B,Bseg_ft1,Bseg_ft2)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0

        FA_seg_ft1,FA_seg_ft2 = seg_net(FA_Gray.view(args.batch_size, 1, args.input_size, args.input_size))
        # for i in range(128):
        #     Length = len(os.listdir('seg_featuremap/'))
        #     if Length < 12800:
        #         save_image_tensor2cv2_gray(FA_seg_ft1[0][i],'seg_featuremap/FA_seg_ft1{0}.jpg'.format(str(i)))
        #         save_image_tensor2cv2_gray(FA_seg_ft2[0][i],'seg_featuremap/FA_seg_ft2{0}.jpg'.format(str(i)))
        recovered_B = A2BG(fake_A,FA_seg_ft1,FA_seg_ft2)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0

        loss_G_cycle = loss_cycle_ABA + loss_cycle_BAB
        G_cycle_losses.append(loss_G_cycle.item())
        train_hist['G_cycle_loss'].append(loss_G_cycle.item())

        ###################################
        # Total loss
        loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB  + loss_identity_A + loss_identity_B  + color_loss_A2B + color_loss_B2A
        
        if args.if_edge_loss:
            loss_G +=  elosses
       
        # if args.if_color_loss:
        #     loss_G +=  color_loss_B2A + color_loss_A2B

        loss_G.backward()

        G_losses.append(loss_G.item())
        train_hist['G_loss'].append(loss_G.item())
        optimizer_G.step()

        ###################################

        # train D

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        # edge phase and image phase(concate)
        edge_feature_ra = lp_realA.reshape(args.batch_size,1,256,256)
        S_I_feature = torch.cat([real_A,edge_feature_ra],dim=1)
        # S_I_feature = real_A + real_A * torch.sigmoid(edge_feature_ra)
        pred_real = AD(S_I_feature)
        # pred_real = AD(S_I_feature)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        edge_feature_fa = lp_fakeA.reshape(args.batch_size,1,256,256)
        S_I_feature_fa = torch.cat([fake_A.detach(),edge_feature_fa],dim=1)
        # S_I_feature_fa = fake_A.detach() + fake_A.detach() * torch.sigmoid(edge_feature_fa)
        # pred_fake = AD(fake_A.detach())
        pred_fake = AD(S_I_feature_fa)
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        # loss_D_A = (loss_D_real + loss_D_fake) * 0.5
        loss_D_A = (loss_D_real + loss_D_fake) * 10

        loss_D_A.backward(retain_graph=True)

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######

        optimizer_D_B.zero_grad()

        # Real loss
        edge_feature_rb = lp_realB.reshape(args.batch_size,1,256,256)
        S_I_feature_rb = torch.cat([real_B,edge_feature_rb],dim=1)
        # S_I_feature_rb =  real_B + real_B * torch.sigmoid(edge_feature_rb)
        # pred_real = BD(real_B)
        pred_real = BD(S_I_feature_rb)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        edge_feature_fb = lp_realB.reshape(args.batch_size,1,256,256)
        S_I_feature_fb = torch.cat([fake_B.detach(),edge_feature_fb],dim=1)
        # S_I_feature_fb = fake_B.detach() + fake_B.detach() * torch.sigmoid(edge_feature_fb)
        # pred_fake = BD(fake_B.detach())
        pred_fake = BD(S_I_feature_fb)
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        loss_D_B.backward(retain_graph=True)

        loss_D = loss_D_A + loss_D_B

        D_losses.append(loss_D.item())
        train_hist['D_loss'].append(loss_D.item())


        optimizer_D_B.step()

    ###################################

    per_epoch_time = time.time() - epoch_start_time
    train_hist['per_epoch_time'].append(per_epoch_time)

    print(
        '[%d/%d] - time: %.2f, G_loss: %.3f, G_identity_loss: %.3f, G_GAN_loss: %.3f, G_cycle_loss: %.3f, D_loss: %.3f,edge_lossAB:%.3f,edge_lossBA:%.3f' % (
        (epoch + 1), args.train_epoch, per_epoch_time, torch.mean(torch.FloatTensor(G_losses)), torch.mean(torch.FloatTensor(G_identity_losses)),
        torch.mean(torch.FloatTensor(G_GAN_losses)), torch.mean(torch.FloatTensor(G_cycle_losses)), torch.mean(torch.FloatTensor(D_losses)),torch.mean(torch.FloatTensor(EdgeLossAB)),torch.mean(torch.FloatTensor(EdgeLossBA))))

    with torch.no_grad():
        A2BG.eval()
        for n, (x, _) in enumerate(test_loader_src):
            x = x.to(device)
            xGray = x[:,0,:,:].view(1, 1, args.input_size, args.input_size)
            xft1,xft2 = seg_net(xGray)
            G_recon = A2BG(x,xft1,xft2)
            result = torch.cat((x[0], G_recon[0]), 2)
            path = os.path.join('results',args.name + '_results', 'Colorization', str(epoch+1) + '_epoch_' + args.name + '_test_' + str(n) + '.png')
            plt.imsave(path, (result.cpu().numpy().transpose(1, 2, 0) + 1) / 2)

    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    # Save models checkpoints
    torch.save(A2BG.state_dict(), os.path.join('results',args.name + '_results', 'A2BG_generator_latest.pkl'))
    torch.save(AD.state_dict(), os.path.join('results',args.name + '_results', 'AD_discriminator_latest.pkl'))
    torch.save(B2AG.state_dict(), os.path.join('results',args.name + '_results', 'B2AG_generator_latest.pkl'))
    torch.save(BD.state_dict(), os.path.join('results',args.name + '_results', 'BD_discriminator_latest.pkl'))

total_time = time.time() - start_time
train_hist['total_time'].append(total_time)

print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_time'])), args.train_epoch, total_time))
print("Training finish!... save training results")

torch.save(A2BG.state_dict(), os.path.join('results',args.name + '_results',  'A2BG_generator_param.pkl'))
torch.save(AD.state_dict(), os.path.join('results',args.name + '_results',  'AD_discriminator_param.pkl'))
torch.save(B2AG.state_dict(), os.path.join('results',args.name + '_results',  'B2AG_generator_param.pkl'))
torch.save(BD.state_dict(), os.path.join('results',args.name + '_results',  'BD_discriminator_param.pkl'))
with open(os.path.join('results',args.name + '_results',  'train_hist.pkl'), 'wb') as f:
    pickle.dump(train_hist, f)