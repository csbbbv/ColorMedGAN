import argparse
import os
import time
import matplotlib.pyplot as plt
import pickle
from torchvision import transforms
import torch
import torch.nn as nn
import utils
import network as net

from collections import OrderedDict
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--name', required=False, default='oasis_color_v15_concat_discri_network(upsample)_bidirectG',  help='project name')
parser.add_argument('--src_data', required=False, default='',  help='src data path')
parser.add_argument('--in_ngc', type=int, default=3, help='input channel for generator')
parser.add_argument('--out_ngc', type=int, default=3, help='output channel for generator')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--input_size', type=int, default=256, help='input size')
args = parser.parse_args()

print('------------ Options -------------')
for k, v in sorted(vars(args).items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')

# results save path
if not os.path.isdir(os.path.join('/home/user/workspace/shaobo/colorMedCycleGAN/results',args.name + '_results', 'test')):
    os.makedirs(os.path.join('/home/user/workspace/shaobo/colorMedCycleGAN/results',args.name + '_results', 'test'))

transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
 ])

test_loader_src = utils.data_load(os.path.join('/media/user/8f28289d-aa45-46ab-b9e4-a7d4d9f08b7e/shaobo/colorization_tmp_oasis/data/', args.src_data), 'test', transform, 1, shuffle=True, drop_last=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# G = net.generator(args.in_ngc, args.out_ngc, args.ngf)

G = net.generator_seg_attention_upsample(args.in_ngc, args.out_ngc, args.ngf)

state_dict = torch.load("/home/user/workspace/shaobo/colorMedCycleGAN/results/oasis_color_v15_concat_discri_network(upsample)_bidirectG_results/A2BG_generator_latest.pkl")
new_state_dict = OrderedDict()
for k, v in state_dict.items(): # k为module.xxx.weight, v为权重
    name = k[7:] # 截取`module.`后面的xxx.weight
    new_state_dict[name] = v
# load params
# net = XXXnet()
G.load_state_dict(new_state_dict)


# G.load_state_dict(torch.load("./results/oasis_color_v5_results/A2BG_generator_param.pkl"))
G.to(device)

seg_net = net.Baseline(num_classes=5)
state_dict = torch.load("/home/user/workspace/shaobo/colorMedCycleGAN/segmentation/checkpoint/oasis_v1_orig/unet-dice_-2021-07-02 22:41:57/30.pkl")
new_state_dict = OrderedDict()
# new_state_dict = {}
for k, v in state_dict.items(): # k为module.xxx.weight, v为权重
    name = k[7:] # 截取`module.`后面的xxx.weight
    new_state_dict[name] = v
seg_net.load_state_dict(new_state_dict)
seg_net.cuda()
seg_net.eval()



with torch.no_grad():
    for n, (x, _) in enumerate(test_loader_src):
        x = x.to(device)
        xGray = x[:,0,:,:].view(1, 1, args.input_size, args.input_size)
        xft1,xft2 = seg_net(xGray)
        G_recon = G(x, xft1,xft2 )
        result1 = x[0]
        result2 = G_recon[0]#torch.cat((x[0], G_recon[0]), 2)
        # path = os.path.join(args.name + '_results', 'test',
        #                 str(n + 1) + args.name + '_test_' + str(n + 1) + '.png')
        path1 = os.path.join('/home/user/workspace/shaobo/colorMedCycleGAN/results',args.name + '_results', 'test','ori'+str(n)+'.png')
        path2 = os.path.join('/home/user/workspace/shaobo/colorMedCycleGAN/results',args.name + '_results', 'test','pred'+str(n)+'.png')
        plt.imsave(path2, (result2.cpu().detach().numpy().transpose(1, 2, 0) + 1) / 2)
        plt.imsave(path1, (result1.cpu().detach().numpy().transpose(1, 2, 0) + 1) / 2)
