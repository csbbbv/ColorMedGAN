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

import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--name', required=False, default='/home/user/workspace/shaobo/colorCycleGAN/data/thyroid_nodule_test/medical/val/benign',  help='project name')
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

# # results save path
# if not os.path.isdir(os.path.join(args.name + '_results', 'test')):
#     os.makedirs(os.path.join(args.name + '_results', 'test'))

transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
 ])

test_loader_src = utils.data_load('data/thyroid_nodule/val','benign',transform,1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

G = net.generator(args.in_ngc, args.out_ngc, args.ngf)
G.load_state_dict(torch.load("./color_edgeloss_results/A2BG_generator_param.pkl"))
G.to(device)
with torch.no_grad():
    for n, (x, name) in enumerate(test_loader_src):
        x = x.to(device)
        G_recon = G(x)
        result1 = x[0]
        result2 = G_recon[0]#torch.cat((x[0], G_recon[0]), 2)
        # path = os.path.join(args.name + '_results', 'test',
        #                 str(n + 1) + args.name + '_test_' + str(n + 1) + '.png')
        # path1 = os.path.join(args.name + '_results', 'test','ori'+str(n)+'.jpg')
        path2 = os.path.join(args.name ,'pred'+str(n)+'.jpg')
        plt.imsave(path2, (result2.cpu().detach().numpy().transpose(1, 2, 0) + 1) / 2)
        # plt.imsave(path1, (result1.cpu().detach().numpy().transpose(1, 2, 0) + 1) / 2)
