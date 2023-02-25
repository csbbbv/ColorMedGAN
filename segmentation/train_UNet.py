from numpy.lib.function_base import append
from dataloader import Brain
import time
import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import tqdm
from torchtools import EarlyStopping
from loss import *
from metrics import diceCoeffv2
import joint_transforms as joint_transforms
import transform as extended_transforms
import tools
import sys 
import random
sys.path.append(".") 


# torch.cuda.empty_cache()

dataset_name = 'oasis_v1_orig'
model_name = 'unet'
loss_name = 'dice_'
times = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
extra_description = ''
num_class = 36
writer = SummaryWriter(os.path.join(
                'segmentation','log/{}/train'.format(dataset_name), model_name+loss_name+str(times )+extra_description))
val_writer = SummaryWriter(os.path.join(
                'segmentation', 'log/{}/val'.format(dataset_name), model_name + loss_name+str(times)+extra_description))

root_path = 'segmentation'
# 超参设置
crop_size = 128
batch_size = 4
n_epoch = 300

lr_scheduler_eps = 1e-3
lr_scheduler_patience = 10
early_stop_patience = 12
threshold_lr = 1e-6

initial_lr = 1e-4

weight_decay = 1e-5
optimizer_type = 'adam'  # adam, sgd
scheduler_type = 'no'  # ReduceLR, StepLR, poly
label_smoothing = 0.01
aux_loss = False
gamma = 0.5
alpha = 0.85
model_number = random.randint(1, 1e6)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"


# unet = net.generator(1, num_class, 64)

def get_mean_std(train_loader):
    samples, mean, std = 0, 0, 0
    for _ , (input, mask) in enumerate(train_loader):
        samples += 1
        mean += np.mean(input.numpy(), axis=(0, 2, 3))
        std += np.std(input.numpy(), axis=(0, 2, 3))
    mean /= samples
    std /= samples
    print(mean, std)

def main():
    # net = nets.generator(1,num_class,64).cuda()
    if model_name == "unet":
        # from networks.unet import Baseline
        from network import Baseline
    elif model_name == "fcn":
        from networks.fcn import Baseline
    elif model_name == "segnet":
        from networks.segnet import Baseline
    elif model_name == "attunet":
        from networks.attunet import Baseline

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    net = Baseline(num_classes= num_class)
    net = nn.DataParallel(net)
    net = net.cuda()
    train_joint_transform = joint_transforms.Compose([
        joint_transforms.Scale(256),
        joint_transforms.RandomRotate(10),
        joint_transforms.RandomHorizontallyFlip()
    ])
    #center_crop = joint_transforms.CenterCrop(crop_size)
    train_input_transform = extended_transforms.ImgToTensor()

    target_transform = extended_transforms.MaskToTensor()
    train_set = Brain('/media/agent001/8f28289d-aa45-46ab-b9e4-a7d4d9f08b7e/shaobo/oasis1/', 'train',
                                joint_transform=train_joint_transform, #center_crop=center_crop,
                                transform=train_input_transform, target_transform=target_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    val_set = Brain('/media/agent001/8f28289d-aa45-46ab-b9e4-a7d4d9f08b7e/shaobo/oasis1/', 'val',
                                joint_transform=train_joint_transform,#center_crop=center_crop,
                                transform=train_input_transform, target_transform=target_transform)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    if loss_name == 'dice_':
        criterion = SoftDiceLoss(num_classes =num_class,activation='sigmoid').cuda()
        # criterion = SoftDiceLossV2(num_classes = 5,activation='sigmoid').cuda()
    elif loss_name == 'bce_':
        criterion = nn.BCEWithLogitsLoss().cuda()
    # elif loss_name == 'wbce_':
    #     criterion = WeightedBCELossWithSigmoid().cuda()
    # elif loss_name == 'er_':
    #     criterion = EdgeRefinementLoss().cuda()
    early_stopping = EarlyStopping(early_stop_patience, verbose=True, delta=lr_scheduler_eps,
                                   path=os.path.join(root_path, 'checkpoint', '{}.pth'.format(model_name)))
    optimizer = optim.Adam(net.parameters(), lr=initial_lr)

    train(train_loader, val_loader,net, criterion, optimizer, n_epoch, 0,early_stopping)


def train(train_loader, val_loader,net, criterion, optimizer, num_epoches , iters,early_stopping):
    for epoch in range(1, num_epoches + 1):
        st = time.time()
        
        train_class_dices = np.array([0] * (num_class - 1), dtype=np.float)
        val_class_dices = np.array([0] * (num_class - 1), dtype=np.float)
        val_dice_arr = []
        train_losses = []
        val_losses = []

        net.train()
        for batch,(inputs, mask) in enumerate(train_loader):
            X = inputs.cuda()
            y = mask.cuda()
            optimizer.zero_grad()
            output = net(X)
            # output = torch.sigmoid(output)
            loss = criterion(output, y)
            # CrossEntropyLoss
            # loss = criterion(output, torch.argmax(y, dim=1))
            
            # output[output < 0.5] = 0
            # output[output > 0.5] = 1
            # bladder_dice = diceCoeffv2(output[:, 0:1, :], y[:, 0:1, :], activation=None).cpu().item()
            # tumor_dice = diceCoeffv2(output[:, 1:2, :], y[:, 1:2, :], activation=None).cpu().item()

            
            # mean_dice = (bladder_dice + tumor_dice) / 2
            # d_len += 1
            # b_dice += bladder_dice
            # t_dice += tumor_dice
            loss.backward()
            optimizer.step()
            iters += batch_size
            train_losses.append(loss.item())
            class_dice = []
            for i in range(1,num_class):
                cur_dice = diceCoeffv2(output[:,i:i+1,:],y[:,i:i+1,:]).cpu().item()
                class_dice.append(cur_dice)
            mean_dice = sum(class_dice) / len(class_dice)
            train_class_dices += np.array(class_dice)

            string_print = "Epoch = %d iters = %d Current_Loss = %.4f Mean Dice=%.4f Cortex Dice=%.4f Subcortical-Gray-Matter Dice=%.4f White-Matter Dice=%.4f CSF Dice=%.4f Time = %.2f"\
                           % (epoch, iters, loss.item(), mean_dice,
                              class_dice[0],class_dice[1],class_dice[2],class_dice[3], time.time() - st)
            tools.log(string_print)
            st = time.time()
            writer.add_scalar('train_main_loss', loss.item(), iters)
        

        train_loss = np.average(train_losses)
        train_class_dices = train_class_dices / batch
        train_mean_dice = train_class_dices.sum() / train_class_dices.size
        writer.add_scalar('main_loss', train_loss, epoch)
        writer.add_scalar('main_dice', train_mean_dice, epoch)
        writer.add_scalar('Cortex Dice', class_dice[0], epoch)
        writer.add_scalar('Subcortical-Gray-Matter  Dice', class_dice[1], epoch)
        writer.add_scalar('White-Matter Dice', class_dice[2], epoch)
        writer.add_scalar('CSF Dice', class_dice[3], epoch)

        # print('Epoch {}/{},Train Mean Dice {:.4}, Bladder Dice {:.4}, Tumor Dice {:.4}'.format(
        #     epoch, num_epoches, m_dice, b_dice, t_dice
        # ))
        print('epoch {}/{} - train_loss: {:.4} - train_mean_dice: {:.4} - dice_Cortex: {:.4} - dice_Subcortical-Gray-Matter: {:.4} - dice_White-Matter: {:.4} - dice_CSF:{:.4}'.format(
                epoch, num_epoches, train_loss, train_mean_dice, train_class_dices[0], train_class_dices[1], train_class_dices[2],train_class_dices[3]))

        # 验证模型
        net.eval()
        for val_batch,(inputs,mask) in enumerate(val_loader):
            val_x = inputs.cuda()
            val_y = mask.cuda()
            pred = net(val_x)
            # pred = torch.sigmoid(pred)
            val_loss = criterion(pred,val_y)
            val_losses.append(val_loss.item())
            pred = pred.cpu().detach()
            val_class_dice = []
            for i in range(1,num_class):
                val_class_dice.append(diceCoeffv2(pred[:,i:i+1,:],mask[:,i:i+1,:]))
            val_dice_arr.append(val_class_dice)
            val_class_dices += np.array(val_class_dice)
        val_loss = np.average(val_losses)

        val_dice_arr = np.array(val_dice_arr)
        std = (np.std(val_dice_arr[:, 1:2]) + np.std(val_dice_arr[:, 2:3]) + np.std(val_dice_arr[:, 3:4])+np.std(val_dice_arr[:, 4:5]) )/ num_class
        val_class_dices = val_class_dices / val_batch
        val_mean_dice = val_class_dices.sum() / val_class_dices.size
        organ_mean_dice = (val_class_dices[0] + val_class_dices[1] + val_class_dices[2]+ val_class_dices[3]) / num_class

        val_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        val_writer.add_scalar('main_loss', val_loss, epoch)
        val_writer.add_scalar('main_dice', val_mean_dice, epoch)
        val_writer.add_scalar('lesion_dice', organ_mean_dice, epoch)
        val_writer.add_scalar('Cortex Dice', val_class_dices[0], epoch)
        val_writer.add_scalar('Subcortical-Gray-Matter  Dice', val_class_dices[1], epoch)
        val_writer.add_scalar('White-Matter Dice', val_class_dices[2], epoch)
        val_writer.add_scalar('CSF Dice', val_class_dices[3], epoch)


        print('val_loss: {:.4} - val_mean_dice: {:.4} - mean: {:.4}±{:.3} - Cortex Dice={:.4} - Subcortical-Gray-Matter Dice={:.4}  - White-Matter Dice={:.4}  -  CSF Dice={:.4} '
            .format(val_loss, val_mean_dice, organ_mean_dice, std, val_class_dices[0], val_class_dices[1], val_class_dices[2],val_class_dices[3]))
        print('lr: {}'.format(optimizer.param_groups[0]['lr']))

        
        if os.path.exists('segmentation/checkpoint/{}/{}'.format(dataset_name,model_name+'-' + loss_name+'-' + str(times)+ extra_description)) is False:
            os.mkdir('segmentation/checkpoint/{}/{}'.format(dataset_name,model_name+'-' + loss_name+'-' + str(times)+ extra_description))
        torch.save(net, 'segmentation/checkpoint/{}/{}/{}.pth'.format(dataset_name,model_name+'-' + loss_name+'-' + str(times)+ extra_description,str(epoch)))
        torch.save(net.state_dict(), 'segmentation/checkpoint/{}/{}/{}.pkl'.format(dataset_name,model_name+'-' + loss_name+'-' + str(times)+ extra_description,str(epoch)))
        early_stopping(organ_mean_dice, net, epoch,)
        if early_stopping.early_stop or optimizer.param_groups[0]['lr'] < threshold_lr:
            print("Early stopping")
            # 结束模型训练
            break



        if epoch == num_epoches:
            # torch.save(net, 'segmentation/checkpoint/exp/{}.pth'.format(model_name + loss_name + times + extra_description))
            writer.close()
    print('----------------------------------------------------------')
    print('save epoch {}'.format(early_stopping.save_epoch))
    print('stoped epoch {}'.format(epoch))
    print('----------------------------------------------------------')

if __name__ == '__main__':
    main()