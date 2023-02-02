import os
import torch
import timeit
import SimpleITK as sItk
import numpy as np
import glob
from matplotlib import pyplot as plt
from PIL import Image
import torch.nn as nn
import cv2
import argparse
from attunet import *
import pytorch_ssim
import math


def denorm(img, max=2200, min=200):
    img = img * std_i + mean_i
    img = (torch.clip(img, min, max) - min) / (max - min)
    return img

def load_3d(name):
    X = sItk.GetArrayFromImage(sItk.ReadImage(name, sItk.sitkFloat32))

    return X


def save_img(I_img, savename):
    I2 = sItk.GetImageFromArray(I_img, isVector=False)
    sItk.WriteImage(I2, savename)


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def load_pretrain_model(model_dir, lr, model_img, num_iteration, lr_decay_times, decay_rate):
    curr_lr = lr
    model_img_name = sorted(glob.glob(model_dir + '/model_img_*.pth'))
    try:
        model_img_name = model_img_name[len(model_img_name) - 1]
        model_img.load_state_dict(torch.load(model_img_name))

        current_iter = int(model_img_name[len(model_img_name) - 10:len(model_img_name) - 4])
        optimizer = torch.optim.Adam(model_img.parameters())
        for iter in range(0, current_iter):
            if (iter + 1) % (int(num_iteration / lr_decay_times)) == 0:
                curr_lr = decay_rate * curr_lr
                update_lr(optimizer, curr_lr)
    except:
        print('no model!')
        current_iter = 0
        optimizer = torch.optim.Adam(model_img.parameters(), lr=curr_lr)
    return current_iter, optimizer, model_img


num_epoch = 100
batch_size = 2
lr = 0.006
lr_decay_times = 3
decay_rate = 0.3
root_dir = './'
result_dir = root_dir + '/Result'
val_epoch = 1
start_channel = 64+32
net_gpu_ind = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = net_gpu_ind

model_dir = result_dir + '/model_epoch_' + str(num_epoch) + '_lr_' + str(lr)

if not os.path.isdir(result_dir):
    os.mkdir(result_dir)
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)

if not os.path.isdir(model_dir + '/figure'):
    os.mkdir(model_dir + '/figure')
if not os.path.isdir(model_dir + '/iter_img'):
    os.mkdir(model_dir + '/iter_img')


model_img = AttU_Net2(img_ch=1, output_ch=1, start_channel=start_channel).cuda()
current_epoch, optimizer, model_img = load_pretrain_model(model_dir, lr, model_img, num_epoch, lr_decay_times, decay_rate)


if current_epoch != num_epoch:
    try:
        loss_value_name = sorted(glob.glob(model_dir + '/loss_totalIter_*.npy'))
        loss_all = np.load(loss_value_name[0])
    except:
        loss_all = np.zeros((2, num_epoch))
    curr_lr = lr

train_inputs_info = glob.glob(root_dir + '/train_input/*.nii.gz')
train_labels_info = glob.glob(root_dir + '/train_label/*.nii.gz')
test_inputs_info = glob.glob(root_dir + '/test_input/*.nii.gz')
test_labels_info = glob.glob(root_dir + '/test_label/*.nii.gz')


mse_loss = torch.nn.MSELoss()
img_shape = load_3d(train_inputs_info[0]).shape

xii = np.zeros((len(range(0, len(train_inputs_info), 20)), img_shape[0], img_shape[1]))
kk=0
for i in range(0, len(train_inputs_info), 20):
    xii[kk, :, :] = load_3d(train_inputs_info[i])
    kk=kk+1
mean_i = np.mean(xii)
std_i = np.std(xii)
for epoch in range(current_epoch, num_epoch):
    start1 = timeit.default_timer()
    train_img_mse_loss = 0.0
    train_img_index_sum = torch.randperm(len(train_inputs_info))
    for j in range(0, int(np.floor(len(train_img_index_sum) / batch_size))):
        train_img_index = train_img_index_sum[j * batch_size: (j + 1) * batch_size]
        in_img = torch.zeros(batch_size, 1, img_shape[0], img_shape[1])
        lb_img = torch.zeros(batch_size, 1, img_shape[0], img_shape[1])
        for k in range(0, batch_size):
            in_img[k, :, :, :] = torch.from_numpy(load_3d(train_inputs_info[train_img_index[k]]))
            lb_img[k, :, :, :] = torch.from_numpy(load_3d(train_labels_info[train_img_index[k]]))
        lb_img = lb_img.cuda()
        in_img = in_img.cuda()

        lb_img = (lb_img-mean_i)/std_i
        in_img = (in_img-mean_i)/std_i

        op_img = model_img(in_img)

        mse_value = mse_loss(op_img, lb_img)
        ssim_value = pytorch_ssim.ssim(op_img, lb_img).item()
        ssim_loss = pytorch_ssim.SSIM()
        ssim_loss_1 = 1 - ssim_loss(op_img, lb_img)

        loss = 0.3*mse_value + 0.7*ssim_loss_1

        train_img_mse_loss = train_img_mse_loss + np.array([loss.item()])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch*int(np.floor(len(train_img_index_sum) / batch_size))+j) % 20 == 0:
            with torch.no_grad():
                save_slice = 1
                in_img = torch.from_numpy(load_3d(train_inputs_info[save_slice])).cuda()
                lb_img = torch.from_numpy(load_3d(train_labels_info[save_slice])).cuda()

                lb_img = (lb_img - mean_i) / std_i
                in_img = (in_img - mean_i) / std_i

                in_img = torch.reshape(in_img, (1, 1, img_shape[0], img_shape[1]))
                lb_img = torch.reshape(lb_img, (1, 1, img_shape[0], img_shape[1]))
                op_img = model_img(in_img)
                op_img = np.squeeze((op_img[0, 0, :, :].cpu().detach().numpy()))
                op_img = 255*(op_img-np.min(op_img))/(np.max(op_img)-np.min(op_img))
                im = Image.fromarray(np.uint8(op_img))
                im.save(model_dir + '/iter_img/iter_op_img_' + str(epoch).rjust(5, '0') + '_' + str(j).rjust(5, '0') + '.png')
                if j == 0:
                    in_img = np.squeeze((in_img[0, 0, :, :].cpu().detach().numpy()))
                    in_img = 255 * (in_img - np.min(in_img)) / (np.max(in_img) - np.min(in_img))
                    lb_img = np.squeeze((lb_img[0, 0, :, :].cpu().detach().numpy()))
                    lb_img = 255 * (lb_img - np.min(lb_img)) / (np.max(lb_img) - np.min(lb_img))

                    im = Image.fromarray(np.uint8(lb_img))
                    im.save(model_dir + '/iter_img/iter_lb_img.png')

                    im = Image.fromarray(np.uint8(in_img))
                    im.save(model_dir + '/iter_img/iter_raw_img.png')

    if (epoch % val_epoch == 0) or (epoch == num_epoch - 1):
        with torch.no_grad():
            test_img_mse_loss = 0.0
            test_img_index_sum = torch.randperm(len(test_inputs_info))
            for j in range(0, int(np.floor(len(test_img_index_sum) / batch_size))):
                test_img_index = test_img_index_sum[j * batch_size: (j + 1) * batch_size]
                in_img = torch.zeros(batch_size, 1, img_shape[0], img_shape[1])
                lb_img = torch.zeros(batch_size, 1, img_shape[0], img_shape[1])
                for k in range(0, batch_size):
                    in_img[k, :, :, :] = torch.from_numpy(load_3d(test_inputs_info[test_img_index[k]]))
                    lb_img[k, :, :, :] = torch.from_numpy(load_3d(test_labels_info[test_img_index[k]]))

                lb_img = lb_img.cuda()
                in_img = in_img.cuda()

                lb_img = (lb_img - mean_i) / std_i
                in_img = (in_img - mean_i) / std_i

                op_img = model_img(in_img)

                mse_value = mse_loss(op_img, lb_img)
                ssim_value = pytorch_ssim.ssim(op_img, lb_img).item()
                ssim_loss = pytorch_ssim.SSIM()
                ssim_loss_1 = 1 - ssim_loss(op_img, lb_img)

                test_loss = 0.3*mse_value + 0.7*ssim_loss_1

                test_img_mse_loss = test_img_mse_loss + np.array([test_loss.item()])

    model_name = model_dir + '/model_img_' + str(epoch + 1).rjust(6, '0') + '.pth'
    torch.save(model_img.state_dict(), model_name)

    loss_all[0, epoch] = train_img_mse_loss / int(np.floor(len(train_img_index_sum) / batch_size))
    loss_all[1, epoch] = test_img_mse_loss / int(np.floor(len(test_img_index_sum) / batch_size))

    fig, ax = plt.subplots()
    ax.plot(np.linspace(1, epoch + 1, epoch + 1), loss_all[0, 0:epoch + 1], label='train')
    ax.plot(np.linspace(1, epoch + 1, epoch + 1), loss_all[1, 0:epoch + 1], label='test')
    ax.set_xlabel('epoch')
    ax.set_ylabel('mse')
    ax.set_title('loss')
    ax.legend()
    plt.show(block=False)
    plt.savefig(model_dir + '/figure/fig_mse_loss_' + str(epoch).rjust(7, '0'))
    plt.close('all')

    np.save(model_dir + '/loss_totalIter_' + str(num_epoch) + '.npy', loss_all)
    print('epoch: ' + str(epoch) + ' -> mse train/test ' + str(round(loss_all[0, epoch], 3)) + ' / ' + str(round(loss_all[1, epoch], 3)))
    if (epoch + 1) % (int(num_epoch / lr_decay_times)) == 0:
        curr_lr = decay_rate * curr_lr
        update_lr(optimizer, curr_lr)
    start2 = timeit.default_timer()
    print('Training: it still takes: ' + str(
        (num_epoch - (epoch + 1)) * (start2 - start1) / 60.0) + 'mins')

if not os.path.isdir(model_dir + '/pred_img'):
    os.mkdir(model_dir + '/pred_img')
for j in range(0, len(test_inputs_info)):
    in_img = torch.from_numpy(load_3d(test_inputs_info[j]))
    in_img = torch.reshape(in_img, (1, 1,) + in_img.shape)
    in_img = in_img.cuda()
    in_img = (in_img - mean_i) / std_i
    op_img = model_img(in_img)
    op_img = op_img*std_i+mean_i
    save_img(op_img.cpu().detach().numpy(), model_dir + '/pred_img/pred_' + test_inputs_info[j][str.find(test_inputs_info[j], 'input_angle'):len(test_inputs_info[j])])
