# -*- coding: utf-8 -*-
# batchsize 32  epochs 60
from __future__ import print_function, division
# from visdom import Visdom  # python -m visdom.server 启动Visdom监控服务器
from tensorboardX import SummaryWriter
from shutil import copyfile
import cv2
import yaml
from random_erasing import RandomErasing
from model import ft_net, ft_net_dense, PCB
import os
import time
import matplotlib.pyplot as plt

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
import torch.backends.cudnn as cudnn
import matplotlib
import wandb
import json
matplotlib.use('agg')
# from PIL import Image


######################################################################
# Save model
######################################################################
def save_network(network, epoch_label, exp_name):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join('./model', exp_name, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda(gpu_ids[0])


######################################################################
# prepare datalodaer
######################################################################
def prepare_data(opt):
    transform_train_list = [
        # transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        # 重置图像大小（分辨率），interpolation为插值方法
        transforms.Resize((256, 128), interpolation=InterpolationMode.BICUBIC),
        transforms.Pad(10),  # 填充
        transforms.RandomCrop((256, 128)),  # 按指定尺寸随机裁剪图像（中心坐标随机）
        transforms.RandomHorizontalFlip(),  # 以0.5概率使图像随机水平翻转  （这些都是增强数据的实际效果，泛化性等）
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [
                             0.229, 0.224, 0.225])  # 数据归一化
    ]

    transform_val_list = [
        # Image.BICUBIC
        transforms.Resize(
            size=(256, 128), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    if opt.PCB:
        transform_train_list = [
            transforms.Resize(
                (384, 192), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        transform_val_list = [
            # Image.BICUBIC
            transforms.Resize(
                size=(384, 192), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

    if opt.erasing_p > 0:  # 随机擦除方法，一种data augmentation方法
        transform_train_list = transform_train_list + \
            [RandomErasing(probability=opt.erasing_p, mean=[0.0, 0.0, 0.0])]

    if opt.color_jitter:  # 对颜色的数据增强：图像亮度、饱和度、对比度变化
        transform_train_list = [transforms.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train_list

    print(transform_train_list)
    data_transforms = {
        'train': transforms.Compose(transform_train_list),  # 组合所有的transform操作
        'val': transforms.Compose(transform_val_list),
    }

    train_all = '_all' if opt.train_all else ''

    image_datasets = {}
    image_datasets['train'] = datasets.ImageFolder(os.path.join(opt.data_dir, 'train' + train_all),
                                                   data_transforms['train'])
    image_datasets['val'] = datasets.ImageFolder(os.path.join(opt.data_dir, 'val'),
                                                 data_transforms['val'])

    dataloader = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                 shuffle=True, num_workers=8, pin_memory=True)
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    return dataloader, dataset_sizes, class_names


######################################################################
# Training the model
######################################################################
def train_model(model, dataloader, use_gpu, num_epochs=25):
    # set criterion
    criterion = nn.CrossEntropyLoss()

    # set optimizer
    if not opt.PCB:
        ignored_params = list(map(id, model.model.fc.parameters())) + \
            list(map(id, model.classifier.parameters()))
        base_params = filter(lambda p: id(
            p) not in ignored_params, model.parameters())
        # 这里的SGD做了很多的改进，这里的optimizer为全局微调，即原有层和新层赋予不同的学习速率，原有层为0.1*opt.lr,新层为opt.lr
        # momentum 动量加速,这是一个改进。
        # 为了有效限制模型中的自由参数数量以避免过度拟合，可以调整成本函数。
        # 一个简单的方法是通过在权重上引入零均值高斯先验值，这相当于将代价函数改变为E〜（w）= E（w）+λ2w2。
        # 在实践中，这会惩罚较大的权重，并有效地限制模型中的自由度。正则化参数λ决定了如何将原始成本E与大权重惩罚进行折衷。这是weight_decay参数的意义
        # nesterov:使用NAG（牛顿加速梯度）算法，是对momentum的改进

        optimizer_ft = optim.SGD([
            {'params': base_params, 'lr': 0.1 * opt.lr},
            {'params': model.model.fc.parameters(), 'lr': opt.lr},
            {'params': model.classifier.parameters(), 'lr': opt.lr}
        ], weight_decay=5e-4, momentum=0.9, nesterov=True)
    else:
        ignored_params = list(map(id, model.model.fc.parameters()))
        ignored_params += (list(map(id, model.classifier0.parameters()))
                            + list(map(id, model.classifier1.parameters()))
                            + list(map(id, model.classifier2.parameters()))
                            + list(map(id, model.classifier3.parameters()))
                            + list(map(id, model.classifier4.parameters()))
                            + list(map(id, model.classifier5.parameters()))
                        #    + list(map(id, model.classifier6.parameters()))
                        #    + list(map(id, model.classifier7.parameters()))
                            )
        base_params = filter(lambda p: id(
            p) not in ignored_params, model.parameters())
        optimizer_ft = optim.SGD([
            {'params': base_params, 'lr': 0.1 * opt.lr},
            {'params': model.model.fc.parameters(), 'lr': opt.lr},
            {'params': model.classifier0.parameters(), 'lr': opt.lr},
            {'params': model.classifier1.parameters(), 'lr': opt.lr},
            {'params': model.classifier2.parameters(), 'lr': opt.lr},
            {'params': model.classifier3.parameters(), 'lr': opt.lr},
            {'params': model.classifier4.parameters(), 'lr': opt.lr},
            {'params': model.classifier5.parameters(), 'lr': opt.lr},
            # {'params': model.classifier6.parameters(), 'lr': 0.01},
            # {'params': model.classifier7.parameters(), 'lr': 0.01}
        ], weight_decay=5e-4, momentum=0.9, nesterov=True)
    print(json.dumps(optimizer.state_dict()["param_groups"][0]["lr"],indent=4))

    # set scheduler
    scheduler = lr_scheduler.StepLR(
        optimizer_ft, step_size=1, gamma=0.1)
    
    best_acc = 0.0
    print("training...")
    for epoch in range(1, num_epochs + 1):
        since = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            # Iterate over data.
            for data in dataloader[phase]:
                # get the inputs
                inputs, labels = data
                now_batch_size, c, h, w = inputs.shape
                if now_batch_size < opt.batchsize:  # skip the last batch
                    continue
                # print(inputs.shape)
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda().detach())
                    labels = Variable(labels.cuda().detach())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                # if we use low precision, input also need to be fp16
                # if fp16:
                #    inputs = inputs.half()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                if phase == 'val':
                    with torch.no_grad():
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)

                if not opt.PCB:
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)
                else:
                    part = {}
                    sm = nn.Softmax(dim=1)
                    num_part = 6
                    for i in range(num_part):
                        part[i] = outputs[i]

                    score = sm(part[0]) + sm(part[1]) + sm(part[2]) + \
                        sm(part[3]) + sm(part[4]) + sm(part[5])
                    _, preds = torch.max(score.data, 1)

                    loss = criterion(part[0], labels)
                    for i in range(num_part - 1):
                        loss += criterion(part[i + 1], labels)

                # backward
                if phase == 'train':
                    # if fp16:  # we use optimier to backward loss
                    #     with amp.scale_loss(loss, optimizer) as scaled_loss:
                    #         scaled_loss.backward()
                    # else:
                    #     loss.backward()
                    loss.backward()
                    optimizer.step()

                # statistics
                # for the new version like 0.4.0, 0.5.0 and 1.0.0
                if int(version[0]) > 0 or int(version[2]) > 3:
                    running_loss += loss.item() * now_batch_size
                else:  # for the old version like 0.3.0 and 0.3.1
                    running_loss += loss.data[0] * now_batch_size
                running_corrects += float(torch.sum(preds == labels.data))
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            if epoch_acc > best_acc:
                save_network(model, 'best', opt.exp_name)
                best_acc = epoch_acc

            wandb.log({"loss": epoch_loss, "epoch": epoch,
                       "lr0": optimizer.state_dict()["param_groups"][0]["lr"], 
                       "lr1": optimizer.state_dict()["param_groups"][1]["lr"],
                       "lr2": optimizer.state_dict()["param_groups"][2]["lr"]})

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val':
                last_model_wts = model.state_dict()
                if epoch % 4 == 0:
                    save_network(model, epoch, opt.exp_name)
                # draw_curve(epoch)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        # print('Best val Acc: {:4f}'.format(best_acc))

    # save last model weights
    model.load_state_dict(last_model_wts)
    save_network(model, 'last', opt.exp_name)


def get_model(opt):
    if opt.use_dense:
        model = ft_net_dense(len(class_names), opt.droprate)
    else:
        model = ft_net(len(class_names), opt.droprate, opt.stride)

    if opt.PCB:
        model = PCB(len(class_names))
    return model


def get_optimizer(model, opt):
    if not opt.PCB:
        ignored_params = list(map(id, model.model.fc.parameters())) + \
            list(map(id, model.classifier.parameters()))
        base_params = filter(lambda p: id(
            p) not in ignored_params, model.parameters())
        # 这里的SGD做了很多的改进，这里的optimizer为全局微调，即原有层和新层赋予不同的学习速率，原有层为0.1*opt.lr,新层为opt.lr
        # momentum 动量加速,这是一个改进。
        # 为了有效限制模型中的自由参数数量以避免过度拟合，可以调整成本函数。
        # 一个简单的方法是通过在权重上引入零均值高斯先验值，这相当于将代价函数改变为E〜（w）= E（w）+λ2w2。
        # 在实践中，这会惩罚较大的权重，并有效地限制模型中的自由度。正则化参数λ决定了如何将原始成本E与大权重惩罚进行折衷。这是weight_decay参数的意义
        # nesterov:使用NAG（牛顿加速梯度）算法，是对momentum的改进

        optimizer_ft = optim.SGD([
            {'params': base_params, 'lr': 0.1 * opt.lr},
            {'params': model.model.fc.parameters(), 'lr': opt.lr},
            {'params': model.classifier.parameters(), 'lr': opt.lr}
        ], weight_decay=5e-4, momentum=0.9, nesterov=True)
    else:
        ignored_params = list(map(id, model.model.fc.parameters()))
        ignored_params += (list(map(id, model.classifier0.parameters()))
                           + list(map(id, model.classifier1.parameters()))
                           + list(map(id, model.classifier2.parameters()))
                           + list(map(id, model.classifier3.parameters()))
                           + list(map(id, model.classifier4.parameters()))
                           + list(map(id, model.classifier5.parameters()))
                        #    + list(map(id, model.classifier6.parameters()))
                        #    + list(map(id, model.classifier7.parameters()))
                           )
        base_params = filter(lambda p: id(
            p) not in ignored_params, model.parameters())
        optimizer_ft = optim.SGD([
            {'params': base_params, 'lr': 0.1 * opt.lr},
            {'params': model.model.fc.parameters(), 'lr': opt.lr},
            {'params': model.classifier0.parameters(), 'lr': opt.lr},
            {'params': model.classifier1.parameters(), 'lr': opt.lr},
            {'params': model.classifier2.parameters(), 'lr': opt.lr},
            {'params': model.classifier3.parameters(), 'lr': opt.lr},
            {'params': model.classifier4.parameters(), 'lr': opt.lr},
            {'params': model.classifier5.parameters(), 'lr': opt.lr},
            # {'params': model.classifier6.parameters(), 'lr': 0.01},
            # {'params': model.classifier7.parameters(), 'lr': 0.01}
        ], weight_decay=5e-4, momentum=0.9, nesterov=True)

    return optimizer_ft


if __name__ == '__main__':
    version = torch.__version__
    writer = SummaryWriter('log')

    # #fp16
    # try:
    #     from apex.fp16_utils import *
    # except ImportError: # will be 3.x series
    #    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')

    ######################################################################
    # Options
    ######################################################################
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--gpu_ids', default='0', type=str,
                        help='gpu_ids: e.g. 0  0,1,2  0,2')
    # parser.add_argument('--name', default='ft_ResNet50', type=str, help='output model name')
    parser.add_argument('--data_dir', default='./market1501-mod',
                        type=str, help='training dir path')
    parser.add_argument('--train_all', action='store_true',
                        help='use all training data')
    parser.add_argument('--color_jitter', action='store_true',
                        help='use color jitter in training')
    parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
    parser.add_argument('--stride', default=2, type=int, help='stride')
    parser.add_argument('--erasing_p', default=0, type=float,
                        help='Random Erasing probability, in [0,1]')
    parser.add_argument('--use_dense', action='store_true',
                        help='use densenet121')
    parser.add_argument('--lr', default=0.06, type=float, help='learning rate')
    parser.add_argument('--droprate', default=0.5,
                        type=float, help='drop rate')
    parser.add_argument('--PCB', action='store_true', help='use PCB+ResNet50')
    parser.add_argument('--exp_name', default='exp', help='name of exp')
    parser.add_argument('--epochs', default=60, help='num of epochs')
    # parser.add_argument('--fp16', action='store_true',
    #                     help='use float16 instead of float32, which will save about 50% memory')
    opt = parser.parse_args()

    wandb.init(project='test-ReID',
               name=opt.exp_name,
               config=dict(learing_rate=opt.lr,
                           batch_size=opt.batchsize, epoch=opt.epochs)
               )

    # fp16 = opt.fp16
    str_ids = opt.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        gid = int(str_id)
        if gid >= 0:
            gpu_ids.append(gid)

    use_gpu = False
    # set gpu ids
    if len(gpu_ids) > 0:
        use_gpu = True
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True

    # # 查看一个batch的图片数据
    # img = torchvision.utils.make_grid(inputs)
    # img = img.numpy().transpose((1, 2, 0))
    # mean = np.array([0.485, 0.456, 0.406])  #反归一化
    # std = np.array([0.229, 0.224, 0.225])
    # img = std * img + mean
    # cv2.imshow('one_batch_datas', img)
    # key_pressed = cv2.waitKey(0)
    # print('time:',time.time() - since)

    dataloader, dataset_sizes, class_names = prepare_data(opt)

    dir_name = os.path.join('./model', opt.exp_name)
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    # record every run
    copyfile('./train.py', os.path.join(dir_name, 'train.py'))
    copyfile('./model.py', os.path.join(dir_name, 'model.py'))
    print("*" * 20)

    # save opts
    opts_path = os.path.join(dir_name, 'opts.yaml')
    with open(opts_path, 'w') as fp:
        yaml.dump(vars(opt), fp, default_flow_style=False)
    model = get_model(opt).cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, opt)
    scheduler = lr_scheduler.StepLR(
        optimizer, step_size=30, gamma=0.1)
    
    print("*" * 20)
    train_model(model, dataloader, use_gpu, opt.epochs)
