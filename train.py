# -*- coding: utf-8 -*-
# batchsize 32  epochs 60
from __future__ import print_function, division
from visdom import Visdom  # python -m visdom.server 启动Visdom监控服务器
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
import torch.backends.cudnn as cudnn
import matplotlib
matplotlib.use('agg')
#from PIL import Image

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
    # --------
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--gpu_ids', default='0', type=str,
                        help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--name', default='ft_ResNet50',
                        type=str, help='output model name')
    parser.add_argument('--data_dir', default='..\\..\\dataset\\Market-1501-v15.09.15\\pytorch',
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
    # parser.add_argument('--fp16', action='store_true',
    #                     help='use float16 instead of float32, which will save about 50% memory')
    opt = parser.parse_args()

    # fp16 = opt.fp16
    data_dir = opt.data_dir
    name = opt.name
    str_ids = opt.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        gid = int(str_id)
        if gid >= 0:
            gpu_ids.append(gid)

    # set gpu ids
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True
    ######################################################################
    # Load Data
    # ---------
    #

    transform_train_list = [
        # transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        # 重置图像大小（分辨率），interpolation为插值方法
        transforms.Resize((256, 128), interpolation=3),
        transforms.Pad(10),  # 填充
        transforms.RandomCrop((256, 128)),  # 按指定尺寸随机裁剪图像（中心坐标随机）
        transforms.RandomHorizontalFlip(),  # 以0.5概率使图像随机水平翻转  （这些都是增强数据的实际效果，泛化性等）
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [
                             0.229, 0.224, 0.225])  # 数据归一化
    ]

    transform_val_list = [
        transforms.Resize(size=(256, 128), interpolation=3),  # Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    if opt.PCB:
        transform_train_list = [
            transforms.Resize((384, 192), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        transform_val_list = [
            # Image.BICUBIC
            transforms.Resize(size=(384, 192), interpolation=3),
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

    train_all = ''
    if opt.train_all:
        train_all = '_all'

    image_datasets = {}
    # ImageFolder 数据加载器，指定路径下加载并执行组合好的transforms操作
    image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train' + train_all),
                                                   data_transforms['train'])
    image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                                 data_transforms['val'])

    # torch.utils.data.DataLoader：
    # 该接口主要用来将自定义的数据读取接口的输出或者PyTorch已有的数据读取接口的输入按照batch size封装成Tensor，后续只需要再包装成Variable即可作为模型的输入
    # shuffle 是否将图片打乱 / num_workers：使用多少个子进程来导入数据 /pin_memory： 在数据返回前，是否将数据复制到CUDA内存中
    #
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                  shuffle=True, num_workers=8, pin_memory=True)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    use_gpu = torch.cuda.is_available()
    #
    # 画一个批次的图像
    #
    since = time.time()
    inputs, classes = next(iter(dataloaders['train']))  # 获取训练数据中的一个 batch
    print(inputs.size())
    '''
    #查看一个batch的图片数据
    img = torchvision.utils.make_grid(inputs)
    img = img.numpy().transpose((1,2,0))
    mean=np.array([0.485,0.456,0.406])  #反归一化
    std=np.array([0.229,0.224,0.225])
    img=std*img+mean
    cv2.imshow('one_batch_datas',img)
    key_pressed=cv2.waitKey(0)
    print('time:',time.time()-since)
    '''

    ######################################################################
    # Training the model
    # ------------------
    #
    # Now, let's write a general function to train a model. Here, we will
    # illustrate:
    #
    # -  Scheduling the learning rate
    # -  Saving the best model
    #
    # In the following, parameter ``scheduler`` is an LR scheduler object from
    # ``torch.optim.lr_scheduler``.

    y_loss = {}  # loss history
    y_loss['train'] = []
    y_loss['val'] = []
    y_err = {}
    y_err['train'] = []
    y_err['val'] = []
    y_acc = {}
    y_acc['train'] = []
    y_acc['val'] = []

    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()

        #best_model_wts = model.state_dict()
        #best_acc = 0.0
        viz = Visdom()
        win_loss = viz.scatter(X=np.asarray([[0, 0]]))
        #viz.line([[0.],[0.]],[0.],win='loss',opts=dict(title='epoch Loss train&val',legend=['train','val']))

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step()
                    model.train(True)  # Set model to training mode
                else:
                    model.train(False)  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0.0
                # Iterate over data.
                for data in dataloaders[phase]:
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

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        if fp16:  # we use optimier to backward loss
                            with amp.scale_loss(loss, optimizer) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()
                        optimizer.step()

                    # statistics
                    # for the new version like 0.4.0, 0.5.0 and 1.0.0
                    if int(version[0]) > 0 or int(version[2]) > 3:
                        running_loss += loss.item() * now_batch_size
                    else:  # for the old version like 0.3.0 and 0.3.1
                        running_loss += loss.data[0] * now_batch_size
                    running_corrects += float(torch.sum(preds == labels.data))

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                y_loss[phase].append(epoch_loss)
                y_err[phase].append(1.0 - epoch_acc)
                y_acc[phase].append(epoch_acc)

                # deep copy the model
                if phase == 'val':
                    last_model_wts = model.state_dict()
                    if epoch % 4 == 3:
                        save_network(model, epoch)
                    draw_curve(epoch)

            viz.scatter(X=np.array([[epoch, y_loss['train'][-1]]]),
                        name="train", win=win_loss, update='append')
            viz.scatter(X=np.array([[epoch, y_loss['val'][-1]]]),
                        name="val", win=win_loss, update='append')

            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        # print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(last_model_wts)
        save_network(model, 'last')

    ######################################################################
    # Draw Curve
    # ---------------------------
    x_epoch = []
    # test_epoch=[]
    fig = plt.figure()
    ax0 = fig.add_subplot(121, title="loss")
    ax1 = fig.add_subplot(122, title="top1_err")
    #ax1 = fig.add_subplot(122, title="top1acc")

    def draw_curve(current_epoch):
        x_epoch.append(current_epoch)
        ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
        ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
        ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
        ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
        if current_epoch == 0:
            ax0.legend()
            ax1.legend()
        fig.savefig(os.path.join('./model', name, 'train.jpg'))

    ######################################################################
    # Save model
    # ---------------------------
    def save_network(network, epoch_label):
        save_filename = 'net_%s.pth' % epoch_label
        save_path = os.path.join('./model', name, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if torch.cuda.is_available():
            network.cuda(gpu_ids[0])

    ######################################################################
    # Finetuning the convnet
    # ----------------------
    #
    # Load a pretrainied model and reset final fully connected layer.
    #
    # 选择不同的模型
    if opt.use_dense:
        model = ft_net_dense(len(class_names), opt.droprate)
    else:
        model = ft_net(len(class_names), opt.droprate, opt.stride)

    if opt.PCB:
        model = PCB(len(class_names))

    dummy_input = torch.rand(25, 3, 28, 28)  # 假设输入20张1*28*28的图片
    with SummaryWriter(comment='ReNet50') as w:
        w.add_graph(model, (dummy_input,))

    # Market_1501测试集galley是750个人，还有一个‘-1’编号是表示检测出来其他人的图（不在这 750 人中）所以最后应分成751类，class_num为751
    print('class_num: ', len(class_names))

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
            {'params': base_params, 'lr': 0.1*opt.lr},
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
                           #+list(map(id, model.classifier6.parameters() ))
                           #+list(map(id, model.classifier7.parameters() ))
                           )
        base_params = filter(lambda p: id(
            p) not in ignored_params, model.parameters())
        optimizer_ft = optim.SGD([
            {'params': base_params, 'lr': 0.1*opt.lr},
            {'params': model.model.fc.parameters(), 'lr': opt.lr},
            {'params': model.classifier0.parameters(), 'lr': opt.lr},
            {'params': model.classifier1.parameters(), 'lr': opt.lr},
            {'params': model.classifier2.parameters(), 'lr': opt.lr},
            {'params': model.classifier3.parameters(), 'lr': opt.lr},
            {'params': model.classifier4.parameters(), 'lr': opt.lr},
            {'params': model.classifier5.parameters(), 'lr': opt.lr},
            #{'params': model.classifier6.parameters(), 'lr': 0.01},
            #{'params': model.classifier7.parameters(), 'lr': 0.01}
        ], weight_decay=5e-4, momentum=0.9, nesterov=True)

    # Decay LR by a factor of 0.1 every 40 epochs
    # 等间隔调整学习率,调整倍数为 gamma 倍,调整间隔为 step_size。间隔单位是step,在0-40个epochs为lr*gamma,40~80epochs为lr*gamma*gamma......
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_ft, step_size=30, gamma=0.1)

    ######################################################################
    # Train and evaluate
    # ^^^^^^^^^^^^^^^^^^
    #
    # It should take around 1-2 hours on GPU.
    #
    dir_name = os.path.join('./model', name)
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    # record every run
    copyfile('./train.py', os.path.join(dir_name, 'train.py'))
    copyfile('./model.py', os.path.join(dir_name, 'model.py'))

    # save opts
    opts_path = os.path.join(dir_name, 'opts.yaml')
    with open(opts_path, 'w') as fp:
        yaml.dump(vars(opt), fp, default_flow_style=False)

    # model to gpu
    # 将模型搬移到GPU上
    model = model.cuda()
    # if fp16:
    #     model = network_to_half(model)
    #     optimizer_ft = FP16_Optimizer(optimizer_ft, static_loss_scale = 128.0)
    # 交叉熵代价函数
    criterion = nn.CrossEntropyLoss()

    train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                num_epochs=60)
