import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import float32, optim
from torch.utils import data
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import numpy as np

import os
import random
import socket
from tqdm import tqdm

from PIL import Image

import utils
import network
import datasets as dt
from metrics import StreamSegMetrics
from utils import ext_transforms as et
from utils import histeq as hq


def build_log(opts, LOGDIR) -> SummaryWriter:
    # Tensorboard option
    if opts.save_log:
        logdir = os.path.join(LOGDIR, 'log')
        writer = SummaryWriter(log_dir=logdir)
    # Validate option
    if opts.save_val_results or opts.save_last_results:
        logdir = os.path.join(LOGDIR, 'val_results')
        os.mkdir(logdir)
        opts.save_val_dir = logdir
    # Train option
    if opts.save_train_results:
        logdir = os.path.join(LOGDIR, 'train_results')
        os.mkdir(logdir)
        opts.save_train_dir = logdir
    # Save best model option
    if opts.save_model:
        logdir = os.path.join(LOGDIR, 'best_param')
        os.mkdir(logdir)
        opts.save_ckpt = logdir
    else:
        logdir = os.path.join(LOGDIR, 'cache_param')
        os.mkdir(logdir)
        opts.save_ckpt = logdir
    # Checkpoint option
    if opts.ckpt == None:
        logdir = os.path.join(LOGDIR, 'ckpt')
        os.mkdir(logdir)

    # Save Options description
    with open(os.path.join(LOGDIR, 'summary.txt'), 'w') as f:
        for k, v in vars(opts).items():
            f.write("{} : {}\n".format(k, v))

    return writer

def get_dataset(opts):
    if opts.is_rgb:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = [0.485]
        std = [0.229]

    train_transform = et.ExtCompose([
        et.ExtResize(size=opts.resize),
        et.ExtRandomCrop(size=opts.crop_size, pad_if_needed=True),
        et.ExtScale(scale=opts.scale_factor),
        et.ExtRandomVerticalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=mean, std=std) 
        ])
    val_transform = et.ExtCompose([
        et.ExtResize(size=opts.resize),
        et.ExtRandomCrop(size=opts.crop_size, pad_if_needed=True),
        et.ExtScale(scale=opts.scale_factor),
        et.ExtToTensor(),
        et.ExtNormalize(mean=mean, std=std) 
        ])

    if opts.dataset == "CPN":
        train_dst = dt.CPNSegmentation(root=opts.data_root, datatype=opts.dataset,
                                        image_set='train', transform=train_transform, 
                                        is_rgb=opts.is_rgb)
        val_dst = dt.CPNSegmentation(root=opts.data_root, datatype=opts.dataset,
                                     image_set='val', transform=val_transform, 
                                     is_rgb=opts.is_rgb)
    elif opts.dataset == "CPN_all":
        train_dst = dt.CPNALLSegmentation(root=opts.data_root, datatype=opts.dataset,
                                         image_set='train', transform=train_transform, 
                                         is_rgb=opts.is_rgb)
        val_dst = dt.CPNALLSegmentation(root=opts.data_root, datatype=opts.dataset,
                                        image_set='val', transform=val_transform, 
                                        is_rgb=opts.is_rgb)
    else:
        train_dst = dt.CPN(root=opts.data_root, datatype=opts.dataset, 
                            image_set='train', transform=train_transform,
                            is_rgb=opts.is_rgb)
        val_dst = dt.CPN(root=opts.data_root, datatype=opts.dataset,
                        image_set='val', transform=val_transform,
                        is_rgb=opts.is_rgb)
    
    return train_dst, val_dst


def train(opts, devices, REPORT, LOGDIR):

    writer = build_log(opts, LOGDIR)
    
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    ''' (1) Get datasets
    '''
    train_dst, val_dst = get_dataset(opts)
    train_loader = DataLoader(train_dst, batch_size=opts.batch_size,
                                shuffle=True, num_workers=opts.num_workers, drop_last=True)
    val_loader = DataLoader(val_dst, batch_size=opts.val_batch_size, 
                                shuffle=True, num_workers=opts.num_workers, drop_last=True)
    print("Dataset: %s, Train set: %d, Val set: %d" % 
                    (opts.dataset, len(train_dst), len(val_dst)))

    ''' (2) Set up criterion
    '''
    if opts.loss_type == 'cross_entropy':
        criterion = utils.CrossEntropyLoss()
    elif opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(gamma=2, alpha=torch.tensor([0.02, 0.98]).to(devices))
    elif opts.loss_type == 'dice_loss':
        criterion = utils.DiceLoss()
    elif opts.loss_type == 'entropy_dice_loss':
        criterion = utils.EntropyDiceLoss()
    elif opts.loss_type == 'ap_cross_entropy':
        criterion = None
    elif opts.loss_type == 'ap_entropy_dice_loss':
        criterion = None
    else:
        raise NotImplementedError

    ''' (3) Load model
    '''
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(devices)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("Train from scratch...")
        resume_epoch = 0
        model.to(devices)

    ''' (4) Set up optimizer
    '''
    if opts.model.startswith("deeplab"):
        optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
        ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    else:
        optimizer = optim.RMSprop(model.parameters(), 
                                    lr=opts.lr, 
                                    weight_decay=opts.weight_decay,
                                    momentum=opts.momentum)
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, 
                                                step_size=opts.step_size, gamma=0.1)
    else:
        raise NotImplementedError

    ''' (4-1) Resume model & scheduler
    '''

    ''' (5) Set data parallel
    '''

    ''' (6) Set up metrics
    '''

    ''' (7) Train
    '''