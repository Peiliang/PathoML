# Base / Native
import csv
from collections import Counter
import copy
import json
import functools
import gc
import logging
import math
import os
import pdb
import pickle
import random
import sys
import tables
import time
from tqdm import tqdm

# Numerical / Array
import numpy as np

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.autograd import Variable
from torch.nn import init, Parameter
from torch.utils.data import DataLoader
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from torchvision import datasets, transforms
import torch.optim.lr_scheduler as lr_scheduler
from torch_geometric.nn import GCNConv, SAGEConv, GraphConv, GatedGraphConv, GATConv
from torch_geometric.nn import GraphConv, TopKPooling, SAGPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.transforms.normalize_features import NormalizeFeatures

# Env
from fusion import *
from options import parse_args
from utils import *


################
# Network Utils
################
def define_net(opt, k):
    net = None
    act = define_act_layer(act_type=opt.act_type)
    init_max = True if opt.init_type == "max" else False

    if opt.mode == "graph":
        net = GraphNetMultiCls(features= 517, grph_dim=opt.grph_dim, dropout_rate=opt.dropout_rate, GNN=GCNConv, pooling_ratio=opt.pooling_ratio, num_classes=3, act=act, init_max=init_max)
    else:
        raise NotImplementedError('model [%s] is not implemented' % opt.model)
    return init_net(net, opt.init_type, opt.init_gain, opt.gpu_ids)

def define_optimizer(opt, model):
    optimizer = None
    if opt.optimizer_type == 'adabound':
        optimizer = adabound.AdaBound(model.parameters(), lr=opt.lr, final_lr=opt.final_lr)
    elif opt.optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=opt.weight_decay)
    elif opt.optimizer_type == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay, initial_accumulator_value=0.1)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % opt.optimizer)
    return optimizer


def define_reg(opt, model):
    loss_reg = None
    
    if opt.reg_type == 'none':
        loss_reg = 0
    elif opt.reg_type == 'path':
        loss_reg = regularize_path_weights(model=model)
    elif opt.reg_type == 'mm':
        loss_reg = regularize_MM_weights(model=model)
    elif opt.reg_type == 'all':
        loss_reg = regularize_weights(model=model)
    elif opt.reg_type == 'omic':
        loss_reg = regularize_MM_omic(model=model)
    else:
        raise NotImplementedError('reg method [%s] is not implemented' % opt.reg_type)
    return loss_reg


def define_scheduler(opt, optimizer):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'exp':
        scheduler = lr_scheduler.ExponentialLR(optimizer, 0.1, last_epoch=-1)
    elif opt.lr_policy == 'step':
       scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
       scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
       scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
       return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def define_act_layer(act_type='Tanh'):
    if act_type == 'Tanh':
        act_layer = nn.Tanh()
    elif act_type == 'ReLU':
        act_layer = nn.ReLU()
    elif act_type == 'Sigmoid':
        act_layer = nn.Sigmoid()
    elif act_type == 'LSM':
        act_layer = nn.LogSoftmax(dim=1)
    elif act_type == "none":
        act_layer = None
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act_type)
    return act_layer


def define_bifusion(fusion_type, skip=1, use_bilinear=1, gate1=1, gate2=1, dim1=32, dim2=32, scale_dim1=1, scale_dim2=1, mmhid=64, dropout_rate=0.25):
    fusion = None
    if fusion_type == 'pofusion':
        fusion = BilinearFusion(skip=skip, use_bilinear=use_bilinear, gate1=gate1, gate2=gate2, dim1=dim1, dim2=dim2, scale_dim1=scale_dim1, scale_dim2=scale_dim2, mmhid=mmhid, dropout_rate=dropout_rate)
    else:
        raise NotImplementedError('fusion type [%s] is not found' % fusion_type)
    return fusion


def define_trifusion(fusion_type, skip=1, use_bilinear=1, gate1=1, gate2=1, gate3=3, dim1=32, dim2=32, dim3=32, scale_dim1=1, scale_dim2=1, scale_dim3=1, mmhid=96, dropout_rate=0.25):
    fusion = None
    if fusion_type == 'pofusion_A':
        fusion = TrilinearFusion_A(skip=skip, use_bilinear=use_bilinear, gate1=gate1, gate2=gate2, gate3=gate3, dim1=dim1, dim2=dim2, dim3=dim3, scale_dim1=scale_dim1, scale_dim2=scale_dim2, scale_dim3=scale_dim3, mmhid=mmhid, dropout_rate=dropout_rate)
    elif fusion_type == 'pofusion_B':
        fusion = TrilinearFusion_B(skip=skip, use_bilinear=use_bilinear, gate1=gate1, gate2=gate2, gate3=gate3, dim1=dim1, dim2=dim2, dim3=dim3, scale_dim1=scale_dim1, scale_dim2=scale_dim2, scale_dim3=scale_dim3, mmhid=mmhid, dropout_rate=dropout_rate)
    else:
        raise NotImplementedError('fusion type [%s] is not found' % fusion_type)
    return fusion


class GraphNetMultiCls(torch.nn.Module):
    def __init__(self, features, nhid=128, grph_dim=32, dropout_rate=0.25,
                 GNN=GCNConv, pooling_ratio=0.20, num_classes=3, init_max=True, act=None):
        """
        Parameter explanation:
            features: Node feature dimension (input dimension)
            nhid: Hidden layer dimension
            grph_dim: Low-dimensional graph representation dimension
            dropout_rate: Dropout rate
            GNN: Type of graph convolution model (e.g., 'GCN' or 'GraphSAGE')
            pooling_ratio: Pooling ratio
            num_classes: Number of classification categories
            act: Activation function for the final output, optional
        """
        super(GraphNetMultiCls, self).__init__()

        self.dropout_rate = dropout_rate
        self.act = act

        # Define graph convolution layers and pooling layers
        self.conv1 = SAGEConv(features, nhid)
        self.pool1 = SAGPooling(nhid, ratio=pooling_ratio, GNN=GNN)
        self.conv2 = SAGEConv(nhid, nhid)
        self.pool2 = SAGPooling(nhid, ratio=pooling_ratio, GNN=GNN)
        self.conv3 = SAGEConv(nhid, nhid)
        self.pool3 = SAGPooling(nhid, ratio=pooling_ratio, GNN=GNN)

        # Two types of global pooling (max and average), output dimension is nhid*2
        self.lin1 = nn.Linear(nhid * 2, nhid)
        self.lin2 = nn.Linear(nhid, grph_dim)
        self.lin3 = nn.Linear(grph_dim, num_classes)

        # If fixed output range is needed (e.g., for regression scenarios), set here
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

        if init_max:
            # You need to implement or import the init_max_weights function
            init_max_weights(self)
            print("Initializing with Max")

    def forward(self, data):
        """
        data: torch_geometric.data.Data object, requires data.x, data.edge_index, data.edge_attr, data.batch
        """
        # If you have preprocessing steps, you can call them here, e.g., NormalizeFeaturesV2, NormalizeEdgesV2
        # data = NormalizeFeaturesV2()(data)
        # data = NormalizeEdgesV2()(data)
        
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # 1st convolution + pooling layer
        x = F.relu(self.conv1(x, edge_index))
        result = self.pool1(x, edge_index, edge_attr, batch)
        if len(result) == 6:
            x, edge_index, edge_attr, batch, perm, score = result
        elif len(result) == 5:
            x, edge_index, edge_attr, batch, perm = result
        elif len(result) == 3:
            x, edge_index, batch = result
        else:
            raise ValueError("Unexpected number of return values from SAGPooling")
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # 2nd convolution + pooling layer
        x = F.relu(self.conv2(x, edge_index))
        result = self.pool2(x, edge_index, edge_attr, batch)
        if len(result) == 6:
            x, edge_index, edge_attr, batch, perm, score = result
        elif len(result) == 5:
            x, edge_index, edge_attr, batch, perm = result
        elif len(result) == 3:
            x, edge_index, batch = result
        else:
            raise ValueError("Unexpected number of return values from SAGPooling")
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # 3rd convolution + pooling layer
        x = F.relu(self.conv3(x, edge_index))
        result = self.pool3(x, edge_index, edge_attr, batch)
        if len(result) == 6:
            x, edge_index, edge_attr, batch, perm, score = result
        elif len(result) == 5:
            x, edge_index, edge_attr, batch, perm = result
        elif len(result) == 3:
            x, edge_index, batch = result
        else:
            raise ValueError("Unexpected number of return values from SAGPooling")
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # Aggregate global information from three layers
        x = x1 + x2 + x3

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        features = F.relu(self.lin2(x))
        out = self.lin3(features)

        if self.act is not None:
            out = self.act(out)
            # If using Sigmoid and need to adjust the output range, transform as needed
            if isinstance(self.act, nn.Sigmoid):
                out = out * self.output_range + self.output_shift

        return features, out


