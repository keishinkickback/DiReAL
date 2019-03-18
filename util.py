from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def divReg_loss(model, threshold=0.6):
    def cos_sim_matrix(matrix):
        d = torch.mm(matrix, matrix.t())
        norm = (matrix * matrix).sum(dim=1, keepdim=True) ** .5
        return d / norm / norm.t()
    loss = 0.0
    for ly in model.parameters():
        dim = ly.size()
        if len(dim) == 4 and dim[1] != 3:
            weight_matrix = ly.view(dim[0], -1)
            sim_matrix = cos_sim_matrix(weight_matrix) # similatity
            mask = sim_matrix - torch.eye(sim_matrix.size(0)).to(sim_matrix.device) # 0s in diagonal
            mask = mask * (threshold < mask.abs() ).float()
            loss += torch.sum(mask ** 2)
    return loss


# def adjust_learning_rate(optimizer, epoch):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     if epoch != 0 and epoch % 20 == 0:
#         for param_group in optimizer.param_groups:
#             param_group['lr'] *= 0.5
#
# def div_scheduler(threshold, epoch):
#     if epoch % 10 == 0 and epoch !=0:
#         threshold *= 1.1
#     return min(threshold, 0.5)
#
# def switcher_scheduler(threshold, epoch):
#     if epoch % 10 == 0 and epoch !=0:
#         threshold *= 0.8
#     return threshold


