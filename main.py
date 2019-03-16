import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import datasets, transforms
from torch.autograd import Variable
import model_resnet
import model

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from util import divReg_loss

from inception_score import inception_score, calc_inception_score


class IgnoreLabelDataset(torch.utils.data.Dataset):
    def __init__(self, orig, size=None):
        self.orig = orig

        self.size = size
        if self.size is None:
            self.size = len(self.orig)

    def __getitem__(self, index):
        return self.orig[index][0]

    def __len__(self):
        # return  self.size
        return 1024


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--loss', type=str, default='bce')
parser.add_argument('--divreg', action='store_true', default=True)
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')

parser.add_argument('--model', type=str, default='resnet_not', help='plain, bn, ln, or wn')
parser.add_argument('--use_clamp', action='store_true', default=False)
parser.add_argument('--clamp_upper', type=float, default=1.0)
parser.add_argument('--clamp_lower', type=float, default=-1.0)

args = parser.parse_args()
print(args)

loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data/', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
        batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)

Z_dim = 128
#number of updates to discriminator for every update to generator 
disc_iters = 1

# discriminator = torch.nn.DataParallel(Discriminator()).cuda() # TODO: try out multi-gpu training
if args.model.lower() == 'plain':
    discriminator = model.Discriminator_plain().cuda()
    generator = model.Generator(Z_dim).cuda()
elif args.model.lower() == 'bn':
    discriminator = model.Discriminator_BN().cuda()
    generator = model.Generator(Z_dim).cuda()
elif args.model.lower() == 'ln':
    discriminator = model.Discriminator_LN().cuda()
    generator = model.Generator(Z_dim).cuda()
elif args.model.lower() == 'wn':
    discriminator = model.Discriminator_WN().cuda()
    generator = model.Generator(Z_dim).cuda()

# spectral norm
elif args.model == 'resnet':
    discriminator = model_resnet.Discriminator().cuda()
    generator = model_resnet.Generator(Z_dim).cuda()
else:
    discriminator = model.Discriminator().cuda()
    generator = model.Generator(Z_dim).cuda()

print(discriminator)

# because the spectral normalization module creates parameters that don't require gradients (u and v), we don't want to 
# optimize these using sgd. We only let the optimizer operate on parameters that _do_ require gradients
# TODO: replace Parameters with buffers, which aren't returned from .parameters() method.
optim_disc = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr, betas=(0.0,0.9))
optim_gen  = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0,0.9))

# use an exponentially decaying learning rate
scheduler_d = optim.lr_scheduler.ExponentialLR(optim_disc, gamma=0.99)
scheduler_g = optim.lr_scheduler.ExponentialLR(optim_gen, gamma=0.99)

def train(epoch):
    for batch_idx, (data, target) in enumerate(loader):
        if data.size()[0] != args.batch_size:
            continue
        data, target = data.cuda(), target.cuda()

        # print('input size', data.size())

        # update discriminator

        # weight clipping
        if args.use_clamp:
            # clamp parameters to a cube
            for p in discriminator.parameters():
                p.data.clamp_(args.clamp_lower, args.clamp_upper)

        for _ in range(disc_iters):
            z = torch.randn(args.batch_size, Z_dim).cuda()
            optim_disc.zero_grad()
            optim_gen.zero_grad()
            if args.loss == 'hinge':
                disc_loss = nn.ReLU()(1.0 - discriminator(data)).mean() + nn.ReLU()(1.0 + discriminator(generator(z))).mean()
            elif args.loss == 'wasserstein':
                disc_loss = -discriminator(data).mean() + discriminator(generator(z)).mean()
            else:
                disc_loss = nn.BCEWithLogitsLoss()(discriminator(data), torch.ones(args.batch_size, 1).cuda()) + \
                    nn.BCEWithLogitsLoss()(discriminator(generator(z)), torch.zeros(args.batch_size, 1).cuda())
            if args.divreg:
                # print('Im here')
                disc_loss += 0.1*divReg_loss(discriminator, 0.5).cuda()
            disc_loss.backward()
            optim_disc.step()

        z = torch.randn(args.batch_size, Z_dim).cuda()

        # update generator
        optim_disc.zero_grad()
        optim_gen.zero_grad()
        if args.loss == 'hinge' or args.loss == 'wasserstein':
            gen_loss = -discriminator(generator(z)).mean()
        else:
            gen_loss = nn.BCEWithLogitsLoss()(discriminator(generator(z)), torch.ones(args.batch_size, 1).cuda())

        if args.divreg:
            gen_loss += 0.001*divReg_loss(generator, 0.5).cuda()
            
        gen_loss.backward()
        optim_gen.step()

        if batch_idx % 100 == 0:
            print('epoch:', epoch, 'batch:', batch_idx, 'disc loss', disc_loss.data[0], 'gen loss', gen_loss.data[0])
    scheduler_d.step()
    scheduler_g.step()

fixed_z = torch.randn(args.batch_size, Z_dim).cuda()
fixed_z2 = torch.randn(5000, Z_dim).cuda()

def evaluate(epoch):

    samples = generator(fixed_z).cpu().data.numpy()[:64]
    samples2 = generator(fixed_z).cpu().data.numpy()

    ## Inception score
    # # inception score
    # netG.eval()
    im, istd = inception_score(samples2, cuda=True, batch_size=32, splits=1)
    # im, istd = calc_inception_score(generator, opt.nz, cuda=True, n_images=4096, batchSize=32)
    print("inception score: %.4f  (%.4f )" % (im, istd))


    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.transpose((1,2,0)) * 0.5 + 0.5)

    if not os.path.exists('out/'):
        os.makedirs('out/')

    plt.savefig('out/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
    plt.close(fig)

    return im, istd

if not os.path.exists(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)

inception_score_epoch = []
inception_score_epoch_std = []
for epoch in range(200):
    train(epoch)
    im, istd = evaluate(epoch)
    inception_score_epoch.append(im)
    inception_score_epoch_std.append(istd)
    torch.save(discriminator.state_dict(), os.path.join(args.checkpoint_dir, 'disc_{}'.format(epoch)))
    torch.save(generator.state_dict(), os.path.join(args.checkpoint_dir, 'gen_{}'.format(epoch)))

idx = np.argmax(inception_score_epoch)
print('epoch with max inception score is:', idx)
print('Max inception score', inception_score_epoch[idx])
print('STD', inception_score_epoch_std[idx])


import pickle

with open('distance_resnet_cifar_sn', 'wb') as fp:
    pickle.dump(inception_score_epoch, fp)
with open('distance_resnet_cifar_sn_std', 'wb') as fp:
    pickle.dump(inception_score_epoch_std, fp)

dataset = datasets.CIFAR10('../data/', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

im, istd = inception_score(IgnoreLabelDataset(dataset), cuda=True, batch_size=32)
inception_score_epoch.append((im, istd))
print("inception score for test data: %.4f  (%.4f )" % (im, istd))

