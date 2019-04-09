import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

import models
import models_resnet

from util import divReg_loss

def train(epoch):
    for batch_idx, (data, target) in enumerate(loader):
        if data.size()[0] != args.batch_size:
            continue
        data, target = data.to(device=args.device), target.to(device=args.device)

        # weight clipping
        if args.use_clamp:
            # clamp parameters to a cube
            for p in discriminator.parameters():
                p.data.clamp_(args.clamp_lower, args.clamp_upper)

        for _ in range(disc_iters):
            z = torch.randn(args.batch_size, Z_dim).to(device=args.device)
            optim_disc.zero_grad()
            optim_gen.zero_grad()
            if args.loss == 'hinge':
                disc_loss = nn.ReLU()(1.0 - discriminator(data)).mean() + nn.ReLU()(
                    1.0 + discriminator(generator(z))).mean()
            elif args.loss == 'wasserstein':
                disc_loss = -discriminator(data).mean() + discriminator(generator(z)).mean()
            else:
                disc_loss = nn.BCEWithLogitsLoss()(discriminator(data), torch.ones(args.batch_size, 1).cuda()) + \
                            nn.BCEWithLogitsLoss()(discriminator(generator(z)), torch.zeros(args.batch_size, 1).cuda())
            if args.divreg:
                disc_loss += divReg_loss(discriminator, 0.6).cuda()
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
        gen_loss.backward()
        optim_gen.step()

        if batch_idx % 100 == 0:
            print('epoch:', epoch, 'batch:', batch_idx, 'disc loss', disc_loss.data[0], 'gen loss', gen_loss.data[0])
    scheduler_d.step()
    scheduler_g.step()





def evaluate(epoch):
    samples = generator(fixed_z).cpu().data.numpy()[:64]

    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.transpose((1, 2, 0)) * 0.5 + 0.5)

    if not os.path.exists('out/'):
        os.makedirs('out/')

    plt.savefig('out/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
    plt.close(fig)



if __name__ == "__main__":

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
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    args.device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    print(args)


    dataset_cifar10 = datasets.CIFAR10('../data/', train=True, download=True,
                                       transform=transforms.Compose([ transforms.ToTensor(),
                                                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                                       )
    loader = torch.utils.data.DataLoader( dataset_cifar10, batch_size=args.batch_size, shuffle=True, num_workers=1,
                                          pin_memory=True)

    Z_dim = 128
    # number of updates to discriminator for every update to generator
    disc_iters = 5


    if args.model.lower() == 'plain':
        discriminator = models.Discriminator_plain()
    elif args.model.lower() == 'bn':
        discriminator = models.Discriminator_BN()
    elif args.model.lower() == 'ln':
        discriminator = models.Discriminator_LN()
    elif args.model.lower() == 'wn':
        discriminator = models.Discriminator_WN()
    # spectral norm
    elif args.model == 'resnet':
        discriminator = models_resnet.Discriminator()
    else:
        discriminator = models.Discriminator()

    if args.model.lower() == 'resnet':
        generator = models_resnet.Generator(Z_dim)
    else:
        generator = models.Generator(Z_dim)

    discriminator = discriminator.to(device=args.device)
    generator = generator.to(device=args.device)
    print(discriminator)
    print(generator)

    # because the spectral normalization module creates parameters that don't require gradients (u and v), we don't want to
    # optimize these using sgd. We only let the optimizer operate on parameters that _do_ require gradients
    # TODO: replace Parameters with buffers, which aren't returned from .parameters() method.
    optim_disc = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr, betas=(0.0, 0.9))
    optim_gen = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0, 0.9))

    # use an exponentially decaying learning rate
    scheduler_d = optim.lr_scheduler.ExponentialLR(optim_disc, gamma=0.99)
    scheduler_g = optim.lr_scheduler.ExponentialLR(optim_gen, gamma=0.99)

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    fixed_z = torch.randn(args.batch_size, Z_dim).cuda()

    for epoch in range(2000):
        train(epoch)
        evaluate(epoch)
        torch.save(discriminator.state_dict(), os.path.join(args.checkpoint_dir, 'disc_{}'.format(epoch)))
        torch.save(generator.state_dict(), os.path.join(args.checkpoint_dir, 'gen_{}'.format(epoch)))
