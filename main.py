import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.utils as vutils

import models
import models_resnet

from util import direal_loss

def train(epoch):

    for batch_idx, (data, target) in enumerate(loader):
        if data.size()[0] != args.batch_size:
            continue
        data, target = data.to(device=args.device), target.to(device=args.device)

        # weight clipping
        if args.use_clamp:
            for p in discriminator.parameters():
                p.data.clamp_(args.clamp_lower, args.clamp_upper)

        for _ in range(args.n_dupdates):
            z = torch.randn(args.batch_size, args.nz).to(device=args.device)
            optim_disc.zero_grad()
            optim_gen.zero_grad()

            if args.loss == 'hinge':
                disc_loss = nn.ReLU()(1.0 - discriminator(data)).mean() + nn.ReLU()(1.0 + discriminator(generator(z))).mean()
            elif args.loss == 'wasserstein':
                disc_loss = -discriminator(data).mean() + discriminator(generator(z)).mean()
            else:
                disc_loss = nn.BCEWithLogitsLoss()(discriminator(data), torch.ones(args.batch_size, 1).to(device=args.device))

                disc_loss += nn.BCEWithLogitsLoss()(discriminator(generator(z)), torch.zeros(args.batch_size, 1).to(device=args.device))
            if args.divreg:
                disc_loss += direal_loss(discriminator, 0.6).to(device=args.device)
            disc_loss.backward()
            optim_disc.step()

        z = torch.randn(args.batch_size, args.nz).cuda()

        # update generator
        optim_disc.zero_grad()
        optim_gen.zero_grad()
        if args.loss == 'hinge' or args.loss == 'wasserstein':
            gen_loss = -discriminator(generator(z)).mean()
        else:
            gen_loss = nn.BCEWithLogitsLoss()(discriminator(generator(z)), torch.ones(args.batch_size, 1).to(device=args.device))
        gen_loss.backward()
        optim_gen.step()

        if batch_idx % 100 == 0:
            print('epoch:', epoch, 'batch:', batch_idx, 'disc loss', disc_loss.data[0], 'gen loss', gen_loss.data[0])
            vutils.save_image(real_cpu, '%s/real_samples.png' % opt.outf, normalize=True)
            fake = netG(fixed_noise)
            vutils.save_image(fake.detach(),
                              '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                              normalize=True)

    scheduler_d.step()
    scheduler_g.step()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--loss', type=str, default='bce')
    parser.add_argument('--direal', action='store_true', default=True)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--discriminator', type=str, default='resnet_not', help='plain, bn, ln, or wn')
    parser.add_argument('--generator', type=str, default='resnet', help='plain, resnet')
    parser.add_argument('--nz', type=int, default=128, help='size of the latent z vector')
    parser.add_argument('--use_clamp', action='store_true', default=False)
    parser.add_argument('--clamp_upper', type=float, default=1.0)
    parser.add_argument('--clamp_lower', type=float, default=-1.0)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--n_dupdates', default=5,type=int, help='number of updates to discriminator for every update to generator')
    args = parser.parse_args()
    args.device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else 'cpu')
    print(args)


    dataset_cifar10 = datasets.CIFAR10('../data/', train=True, download=True,
                                       transform=transforms.Compose([ transforms.ToTensor(),
                                                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                                       )
    loader = torch.utils.data.DataLoader( dataset_cifar10, batch_size=args.batch_size, shuffle=True, num_workers=1,
                                          pin_memory=True)

    # discriminator
    if args.discriminator.lower() == 'plain':
        discriminator = models.Discriminator_plain()
    elif args.discriminator.lower() == 'bn':
        discriminator = models.Discriminator_BN()
    elif args.discriminator.lower() == 'ln':
        discriminator = models.Discriminator_LN()
    elif args.discriminator.lower() == 'wn':
        discriminator = models.Discriminator_WN()
    elif args.discriminator.lower() == 'sn':
        discriminator = models.Discriminator_SN()
    elif args.discriminator.lower() == 'resnet':
        discriminator = models_resnet.Discriminator()

    #generator
    if args.generator.lower() == 'resnet':
        generator = models_resnet.Generator(args.nz)
    elif args.generator.lower() == 'plain':
        generator = models.Generator(args.nz)

    assert discriminator
    assert generator

    discriminator = discriminator.to(device=args.device)
    generator = generator.to(device=args.device)
    print(discriminator)
    print(generator)

    optim_disc = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr, betas=(0.0, 0.9))
    optim_gen = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0, 0.9))

    scheduler_d = optim.lr_scheduler.ExponentialLR(optim_disc, gamma=0.99)
    scheduler_g = optim.lr_scheduler.ExponentialLR(optim_gen, gamma=0.99)

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    fixed_z = torch.randn(args.batch_size, args.nz).cuda()

    for epoch in range(200):
        train(epoch)
        torch.save(discriminator.state_dict(), os.path.join(args.checkpoint_dir, 'disc_{}'.format(epoch)))
        torch.save(generator.state_dict(), os.path.join(args.checkpoint_dir, 'gen_{}'.format(epoch)))
