from __future__ import print_function
import argparse
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3
import torchvision.transforms as transforms

import numpy as np
from scipy.stats import entropy

from models import Generator

def inception_score(imgs, cuda=True, batch_size=32, resize=True, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear',align_corners=False).type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x,dim=1).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batch)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


def test(opt):
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
    import torchvision.datasets as dset
    import torchvision.transforms as transforms

    cifar = dset.CIFAR10(root='.', download=True,
                             transform=transforms.Compose([
                                 transforms.Scale(32),
                                 transforms.ToTensor(),
                                 # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                 transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                             ])
    )

    # IgnoreLabelDataset(cifar, opt.n_images)

    print ("Calculating Inception Score...")
    print (inception_score(IgnoreLabelDataset(cifar, opt.n_images), cuda=opt.cuda, batch_size=opt.batchSize, resize=True, splits=1))


class gan_Dataset(torch.utils.data.Dataset):
    def __init__(self, model, nz=100, cuda=True, n_images=1024):
        self.model = model.eval()
        self.nz = nz
        self.noise = torch.FloatTensor(1, self.nz, 1, 1)
        self.n_images = n_images
        self.cuda = cuda
        if self.cuda:
            self.model.cuda()
            self.noise = self.noise.cuda()

        self.transforms=transforms.Compose([
                             # transforms.ToPILImage(),
                             # transforms.Resize(32),
                             # transforms.ToTensor(),
                             # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                         ])

    def __getitem__(self, index):
        self.noise.resize_(1, self.nz, 1, 1).normal_(0, 1)
        fake = self.model(self.noise)
        fake  = fake.cpu().data.squeeze()
        fake = (fake + 1.0) / 2.0
        # return fake
        return self.transforms(fake)

    def __len__(self):
        return self.n_images

def calc_inception_score_from_state_dict(opt):
    nz = opt.nz
    cuda = opt.cuda

    netG = Generator(nz)
    netG.load_state_dict(torch.load(opt.netG))
    print(netG)
    print("Calculating Inception Score...")
    print(
        inception_score(gan_Dataset(netG,nz=opt.nz,cuda=opt.cuda,n_images=opt.n_images),
        cuda=opt.cuda, batch_size=opt.batchSize, resize=True, splits=10))

def calc_inception_score(netG, nz, cuda=True, n_images=1024, batchSize=32):
    print("Calculating Inception Score...")
    print(
        inception_score(gan_Dataset(netG, nz=nz,cuda=cuda, n_images=n_images),
            batch_size=batchSize, resize=True, splits=10))



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='test inception score of cifar10')
    parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=128, help='size of the latent z vector')
    parser.add_argument('--cuda', action='store_true', help='enables cuda', default=True)
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--n_images', default=4096, help="number of generating images")
    opt = parser.parse_args()
    print(opt)
    return opt


if __name__ == '__main__':
    opt = get_args()
    opt.cuda = True
    opt.netG = '/home/keishin/sandbox/pytorch-spectral-normalization-gan/checkpoints_dcgan_divreg/gen_100'
    if opt.test:
        test(opt)
    else:
        calc_inception_score_from_state_dict(opt)


