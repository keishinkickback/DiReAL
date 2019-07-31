# DCGAN-like generator and discriminator
from torch import nn
from torch.nn.modules import normalization as Norm
from torch.nn.utils.weight_norm import weight_norm

from spectral_normalization import SpectralNorm

class Generator(nn.Module):

    def __init__(self, z_dim, nc=3):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.model = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 512, 4, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, nc, 3, stride=1, padding=(1,1)),
            nn.Tanh())

    def forward(self, z):
        return self.model(z.view(-1, self.z_dim, 1, 1))


class Discriminator_plain(nn.Module):

    def __init__(self, nc=3, ndf=64):
        super(Discriminator_plain, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 1, bias=False)
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1)


class Discriminator_BN(Discriminator_plain):
    def __init__(self, nc=3, ndf=64):
        super(Discriminator_BN, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 1, bias=False)
        )


class Discriminator_LN(Discriminator_plain):
    def __init__(self, nc=3, ndf=64):
        super(Discriminator_LN, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            Norm.LayerNorm((ndf * 2, 8, 8)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            Norm.LayerNorm((ndf * 4, 4, 4)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            Norm.LayerNorm((ndf * 8, 2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 1, bias=False),
        )


class Discriminator_WN(Discriminator_plain):
    def __init__(self, nc=3, ndf=64):
        super(Discriminator_WN, self).__init__()
        self.main = nn.Sequential(
            weight_norm(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            weight_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            weight_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            weight_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            weight_norm(nn.Conv2d(ndf * 8, 1, 4, 1, 1, bias=False))
        )

class Discriminator_SN(nn.Module):

    def __init__(self, nc=3, leak = 0.1, w_g = 4):
        super(Discriminator_SN, self).__init__()
        self.w_g = w_g
        self.main = nn.Sequential(
            SpectralNorm(nn.Conv2d(nc, 64, 3, stride=1, padding=(1,1))),
            nn.LeakyReLU(leak, inplace=True),
            SpectralNorm(nn.Conv2d(64, 64, 4, stride=2, padding=(1,1))),
            nn.LeakyReLU(leak, inplace=True),
            SpectralNorm(nn.Conv2d(64, 128, 3, stride=1, padding=(1,1))),
            nn.LeakyReLU(leak, inplace=True),
            SpectralNorm(nn.Conv2d(128, 128, 4, stride=2, padding=(1,1))),
            nn.LeakyReLU(leak, inplace=True),
            SpectralNorm(nn.Conv2d(128, 256, 3, stride=1, padding=(1,1))),
            nn.LeakyReLU(leak, inplace=True),
            SpectralNorm(nn.Conv2d(256, 256, 4, stride=2, padding=(1,1))),
            nn.LeakyReLU(leak, inplace=True),
            SpectralNorm(nn.Conv2d(256, 512, 3, stride=1, padding=(1,1))),
            nn.LeakyReLU(leak, inplace=True)
        )
        self.fc = SpectralNorm(nn.Linear(4 * 4* 512, 1))

    def forward(self, x):
        m = self.main(x)
        return self.fc(m.view(-1, 4 * 4 * 512))


if __name__ == '__main__':
    import torch
    z = torch.rand(128)
    g = Generator(128)
    x = g(z)

    lst_Ds = [Discriminator_plain, Discriminator_BN, Discriminator_LN, Discriminator_WN, Discriminator_SN]

    for ds in lst_Ds:
        d = ds()
        y = d(x)
        print(ds.__name__,'z',z.size(), 'x', x.size(), 'y', y.size())
