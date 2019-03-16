# DCGAN-like generator and discriminator
from torch import nn
from spectral_normalization import SpectralNorm

class Generator(nn.Module):
    def __init__(self, z_dim, n_out_channels=3):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.n_out_channels = n_out_channels
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
            nn.ConvTranspose2d(64, self.n_out_channels, 3, stride=1, padding=(1,1)),
            nn.Tanh())
    def forward(self, z):
        return self.model(z.view(-1, self.z_dim, 1, 1))

class Discriminator(nn.Module):
    def __init__(self, n_channerls=3, leak = 0.1, w_g = 4):
        super(Discriminator, self).__init__()
        self.n_channels = n_channerls
        self.leak = leak
        self.w_g = w_g
        self.conv1 = SpectralNorm(nn.Conv2d(self.n_channels, 64, 3, stride=1, padding=(1,1)))
        self.conv2 = SpectralNorm(nn.Conv2d(64, 64, 4, stride=2, padding=(1,1)))
        self.conv3 = SpectralNorm(nn.Conv2d(64, 128, 3, stride=1, padding=(1,1)))
        self.conv4 = SpectralNorm(nn.Conv2d(128, 128, 4, stride=2, padding=(1,1)))
        self.conv5 = SpectralNorm(nn.Conv2d(128, 256, 3, stride=1, padding=(1,1)))
        self.conv6 = SpectralNorm(nn.Conv2d(256, 256, 4, stride=2, padding=(1,1)))
        self.conv7 = SpectralNorm(nn.Conv2d(256, 512, 3, stride=1, padding=(1,1)))
        self.fc = SpectralNorm(nn.Linear(self.w_g * self.w_g * 512, 1))

    def forward(self, x):
        m = nn.LeakyReLU(self.leak)(self.conv1(x))
        m = nn.LeakyReLU(self.leak)(self.conv2(m))
        m = nn.LeakyReLU(self.leak)(self.conv3(m))
        m = nn.LeakyReLU(self.leak)(self.conv4(m))
        m = nn.LeakyReLU(self.leak)(self.conv5(m))
        m = nn.LeakyReLU(self.leak)(self.conv6(m))
        m = nn.LeakyReLU(self.leak)(self.conv7(m))
        return self.fc(m.view(-1,self.w_g * self.w_g * 512))