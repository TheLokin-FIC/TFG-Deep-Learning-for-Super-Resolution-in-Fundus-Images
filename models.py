import math
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )

    def forward(self, x):
        return x + self.net(x)


class UpsampleBlock(nn.Module):
    def __init__(self, channels, scale_factor):
        super(UpsampleBlock, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(channels, channels * (scale_factor ** 2),
                      kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.PReLU()
        )

    def forward(self, x):
        return self.net(x)


class Generator(nn.Module):
    def __init__(self, n_residual_blocks, upsample_factor):
        super(Generator, self).__init__()

        self.n_residual_blocks = n_residual_blocks
        self.upsample_factor = upsample_factor

        # First convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )

        # Residual blocks
        for i in range(n_residual_blocks):
            self.add_module("residual" + str(i + 1), ResidualBlock(64))

        # Second convolutional layer post residual blocks
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )

        # Upsampling layers
        for i in range(int(math.log(upsample_factor, 2))):
            self.add_module("upsample" + str(i + 1), UpsampleBlock(64, 2))

        # Third convolutional layer post upsampling blocks
        self.conv3 = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        out = self.conv1(x)
        cache = out.clone()

        for i in range(self.n_residual_blocks):
            out = self.__getattr__("residual" + str(i + 1))(out)

        out = self.conv2(out) + cache

        for i in range(int(math.log(self.upsample_factor, 2))):
            out = self.__getattr__("upsample" + str(i + 1))(out)

        out = self.conv3(out)

        return (out.tanh() + 1) / 2


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        return (self.net(x).view(x.size(0))).sigmoid()
