import torch
import torch.nn as nn
import torch.nn.functional as F

class doubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(doubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.enc1 = doubleConv(in_channels, 64)
        self.enc2 = doubleConv(64, 128)
        self.enc3 = doubleConv(128, 256)
        self.enc4 = doubleConv(256, 512)

        self.bottleneck = doubleConv(512, 1024)

        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec1 = doubleConv(1024, 512)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = doubleConv(512, 256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = doubleConv(256, 128)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = doubleConv(128, 64)

        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d1 = self.upconv1(b)
        d1 = self.dec1(torch.cat((e4, d1), dim=1))

        d2 = self.upconv2(d1)
        d2 = self.dec2(torch.cat((e3, d2), dim=1))

        d3 = self.upconv3(d2)
        d3 = self.dec3(torch.cat((e2, d3), dim=1))

        d4 = self.upconv4(d3)
        d4 = self.dec4(torch.cat((e1, d4), dim=1))

        out = self.output(d4)
        return out