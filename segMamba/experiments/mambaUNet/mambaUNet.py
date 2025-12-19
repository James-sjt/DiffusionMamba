import torch
import torch.nn as nn
from core import VSSBlock

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def default_init(m):
    if isinstance(m, nn.Conv2d):
        if m.kernel_size == (1, 1):  # projection conv
            nn.init.xavier_uniform_(m.weight)
        else:
            nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in", nonlinearity="leaky_relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class ResBlockT(nn.Module):
    def __init__(self, in_ch, out_ch=None, groups=8, use_conv_shortcut=False):
        super().__init__()
        out_ch = out_ch or in_ch
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        if in_ch != out_ch:
            if use_conv_shortcut:
                self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1)
            else:
                self.shortcut = nn.Sequential(
                    nn.GroupNorm(groups, in_ch),
                    nn.SiLU(),
                    nn.Conv2d(in_ch, out_ch, kernel_size=1)
                )
        else:
            self.shortcut = nn.Identity()

        default_init(self)

    def forward(self, x):
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)

        return self.shortcut(x) + h

class MidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, depth=2):
        super().__init__()
        self.depth = depth
        self.in_res = ResBlockT(in_channels, out_channels)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleDict({
                "VSSBlock1": VSSBlock(hidden_dim=out_channels),
                "VSSBlock2": VSSBlock(hidden_dim=out_channels),
                "res": ResBlockT(out_channels, out_channels)
            }))

        default_init(self.in_res)
        for layer in self.layers:
            default_init(layer)

    def forward(self, x):
        h = self.in_res(x)

        for layer in self.layers:
            h = h.permute(0, 2, 3, 1)
            h = layer["VSSBlock1"](h)
            h = layer["VSSBlock2"](h)
            h = h.permute(0, 3, 1, 2)
            h = layer["res"](h)

        return h

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, depth=1):
        super().__init__()
        self.depth = depth
        self.in_res = ResBlockT(in_channels, out_channels)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleDict({
                "VSSBlock1": VSSBlock(hidden_dim=out_channels),
                "VSSBlock2": VSSBlock(hidden_dim=out_channels),
                "res": ResBlockT(out_channels, out_channels)
            }))
        self.pool = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        default_init(self.in_res)
        for layer in self.layers:
            default_init(layer)

    def forward(self, x):
        h = self.in_res(x)

        for layer in self.layers:
            h = h.permute(0, 2, 3, 1)
            h = layer["VSSBlock1"](h)
            h = layer["VSSBlock2"](h)
            h = h.permute(0, 3, 1, 2)
            h = layer["res"](h)

        return self.pool(h), h

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, depth=1):
        super().__init__()
        self.depth = depth
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),  # Nearest-neighbor upsampling
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)  # Learnable refinement
        )
        self.in_res = ResBlockT(in_channels, out_channels)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleDict({
                "VSSBlock1": VSSBlock(hidden_dim=out_channels),
                "VSSBlock2": VSSBlock(hidden_dim=out_channels),
                "res": ResBlockT(out_channels, out_channels)
            }))
        default_init(self.in_res)
        for layer in self.layers:
            default_init(layer)

    def forward(self, x, skip):
        x = self.up(x)

        x = torch.cat([x, skip], dim=1)
        h = self.in_res(x)  # expand channels

        for layer in self.layers:
            h = h.permute(0, 2, 3, 1)
            h = layer["VSSBlock1"](h)
            h = layer["VSSBlock2"](h)
            h = h.permute(0, 3, 1, 2)
            h = layer["res"](h)
        return h

class mambaUNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=32):  # cond_dim=512
        super().__init__()
        self.in_conv = nn.Sequential(nn.Conv2d(in_ch, base_ch // 2, kernel_size=3, stride=2, padding=1),
                                     nn.SiLU(),
                                     nn.Conv2d(base_ch // 2, base_ch, kernel_size=3, stride=2, padding=1),
                                     )
        # down path
        self.down1 = DownBlock(base_ch, base_ch)
        self.down2 = DownBlock(base_ch, base_ch * 2)
        self.down3 = DownBlock(base_ch * 2, base_ch * 4)

        # middle
        self.mid = MidBlock(base_ch * 4, base_ch * 8)

        # up path
        self.up3 = UpBlock(base_ch * 8, base_ch * 4)
        self.up2 = UpBlock(base_ch * 4, base_ch * 2)
        self.up1 = UpBlock(base_ch * 2, base_ch)

        self.out_conv = nn.Sequential(nn.ConvTranspose2d(base_ch, base_ch // 2, kernel_size=2, stride=2),
                                      nn.SiLU(),
                                      nn.ConvTranspose2d(base_ch // 2, in_ch, kernel_size=2, stride=2),)

        self.init_weights()

    def init_weights(self):
        for block in [self.in_conv, self.out_conv, self.down1, self.down2, self.down3, self.mid, self.up3, self.up2, self.up1]:
            for sub in block.modules():
                default_init(sub)

    def forward(self, img):
        x0 = self.in_conv(img)

        x1, skip1 = self.down1(x0)
        x2, skip2 = self.down2(x1)
        x3, skip3 = self.down3(x2)

        m = self.mid(x3)

        u3 = self.up3(m, skip3)
        u2 = self.up2(u3, skip2)
        u1 = self.up1(u2, skip1)

        out = self.out_conv(u1)
        return out


