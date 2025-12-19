import torch
import torch.nn as nn
from VMUNet import VSSBlock
import random
from mambaSeg import default_init, ResBlockT, imgEncoder, combiner, OutputRefineBlock

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, depth=2):
        super().__init__()
        self.depth = depth
        self.in_res = ResBlockT(in_channels, out_channels)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleDict({
                "VSSBlock1": VSSBlock(hidden_dim=out_channels, mlp_drop_rate=0.15, ssm_drop_rate=0.15),
                "res": ResBlockT(out_channels, out_channels)
            }))

        default_init(self.in_res)
        for layer in self.layers:
            default_init(layer["res"])

    def forward(self, x):
        h = self.in_res(x)

        for layer in self.layers:
            h = h.permute(0, 2, 3, 1)
            h = layer["VSSBlock1"](h)
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
                "VSSBlock1": VSSBlock(hidden_dim=out_channels, mlp_drop_rate=0.15, ssm_drop_rate=0.15),
                "res": ResBlockT(out_channels, out_channels)
            }))
        self.pool = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        default_init(self.in_res)
        for layer in self.layers:
            default_init(layer["res"])

    def forward(self, x):
        h = self.in_res(x)

        for layer in self.layers:
            # h = layer["self_attn"](h)
            h = h.permute(0, 2, 3, 1)
            h = layer["VSSBlock1"](h)
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
                "VSSBlock1": VSSBlock(hidden_dim=out_channels, mlp_drop_rate=0.15, ssm_drop_rate=0.15),
                "res": ResBlockT(out_channels, out_channels)
            }))
        default_init(self.in_res)
        for layer in self.layers:
            default_init(layer["res"])

    def forward(self, x, skip):
        x = self.up(x)

        x = torch.cat([x, skip], dim=1)
        h = self.in_res(x)  # expand channels

        for layer in self.layers:
            # h = layer["self_attn"](h)
            h = h.permute(0, 2, 3, 1)
            h = layer["VSSBlock1"](h)
            h = h.permute(0, 3, 1, 2)
            h = layer["res"](h)
        return h

class student(nn.Module):
    def __init__(self, in_ch=1, base_ch=32, cond=True):  # cond_dim=512
        super().__init__()
        self.cond = cond
        self.encoderEnhanced = imgEncoder(in_ch=in_ch, out_ch=base_ch)
        self.encoderOriginal = imgEncoder(in_ch=in_ch, out_ch=base_ch)

        self.combiner = combiner(query_dim=base_ch, cond_dim=base_ch, num_head=4)

        # down path
        self.down1 = DownBlock(base_ch, base_ch)  # 64
        self.down2 = DownBlock(base_ch, base_ch * 2)  # 128
        self.down3 = DownBlock(base_ch * 2, base_ch * 4)  # 256

        # middle
        self.mid = MidBlock(base_ch * 4, base_ch * 8)  # 512

        # up path
        self.up3 = UpBlock(base_ch * 8, base_ch * 4) # 256
        self.up2 = UpBlock(base_ch * 4, base_ch * 2)  # 128
        self.up1 = UpBlock(base_ch * 2, base_ch)  # 64

        # output
        self.out_conv = OutputRefineBlock(base_ch, in_ch)

        self.init_weights()

    def init_weights(self):
        for block in [self.encoderOriginal, self.encoderEnhanced, self.combiner, self.down1, self.down2, self.down3, self.mid, self.up3, self.up2, self.up1, self.out_conv]: # self.encoderEnhanced, self.combiner *************
            for sub in block.modules():
                default_init(sub)

    def model_with_combiner(self, featEnhanced, featImg):
        x = self.combiner(featImg, featEnhanced)
        return x

    def forward(self, original, enhanced=None):
        x_original = self.encoderOriginal(original)
        x_enhanced = self.encoderEnhanced(enhanced)
        x_combined = self.model_with_combiner(x_enhanced, x_original)

        x1, skip1 = self.down1(x_combined)  # x1: base_ch/ pooled  ****************
        x2, skip2 = self.down2(x1)  # x2: base_ch*2

        x3, skip3 = self.down3(x2)  # x3: base_ch*4, h/8, w/8

        m = self.mid(x3)  # (B, base_ch*4, H/8, W/8)
        # up path
        u3 = self.up3(m, skip3)  # (B, base_ch*2, H/4, W/4)
        u2 = self.up2(u3, skip2)  # (B, base_ch, H/2, W/2)
        u1 = self.up1(u2, skip1)  # (B, base_ch, H, W)

        fused = torch.cat([u1, x_combined], dim=1)
        out = self.out_conv(fused)
        return out
