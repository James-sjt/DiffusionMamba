import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from VMUNet import VSSBlock
import random

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


class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads=4, ffn_dim=2048):
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, ffn_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ffn_dim, channels, kernel_size=1)
        )

        # Layer Normalization
        self.ln1 = nn.LayerNorm(channels)  # For pre-attention normalization
        self.ln2 = nn.LayerNorm(channels)  # For post-attention normalization

        default_init(self)

    def forward(self, x):
        B, C, H, W = x.shape

        # Apply LayerNorm before attention (reshape to 2D for LN)
        x_ln = self.ln1(x.view(B, C, -1).permute(0, 2, 1)).permute(0, 2, 1).view(B, C, H, W)

        qkv = self.qkv(x_ln)  # (B, 3C, H, W)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape for multi-head attention
        def reshape_heads(t):
            return t.reshape(B, self.num_heads, C // self.num_heads, H * W)

        qh = reshape_heads(q)
        kh = reshape_heads(k)
        vh = reshape_heads(v)

        # Scaled dot-product attention
        qh = qh / math.sqrt(C // self.num_heads)
        attn = torch.einsum("bhnc,bhmc->bhnm", qh, kh)  # (B, heads, N, N)
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum("bhnm,bhmc->bhnc", attn, vh)  # (B, heads, C//heads, N)
        out = out.reshape(B, C, H, W)
        out = self.proj(out)

        # Apply LayerNorm after attention (reshape to 2D for LN)
        out = self.ln2(out.view(B, C, -1).permute(0, 2, 1)).permute(0, 2, 1).view(B, C, H, W)

        # Apply Feed-Forward Network (FFN)
        out = self.ffn(out)

        return out + x  # Residual connection

class CrossAttention(nn.Module):
    def __init__(self, query_dim, cond_dim, num_heads=4, ffn_dim=None):
        super().__init__()
        assert query_dim % num_heads == 0
        self.num_heads = num_heads
        self.dim_head = query_dim // num_heads
        self.scale = math.sqrt(self.dim_head)

        self.to_q = nn.Linear(query_dim, query_dim)
        self.to_k = nn.Linear(cond_dim, query_dim)
        self.to_v = nn.Linear(cond_dim, query_dim)
        self.to_out = nn.Linear(query_dim, query_dim)

        if ffn_dim is None:
            ffn_dim = query_dim * 4
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, query_dim)
        )

        self.norm_feat = nn.LayerNorm(query_dim)
        self.norm_cond = nn.LayerNorm(query_dim)
        self.norm_ffn = nn.LayerNorm(query_dim)

        default_init(self)

    def forward(self, feat, cond_tokens):
        B, C, H, W = feat.shape
        N = H * W

        feat = self.norm_feat(feat.view(B, C, N).permute(0, 2, 1))  # Normalize feat (query)
        cond_tokens = self.norm_cond(cond_tokens)  # Normalize cond_tokens (key/values)

        q = feat.view(B, C, N).permute(0, 2, 1)  # (B, N, C)

        q = self.to_q(q)  # (B, N, C)
        k = self.to_k(cond_tokens)  # (B, N_cond, C)
        v = self.to_v(cond_tokens)  # (B, N_cond, C)

        q = q.view(B, N, self.num_heads, self.dim_head).permute(0, 2, 1, 3)  # (B, heads, N, dh)
        k = k.view(B, -1, self.num_heads, self.dim_head).permute(0, 2, 3, 1)  # (B, heads, dh, N_cond)
        v = v.view(B, -1, self.num_heads, self.dim_head).permute(0, 2, 1, 3)  # (B, heads, N_cond, dh)

        attn = torch.matmul(q, k) / self.scale  # (B, heads, N, N_cond)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # (B, heads, N, dh)
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)  # (B, N, C)

        out = self.to_out(out)
        out = out + feat

        out = self.norm_ffn(out)

        out = self.ffn(out)

        out = out + feat

        # Reshape back to (B, C, H, W)
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return out

class MidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, depth=2):
        super().__init__()
        self.depth = depth
        self.in_res = ResBlockT(in_channels, out_channels)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleDict({
                "VSSBlock1": VSSBlock(hidden_dim=out_channels, mlp_drop_rate=0.2, ssm_drop_rate=0.2),
                "VSSBlock2": VSSBlock(hidden_dim=out_channels, mlp_drop_rate=0.2, ssm_drop_rate=0.2),
                "res": ResBlockT(out_channels, out_channels)
            }))

        default_init(self.in_res)
        for layer in self.layers:
            default_init(layer)

    def forward(self, x):
        h = self.in_res(x)

        for layer in self.layers:
            # h = layer["self_attn"](h)
            h = h.permute(0, 2, 3, 1)
            h = layer["VSSBlock1"](h)
            # h = layer["VSSBlock2"](h)
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
                #"self_attn": SelfAttention(out_channels, num_heads=num_heads),
                "VSSBlock1": VSSBlock(hidden_dim=out_channels, mlp_drop_rate=0.2, ssm_drop_rate=0.2),
                "VSSBlock2": VSSBlock(hidden_dim=out_channels, mlp_drop_rate=0.2, ssm_drop_rate=0.2),
                "res": ResBlockT(out_channels, out_channels)
            }))
        self.pool = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        default_init(self.in_res)
        for layer in self.layers:
            default_init(layer)

    def forward(self, x):
        h = self.in_res(x)

        for layer in self.layers:
            #h = layer["self_attn"](h)
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
                #"self_attn": SelfAttention(out_channels, num_heads=num_heads),
                "VSSBlock1": VSSBlock(hidden_dim=out_channels, mlp_drop_rate=0.2, ssm_drop_rate=0.2),
                "VSSBlock2": VSSBlock(hidden_dim=out_channels, mlp_drop_rate=0.2, ssm_drop_rate=0.2),
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
            #h = layer["self_attn"](h)
            h = h.permute(0, 2, 3, 1)
            h = layer["VSSBlock1"](h)
            h = layer["VSSBlock2"](h)
            h = h.permute(0, 3, 1, 2)
            h = layer["res"](h)
        return h


# --- CBAM modules ---
class ChannelAttention(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(ch, ch // r, 1, bias=False)
        self.fc2 = nn.Conv2d(ch // r, ch, 1, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        avg_out = self.fc2(self.act(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.act(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return torch.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return torch.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, ch, r=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(ch, r)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class imgEncoder(nn.Module):
    def __init__(self, in_ch=1, out_ch=64):  # ************************************
        super().__init__()

        # Stage 1
        mid_ch1 = out_ch // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch1, 3, 1, 1),
            nn.GroupNorm(4, mid_ch1),
            nn.SiLU(),
            nn.Conv2d(mid_ch1, mid_ch1, 3, 1, 1),
            nn.GroupNorm(4, mid_ch1),
        )
        self.cbam1 = CBAM(mid_ch1)
        self.res1 = nn.Conv2d(in_ch, mid_ch1, 1)
        self.down1 = nn.Conv2d(mid_ch1, mid_ch1, 3, 2, 1)

        # Stage 2
        mid_ch2 = out_ch
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_ch1, mid_ch2, 3, 1, 1),
            nn.GroupNorm(8, mid_ch2),
            nn.SiLU(),
            nn.Conv2d(mid_ch2, mid_ch2, 3, 1, 1),
            nn.GroupNorm(8, mid_ch2),
        )
        self.cbam2 = CBAM(mid_ch2)
        self.res2 = nn.Conv2d(mid_ch1, mid_ch2, 1)
        self.down2 = nn.Conv2d(mid_ch2, mid_ch2, 3, 2, 1)

    def forward(self, x):
        # Stage 1
        res1 = self.res1(x)
        x = self.conv1(x)
        x = self.cbam1(x)
        x = F.silu(x + res1)
        x = self.down1(x)

        # Stage 2
        res2 = self.res2(x)
        x = self.conv2(x)
        x = self.cbam2(x)
        x = F.silu(x + res2)
        x = self.down2(x)

        return x

class combiner(nn.Module):
    def __init__(self, query_dim, cond_dim, num_head):
        super().__init__()
        self.crossAtt = CrossAttention(query_dim, cond_dim, num_head)

    def to_tokens(self, feat):
        B, C, H, W = feat.shape
        return feat.view(B, C, H * W).permute(0, 2, 1)  # (B, N, C)

    def forward(self, feat, cond):
        fused_feat = self.crossAtt(feat, self.to_tokens(cond))
        return fused_feat

class OutputRefineBlock(nn.Module):
    def __init__(self, base_ch, in_ch):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(2*base_ch, base_ch, kernel_size=2, stride=2),
            nn.GroupNorm(8, base_ch),
            nn.SiLU()
        )
        self.refine1 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_ch)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_ch, base_ch, kernel_size=2, stride=2),
            nn.GroupNorm(8, base_ch),
            nn.SiLU()
        )
        self.refine2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_ch)
        )
        self.out_conv = nn.Conv2d(base_ch, in_ch, kernel_size=1)

    def forward(self, x):
        x = self.up1(x)
        res1 = self.refine1(x)
        x = x + res1                     # residual refinement

        x = self.up2(x)
        res2 = self.refine2(x)
        x = x + res2                     # residual refinement

        out = self.out_conv(x)
        return out

class mambaSeg(nn.Module):
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


