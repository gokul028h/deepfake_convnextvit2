import torch
from torch import nn
import timm
from einops import rearrange


class ArtifactAttention(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels // 8, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        attn = self.conv1(x)
        attn = self.relu(attn)
        attn = self.conv2(attn)
        attn = self.sigmoid(attn)

        return x * attn


class CrossAttention(nn.Module):

    def __init__(self, dim=768, heads=8):
        super().__init__()

        self.heads = heads
        self.scale = dim ** -0.5

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.to_out = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(0.1)

    def forward(self, x, return_attention=False):

        B, N, C = x.shape

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = dots.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        if return_attention:
            return out, attn

        return out


class ConvNeXtCrossViT(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone = timm.create_model(
            "convnext_tiny",
            pretrained=True,
            num_classes=0,
            global_pool=""
        )

        # Replace stem for 6-channel input (3 RGB + 3 DIP: edge, FFT, ELA)
        old_conv = self.backbone.stem[0]

        new_conv = nn.Conv2d(
            6,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )

        # Copy pretrained RGB weights to first 3 channels
        new_conv.weight.data[:, :3] = old_conv.weight.data
        # Copy pretrained weights to DIP channels too (better than random init
        # since DIP maps are structured image-like data)
        new_conv.weight.data[:, 3:] = old_conv.weight.data

        if old_conv.bias is not None:
            new_conv.bias.data = old_conv.bias.data

        self.backbone.stem[0] = new_conv

        channels = self.backbone.num_features

        self.artifact_attention = ArtifactAttention(channels)

        self.cross_attn = CrossAttention(dim=channels)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, 1)
        )

    def forward(self, x):

        x = self.backbone.forward_features(x)

        x = self.artifact_attention(x)

        B, C, H, W = x.shape

        x = rearrange(x, "b c h w -> b (h w) c")

        x = self.cross_attn(x)

        x = x.transpose(1, 2)

        x = self.pool(x).squeeze(-1)

        return self.fc(x)


    def forward_attention(self, x):

        x = self.backbone.forward_features(x)

        x = self.artifact_attention(x)

        B, C, H, W = x.shape

        x = rearrange(x, "b c h w -> b (h w) c")

        x, attn = self.cross_attn(x, return_attention=True)

        return x, attn