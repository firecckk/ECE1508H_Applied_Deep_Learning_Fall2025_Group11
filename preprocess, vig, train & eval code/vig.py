'''
isotropic vig (https://arxiv.org/pdf/2206.00272)
- 3 variants: vig_isotropic_tiny/small/base
'''
import torch
import torch.nn as nn
from typing import Optional

# helper function
def conv_bn_act(in_ch, out_ch, kernel_size=3, stride=1, padding=1, act='gelu'):
    layers = [
        nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_ch)
    ]
    
    if act == 'relu':
        layers.append(nn.ReLU(inplace=True))
    elif act == 'gelu':
        layers.append(nn.GELU())
    else:
        layers.append(nn.ReLU(inplace=True))
        
    return nn.Sequential(*layers)

# stochastic path
class SimpleDropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.dim() - 1)
        random_tensor = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        
        return x.div(keep) * random_tensor

# overlap patch stem
# - downsamples the input (224 -> 14) using stride-2 convolution
# - each output spatial location will be a node for the graph
class OverlapPatchStem(nn.Module):
    def __init__(self, in_ch=3, embed_dim=192, act='gelu'):
        super().__init__()
        c1 = max(8, embed_dim // 8)
        c2 = max(16, embed_dim // 4)
        c3 = max(32, embed_dim // 2)
        self.stages = nn.Sequential(
            conv_bn_act(in_ch, c1, kernel_size=3, stride=2, padding=1, act=act), # 224->112
            conv_bn_act(c1, c2, kernel_size=3, stride=2, padding=1, act=act), # 112->56
            conv_bn_act(c2, c3, kernel_size=3, stride=2, padding=1, act=act), # 56->28
            conv_bn_act(c3, embed_dim, kernel_size=3, stride=2, padding=1, act=act), # 28->14
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim)
        )

    def forward(self, x):
        return self.stages(x)

# compute k nearest neighbor indices excluding self for each node
def knn_indices(x_flat: torch.Tensor, k: int) -> torch.Tensor:
    B, N, C = x_flat.shape
    dist = torch.cdist(x_flat, x_flat, p=2) # (B, N, N)
    device = dist.device

    # mask diagonal so self is not selected as nearest neighbor
    idx_diag = torch.arange(N, device=device)
    dist[:, idx_diag, idx_diag] = float('inf')
    _, idx = dist.topk(k=k, dim=-1, largest=False, sorted=False) # (B, N, k)

    return idx

# max-relative graph convolution
class GraphConvBlock(nn.Module):
    def __init__(self, dim: int, k: int = 9, heads: int = 1, act='gelu', drop_path=0.0):
        super().__init__()
        assert dim % heads == 0, 'dim must be divisible by heads'

        self.dim = dim
        self.k = k
        self.heads = heads
        self.updater = nn.Conv2d(2 * dim, dim, kernel_size=1, groups=heads, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.GELU() if act == 'gelu' else nn.ReLU(inplace=True)
        self.drop_path = SimpleDropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        N = H * W
        x_flat = x.view(B, C, N).transpose(1, 2).contiguous() # (B, N, C)

        # compute kNN
        idx = knn_indices(x_flat, self.k) # (B, N, k)

        # gather neighbors and compute max-relative aggregation agg_i = max_j (x_j - x_i) per batch
        aggregated_list = []

        for b in range(B):
            xb = x_flat[b] # (N, C)
            idx_b = idx[b] # (N, k)
            neighbors = xb[idx_b] # (N, k, C)
            rel = neighbors - xb.unsqueeze(1) # (N, k, C)
            agg, _ = rel.max(dim=1) # (N, C)
            aggregated_list.append(agg)
            
        aggregated = torch.stack(aggregated_list, dim=0) # (B, N, C)

        # concat xi and agg_i
        cat = torch.cat([x_flat, aggregated], dim=-1)  # (B, N, 2C)
        cat2d = cat.transpose(1, 2).view(B, 2 * C, H, W)  # (B, 2C, H, W)

        # update via grouped conv per-head
        out = self.updater(cat2d)
        out = self.bn(out)
        out = self.act(out)
        out = self.drop_path(out) + x

        return out

# convolutional feed-forward w/ residual
class ConvFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, act='gelu', drop_path=0.0):
        super().__init__()
        hidden_dim = hidden_dim or (dim * 4)
        self.fc1 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU() if act == 'gelu' else nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )
        self.drop_path = SimpleDropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.drop_path(out) + x

        return out

# isotropic vig: stem -> pos_embed -> repeated blocks (graphconv -> ffn) -> head
class IsotropicViG(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        in_ch: int = 3,
        embed_dim: int = 192,
        num_blocks: int = 12,
        k: int = 9,
        num_classes: int = 5,
        drop_path_rate: float = 0.0,
        act: str = 'gelu',
        heads: int = 1,
        head_dropout: float = 0.2
    ):
        super().__init__()
        assert image_size % (2 ** 4) == 0, 'image_size must be divisible by 16 for this stem'

        self.stem = OverlapPatchStem(in_ch=in_ch, embed_dim=embed_dim, act=act)

        down = 2 ** 4
        H = image_size // down
        W = H
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, H, W))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks)]
        blocks = []

        for i in range(num_blocks):
            blocks.append(nn.Sequential(
                GraphConvBlock(embed_dim, k=k, heads=heads, act=act, drop_path=dpr[i]),
                ConvFFN(embed_dim, hidden_dim=embed_dim * 4, act=act, drop_path=dpr[i]),
            ))

        self.blocks = nn.ModuleList(blocks)

        # classification head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(embed_dim, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.GELU() if act == 'gelu' else nn.ReLU(),
            nn.Dropout(p=head_dropout),
            nn.Linear(1024, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        # kaiming initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if getattr(m, 'weight', None) is not None:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                try:
                    nn.init.trunc_normal_(m.weight, std=0.02)
                except Exception:
                    nn.init.normal_(m.weight, std=0.02)

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        logits = self.head(x)

        return logits

# build wrappers
def create_vig_isotropic_tiny(num_classes=5, heads=1, **kwargs):
    return IsotropicViG(embed_dim=192, num_blocks=12, num_classes=num_classes, heads=heads, **kwargs)

def create_vig_isotropic_small(num_classes=5, heads=1, **kwargs):
    return IsotropicViG(embed_dim=320, num_blocks=16, num_classes=num_classes, heads=heads, **kwargs)

def create_vig_isotropic_base(num_classes=5, heads=1, **kwargs):
    return IsotropicViG(embed_dim=640, num_blocks=16, num_classes=num_classes, heads=heads, **kwargs)

# builder used by training script
def build_vig(
    variant: str = 'tiny',
    num_classes: int = 5,
    in_channels: int = 3,
    img_size: int = 224,
    k: int = 9,
    heads: int = 1,
    drop_path_rate: float = 0.0,
    head_dropout: float = 0.2,
    pretrained: bool = False
):
    v = variant.lower()

    if 'tiny' in v:
        model = create_vig_isotropic_tiny(
            num_classes=num_classes,
            heads=heads,
            image_size=img_size,
            in_ch=in_channels,
            k=k,
            drop_path_rate=drop_path_rate,
            head_dropout=head_dropout
        )
    elif 'small' in v:
        model = create_vig_isotropic_small(
            num_classes=num_classes,
            heads=heads,
            image_size=img_size,
            in_ch=in_channels,
            k=k,
            drop_path_rate=drop_path_rate,
            head_dropout=head_dropout
        )
    elif 'base' in v:
        model = create_vig_isotropic_base(
            num_classes=num_classes,
            heads=heads,
            image_size=img_size,
            in_ch=in_channels,
            k=k,
            drop_path_rate=drop_path_rate,
            head_dropout=head_dropout
        )
    else:
        # default to tiny
        model = create_vig_isotropic_tiny(
            num_classes=num_classes,
            heads=heads,
            image_size=img_size,
            in_ch=in_channels,
            k=k,
            drop_path_rate=drop_path_rate,
            head_dropout=head_dropout
        )

    # we train from scratch
    pretrained_name = None

    return model, pretrained_name

# kaiming initialization
def apply_kaiming_init(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if getattr(m, 'weight', None) is not None:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            if getattr(m, 'bias', None) is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.Linear):
            if getattr(m, 'weight', None) is not None:
                nn.init.kaiming_uniform_(m.weight, nonlinearity='linear')

            if getattr(m, 'bias', None) is not None:
                nn.init.zeros_(m.bias)

# test
if __name__ == "__main__":
    m, _ = build_vig(variant='tiny', num_classes=5, k=9, heads=1)
    x = torch.randn(2, 3, 224, 224)
    y = m(x)
    print("output shape:", y.shape)