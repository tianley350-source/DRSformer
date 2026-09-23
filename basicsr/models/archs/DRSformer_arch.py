import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
import random

from einops import rearrange

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##  Mixed-Scale Feed-forward Network (MSFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv3x3 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.dwconv5x5 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=5, stride=1, padding=2, groups=hidden_features * 2, bias=bias)
        self.relu3 = nn.ReLU()
        self.relu5 = nn.ReLU()

        self.dwconv3x3_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features , bias=bias)
        self.dwconv5x5_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features , bias=bias)

        self.relu3_1 = nn.ReLU()
        self.relu5_1 = nn.ReLU()

        self.project_out = nn.Conv2d(hidden_features * 2, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1_3, x2_3 = self.relu3(self.dwconv3x3(x)).chunk(2, dim=1)
        x1_5, x2_5 = self.relu5(self.dwconv5x5(x)).chunk(2, dim=1)

        x1 = torch.cat([x1_3, x1_5], dim=1)
        x2 = torch.cat([x2_3, x2_5], dim=1)

        x1 = self.relu3_1(self.dwconv3x3_1(x1))
        x2 = self.relu5_1(self.dwconv5x5_1(x2))

        x = torch.cat([x1, x2], dim=1)

        x = self.project_out(x)

        return x

##  Top-K Sparse Attention (TKSA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.)

        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, x, mix_weights=None):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, C, _ = q.shape

        mask1 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask2 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask3 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask4 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        index = torch.topk(attn, k=int(C/2), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*2/3), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*3/4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*4/5), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))

        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)

        out1 = (attn1 @ v)
        out2 = (attn2 @ v)
        out3 = (attn3 @ v)
        out4 = (attn4 @ v)

        if mix_weights is None:
            out = (out1 * self.attn1 + out2 * self.attn2 +
                   out3 * self.attn3 + out4 * self.attn4)
        else:
            if mix_weights.shape != (b, self.num_heads, 4):
                raise ValueError(
                    'mix_weights must have shape '
                    f'[{b}, {self.num_heads}, 4], got {list(mix_weights.shape)}.')
            branches = torch.stack((out1, out2, out3, out4), dim=-1)
            out = (branches * mix_weights[:, :, None, None, :]).sum(dim=-1)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

##  Sparse Transformer Block (STB) 
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, attention_weights=None):
        x = x + self.attn(self.norm1(x), mix_weights=attention_weights)
        x = x + self.ffn(self.norm2(x))

        return x

class OperationLayer(nn.Module):
    def __init__(self, C, stride):
        super(OperationLayer, self).__init__()
        self._ops = nn.ModuleList()
        for o in Operations:
            op = OPS[o](C, stride, False)
            self._ops.append(op)

        self._out = nn.Sequential(nn.Conv2d(C * len(Operations), C, 1, padding=0, bias=False), nn.ReLU())

    def forward(self, x, weights):
        weights = weights.transpose(1, 0)
        states = []
        for w, op in zip(weights, self._ops):
            states.append(op(x) * w.view([-1, 1, 1, 1]))
        return self._out(torch.cat(states[:], dim=1))

class GroupOLs(nn.Module):
    def __init__(self, steps, C):
        super(GroupOLs, self).__init__()
        self.preprocess = ReLUConv(C, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._ops = nn.ModuleList()
        self.relu = nn.ReLU()
        stride = 1

        for _ in range(self._steps):
            op = OperationLayer(C, stride)
            self._ops.append(op)

    def forward(self, s0, weights):
        s0 = self.preprocess(s0)
        for i in range(self._steps):
            res = s0
            s0 = self._ops[i](s0, weights[:, i, :])
            s0 = self.relu(s0 + res)
        return s0

class OALayer(nn.Module):
    def __init__(self, channel, k, num_ops):
        super(OALayer, self).__init__()
        self.k = k
        self.num_ops = num_ops
        self.output = k * num_ops
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca_fc = nn.Sequential(
            nn.Linear(channel, self.output * 2),
            nn.ReLU(),
            nn.Linear(self.output * 2, self.k * self.num_ops))

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.view(x.size(0), -1)
        y = self.ca_fc(y)
        y = y.view(-1, self.k, self.num_ops)
        return y

Operations = [
    'sep_conv_1x1',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'sep_conv_7x7',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'dil_conv_7x7',
    'avg_pool_3x3'
]

OPS = {
    'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'sep_conv_1x1' : lambda C, stride, affine: SepConv(C, C, 1, stride, 0, affine=affine),
    'sep_conv_3x3' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7' : lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'dil_conv_7x7' : lambda C, stride, affine: DilConv(C, C, 7, stride, 6, 2, affine=affine),
}

class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        return self.op(x)

class ReLUConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.ReLU(inplace=False))

    def forward(self, x):
        return self.op(x)

class DilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),)

    def forward(self, x):
        return self.op(x)

class ResBlock(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False)
        self.conv2 = nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False)
        self.relu  = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + residual
        out = self.relu(out)
        return out

class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),)

    def forward(self, x):
        return self.op(x)

## Mixture of Experts Feature Compensator (MEFC)
class subnet(nn.Module):
    def __init__(self, dim, layer_num=1, steps=4):
        super(subnet,self).__init__()

        self._C = dim
        self.num_ops = len(Operations)
        self._layer_num = layer_num
        self._steps = steps

        self.layers = nn.ModuleList()
        for _ in range(self._layer_num):
            attention = OALayer(self._C, self._steps, self.num_ops)
            self.layers += [attention]
            layer = GroupOLs(steps, self._C)
            self.layers += [layer]

    def forward(self, x):
    
        for _, layer in enumerate(self.layers):
            if isinstance(layer, OALayer):
                weights = layer(x)
                weights = F.softmax(weights, dim=-1)
            else:
                x = layer(x, weights)

        return x

## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class RecurrentLatentCore(nn.Module):
    """Shared recurrent latent blocks used by LoopDRSformer.

    The pre/post blocks remain unshared. Only ``shared_core`` is executed more
    than once, so parameter count and effective depth can be controlled
    independently.
    """

    def __init__(self,
                 dim,
                 num_heads,
                 ffn_expansion_factor,
                 bias,
                 LayerNorm_type,
                 latent_pre_blocks=1,
                 latent_shared_blocks=2,
                 latent_post_blocks=1,
                 max_loops=4,
                 train_loop_range=None,
                 loop_embedding=True,
                 input_injection=True,
                 layer_scale_init=0.1):
        super(RecurrentLatentCore, self).__init__()

        if max_loops < 1:
            raise ValueError('max_loops must be at least 1.')
        if latent_shared_blocks < 1:
            raise ValueError('latent_shared_blocks must be at least 1.')

        if train_loop_range is None:
            train_loop_range = (max_loops, max_loops)
        if len(train_loop_range) != 2:
            raise ValueError('train_loop_range must contain [min_loops, max_loops].')
        train_min, train_max = (int(train_loop_range[0]), int(train_loop_range[1]))
        if not 1 <= train_min <= train_max <= max_loops:
            raise ValueError(
                'train_loop_range must satisfy 1 <= min <= max <= max_loops.')

        def make_block():
            return TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type)

        self.latent_pre = nn.Sequential(
            *[make_block() for _ in range(latent_pre_blocks)])
        self.shared_core = nn.Sequential(
            *[make_block() for _ in range(latent_shared_blocks)])
        self.latent_post = nn.Sequential(
            *[make_block() for _ in range(latent_post_blocks)])

        self.max_loops = int(max_loops)
        self.train_loop_range = (train_min, train_max)
        self.input_inject = (nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
                             if input_injection else None)
        if input_injection:
            self.input_scale = nn.Parameter(
                torch.full((self.max_loops, 1, 1, 1), float(layer_scale_init)))
        else:
            self.register_parameter('input_scale', None)

        if loop_embedding:
            self.loop_embedding = nn.Parameter(
                torch.zeros(self.max_loops, dim, 1, 1))
        else:
            self.register_parameter('loop_embedding', None)

        self.layer_scale = nn.Parameter(
            torch.full((self.max_loops, 1, 1, 1), float(layer_scale_init)))

    def _resolve_num_loops(self, max_loops):
        if max_loops is not None:
            num_loops = int(max_loops)
        elif self.training:
            num_loops = random.randint(*self.train_loop_range)
        else:
            num_loops = self.max_loops

        if not 1 <= num_loops <= self.max_loops:
            raise ValueError(
                f'num_loops must be in [1, {self.max_loops}], got {num_loops}.')
        return num_loops

    def forward(self, x, max_loops=None, return_states=False):
        num_loops = self._resolve_num_loops(max_loops)
        h0 = self.latent_pre(x)
        h = h0
        states = []
        injected = self.input_inject(h0) if self.input_inject is not None else None

        for loop_id in range(num_loops):
            u = h
            if injected is not None:
                u = u + self.input_scale[loop_id] * injected
            if self.loop_embedding is not None:
                u = u + self.loop_embedding[loop_id]

            z = self.shared_core(u)
            delta = z - u
            h = h + self.layer_scale[loop_id] * delta
            states.append(h)

        final_state = self.latent_post(h)
        exit_stats = {'num_loops': num_loops}
        if return_states:
            return final_state, states, exit_stats
        return final_state, [], exit_stats


class DynamicTKSARouter(nn.Module):
    """Predict per-sample, per-head mixtures over the four TKSA branches."""

    def __init__(self, dim, num_heads, max_loops, hidden_ratio=0.25):
        super(DynamicTKSARouter, self).__init__()
        hidden_dim = max(16, int(dim * hidden_ratio))
        self.num_heads = num_heads
        self.loop_embedding = nn.Embedding(max_loops, dim)
        self.proj = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads * 4))
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x, loop_id):
        pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)
        loop_ids = torch.full(
            (x.shape[0],), int(loop_id), dtype=torch.long, device=x.device)
        context = torch.cat((pooled, self.loop_embedding(loop_ids)), dim=1)
        logits = self.proj(context).view(x.shape[0], self.num_heads, 4)
        return F.softmax(logits, dim=-1)


class RainGate(nn.Module):
    """Predict a loop-conditioned, single-channel soft update mask."""

    def __init__(self, dim, max_loops):
        super(RainGate, self).__init__()
        hidden_dim = max(16, dim // 8)
        self.reduce = nn.Conv2d(dim * 2, hidden_dim, kernel_size=1)
        self.loop_embedding = nn.Embedding(max_loops, hidden_dim)
        self.spatial = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=3, padding=1,
            groups=hidden_dim)
        self.output = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        nn.init.constant_(self.output.bias, 2.0)

    def forward(self, h, h0, loop_id):
        features = F.gelu(self.reduce(torch.cat((h, h0), dim=1)))
        loop_ids = torch.full(
            (h.shape[0],), int(loop_id), dtype=torch.long, device=h.device)
        condition = self.loop_embedding(loop_ids)[:, :, None, None]
        features = F.gelu(self.spatial(features + condition))
        return torch.sigmoid(self.output(features))


class LoopAdapter(nn.Module):
    """Low-rank residual adapter specialized for one recurrent round."""

    def __init__(self, dim, reduction=16):
        super(LoopAdapter, self).__init__()
        hidden_dim = max(8, dim // reduction)
        self.body = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1))
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, x):
        return self.body(x)


class LoopConditionedTransformerBlock(nn.Module):
    """Transformer block whose TKSA mixture depends on input and loop id."""

    def __init__(self,
                 dim,
                 num_heads,
                 ffn_expansion_factor,
                 bias,
                 LayerNorm_type,
                 max_loops,
                 dynamic_tksa=True,
                 loop_adapters=True):
        super(LoopConditionedTransformerBlock, self).__init__()
        self.block = TransformerBlock(
            dim=dim,
            num_heads=num_heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type)
        self.router = (DynamicTKSARouter(dim, num_heads, max_loops)
                       if dynamic_tksa else None)
        self.adapters = (nn.ModuleList([
            LoopAdapter(dim) for _ in range(max_loops)
        ]) if loop_adapters else None)

    def forward(self, x, loop_id):
        router_weights = (self.router(x, loop_id)
                          if self.router is not None else None)
        x = self.block(x, attention_weights=router_weights)
        if self.adapters is not None:
            adapter_update = self.adapters[loop_id](x)
            x = x + adapter_update
            adapter_norm = adapter_update.square().mean().sqrt()
        else:
            adapter_norm = x.new_zeros(())
        return x, router_weights, adapter_norm


class RecurrentLatentCoreV2(RecurrentLatentCore):
    """Recurrent latent core with loop-conditioned dynamic sparse attention."""

    def __init__(self,
                 dim,
                 num_heads,
                 ffn_expansion_factor,
                 bias,
                 LayerNorm_type,
                 latent_pre_blocks=1,
                 latent_shared_blocks=4,
                 latent_post_blocks=1,
                 max_loops=3,
                 train_loop_range=None,
                 loop_embedding=True,
                 input_injection=True,
                 layer_scale_init=0.1,
                 dynamic_tksa=True,
                 rain_gate=True,
                 loop_adapters=True):
        super(RecurrentLatentCoreV2, self).__init__(
            dim=dim,
            num_heads=num_heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
            latent_pre_blocks=latent_pre_blocks,
            latent_shared_blocks=latent_shared_blocks,
            latent_post_blocks=latent_post_blocks,
            max_loops=max_loops,
            train_loop_range=train_loop_range,
            loop_embedding=loop_embedding,
            input_injection=input_injection,
            layer_scale_init=layer_scale_init)
        self.num_heads = num_heads
        self.shared_core = nn.ModuleList([
            LoopConditionedTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                max_loops=max_loops,
                dynamic_tksa=dynamic_tksa,
                loop_adapters=loop_adapters)
            for _ in range(latent_shared_blocks)
        ])
        self.rain_gate = RainGate(dim, max_loops) if rain_gate else None

    def forward(self, x, max_loops=None, return_states=False):
        num_loops = self._resolve_num_loops(max_loops)
        h0 = self.latent_pre(x)
        h = h0
        states = []
        all_router_weights = []
        rain_gate_means = []
        all_adapter_norms = []
        injected = self.input_inject(h0) if self.input_inject is not None else None

        for loop_id in range(num_loops):
            u = h
            if injected is not None:
                u = u + self.input_scale[loop_id] * injected
            if self.loop_embedding is not None:
                u = u + self.loop_embedding[loop_id]

            z = u
            loop_router_weights = []
            loop_adapter_norms = []
            for block in self.shared_core:
                z, router_weights, adapter_norm = block(z, loop_id)
                if router_weights is not None:
                    loop_router_weights.append(router_weights.mean(dim=0))
                loop_adapter_norms.append(adapter_norm)
            if loop_router_weights:
                all_router_weights.append(torch.stack(loop_router_weights))
            all_adapter_norms.append(torch.stack(loop_adapter_norms))

            delta = z - u
            gate = (self.rain_gate(h, h0, loop_id)
                    if self.rain_gate is not None else 1.0)
            h = h + self.layer_scale[loop_id] * gate * delta
            if self.rain_gate is not None:
                rain_gate_means.append(gate.mean())
            states.append(h)

        final_state = self.latent_post(h)
        if all_router_weights:
            router_stats = torch.stack(all_router_weights).detach()
        else:
            router_stats = torch.empty(
                num_loops, len(self.shared_core), self.num_heads, 0,
                device=x.device)
        exit_stats = {
            'num_loops': num_loops,
            'router_weights': router_stats,
            'rain_gate_means': (torch.stack(rain_gate_means).detach()
                                if rain_gate_means else
                                torch.empty(0, device=x.device)),
            'adapter_update_norms': torch.stack(all_adapter_norms).detach(),
        }
        if return_states:
            return final_state, states, exit_stats
        return final_state, [], exit_stats

class DRSformer(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias'  ## Other option 'BiasFree'
                 ):

        super(DRSformer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        
        self.encoder_level0 = subnet(dim)  ## We do not use MEFC for training Rain200L and SPA-Data

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = subnet(dim=int(dim*2**1)) ## We do not use MEFC for training Rain200L and SPA-Data

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def _encode(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        inp_enc_level0 = self.encoder_level0(inp_enc_level1) ## We do not use MEFC for training Rain200L and SPA-Data
        out_enc_level1 = self.encoder_level1(inp_enc_level0)  

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        return inp_enc_level4, (out_enc_level1, out_enc_level2, out_enc_level3)

    def _decode(self, latent, encoder_skips, inp_img):
        out_enc_level1, out_enc_level2, out_enc_level3 = encoder_skips
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1) ## We do not use MEFC for training Rain200L and SPA-Data

        out_dec_level1 = self.output(out_dec_level1) + inp_img
        return out_dec_level1

    def forward(self, inp_img):
        inp_enc_level4, encoder_skips = self._encode(inp_img)
        latent = self.latent(inp_enc_level4)
        return self._decode(latent, encoder_skips, inp_img)


class LoopDRSformer(DRSformer):
    """DRSformer with a recurrent, parameter-shared latent stage."""

    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 latent_pre_blocks=1,
                 latent_shared_blocks=2,
                 latent_post_blocks=1,
                 max_loops=4,
                 train_loop_range=None,
                 loop_embedding=True,
                 input_injection=True,
                 layer_scale_init=0.1,
                 intermediate_supervision=True,
                 dynamic_tksa=False,
                 rain_gate=False,
                 adaptive_exit=False):
        if dynamic_tksa or rain_gate or adaptive_exit:
            raise NotImplementedError(
                'dynamic_tksa, rain_gate, and adaptive_exit belong to later '
                'stages and must remain disabled for the fixed-loop model.')

        super(LoopDRSformer, self).__init__(
            inp_channels=inp_channels,
            out_channels=out_channels,
            dim=dim,
            num_blocks=num_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type)

        latent_dim = int(dim * 2 ** 3)
        self.latent = RecurrentLatentCore(
            dim=latent_dim,
            num_heads=heads[3],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
            latent_pre_blocks=latent_pre_blocks,
            latent_shared_blocks=latent_shared_blocks,
            latent_post_blocks=latent_post_blocks,
            max_loops=max_loops,
            train_loop_range=train_loop_range,
            loop_embedding=loop_embedding,
            input_injection=input_injection,
            layer_scale_init=layer_scale_init)
        self.intermediate_supervision = bool(intermediate_supervision)
        self.aux_output = nn.Conv2d(
            latent_dim, out_channels, kernel_size=3, stride=1, padding=1,
            bias=bias)
        self.last_loop_stats = None

    def _aux_prediction(self, latent, inp_img):
        residual = self.aux_output(latent)
        residual = F.interpolate(
            residual, size=inp_img.shape[-2:], mode='bilinear',
            align_corners=False)
        return residual + inp_img

    def forward(self, inp_img, force_loops=None, return_aux=None):
        if return_aux is None:
            return_aux = self.training and self.intermediate_supervision

        inp_enc_level4, encoder_skips = self._encode(inp_img)
        latent, loop_states, exit_stats = self.latent(
            inp_enc_level4,
            max_loops=force_loops,
            return_states=return_aux)
        self.last_loop_stats = exit_stats
        output = self._decode(latent, encoder_skips, inp_img)

        if not return_aux:
            return output

        # The last loop state is supervised by the full decoder output. Earlier
        # states use one shared lightweight head, as proposed in the design.
        predictions = [
            self._aux_prediction(state, inp_img) for state in loop_states[:-1]
        ]
        predictions.append(output)
        return predictions


class LoopDRSformerV2(LoopDRSformer):
    """Capacity-balanced recurrent model with quality-oriented conditioning."""

    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 latent_pre_blocks=1,
                 latent_shared_blocks=4,
                 latent_post_blocks=1,
                 max_loops=3,
                 train_loop_range=None,
                 loop_embedding=True,
                 input_injection=True,
                 layer_scale_init=0.1,
                 intermediate_supervision=True,
                 loop_adapters=True,
                 dynamic_tksa=True,
                 rain_gate=True,
                 adaptive_exit=False):
        if adaptive_exit:
            raise NotImplementedError(
                'adaptive_exit is not part of the quality-first V2 model.')
        super(LoopDRSformerV2, self).__init__(
            inp_channels=inp_channels,
            out_channels=out_channels,
            dim=dim,
            num_blocks=num_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
            latent_pre_blocks=latent_pre_blocks,
            latent_shared_blocks=latent_shared_blocks,
            latent_post_blocks=latent_post_blocks,
            max_loops=max_loops,
            train_loop_range=train_loop_range,
            loop_embedding=loop_embedding,
            input_injection=input_injection,
            layer_scale_init=layer_scale_init,
            intermediate_supervision=intermediate_supervision,
            dynamic_tksa=False,
            rain_gate=False,
            adaptive_exit=False)
        latent_dim = int(dim * 2 ** 3)
        self.latent = RecurrentLatentCoreV2(
            dim=latent_dim,
            num_heads=heads[3],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
            latent_pre_blocks=latent_pre_blocks,
            latent_shared_blocks=latent_shared_blocks,
            latent_post_blocks=latent_post_blocks,
            max_loops=max_loops,
            train_loop_range=train_loop_range,
            loop_embedding=loop_embedding,
            input_injection=input_injection,
            layer_scale_init=layer_scale_init,
            dynamic_tksa=dynamic_tksa,
            rain_gate=rain_gate,
            loop_adapters=loop_adapters)
        self.use_loop_adapters = bool(loop_adapters)
        self.use_dynamic_tksa = bool(dynamic_tksa)
        self.use_rain_gate = bool(rain_gate)

    def forward(self,
                inp_img,
                force_loops=None,
                return_aux=None,
                return_details=False):
        if return_aux is None:
            return_aux = self.training and self.intermediate_supervision

        inp_enc_level4, encoder_skips = self._encode(inp_img)
        latent, loop_states, loop_stats = self.latent(
            inp_enc_level4,
            max_loops=force_loops,
            return_states=return_aux)
        self.last_loop_stats = loop_stats
        output = self._decode(latent, encoder_skips, inp_img)

        predictions = [output]
        if return_aux:
            predictions = [
                self._aux_prediction(state, inp_img)
                for state in loop_states[:-1]
            ]
            predictions.append(output)

        if return_details:
            return {
                'output': output,
                'predictions': predictions,
                'loop_stats': loop_stats,
            }
        if return_aux:
            return predictions
        return output

if __name__ == '__main__':
    input = torch.rand(1, 3, 256, 256)
    model = DRSformer()
   # output = model(input)

    from fvcore.nn import FlopCountAnalysis, parameter_count_table

    flops = FlopCountAnalysis(model, input)
    print("FLOPs: ", flops.total())
