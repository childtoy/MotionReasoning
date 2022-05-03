import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, return_attention=False):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        if return_attention:
            return attn
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.depth = depth
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, return_attention=False):
        i =0 
        for attn, ff in self.layers:
            if i < len(self.layers) - 1:
                x = attn(x) + x
                x = ff(x) + x
                i+=1
            else : 
                if not return_attention :
                    x = attn(x) + x 
                    x = ff(x) + x
                    return x
                else : 
                    x = attn(x, return_attention=True)
                    print('attentionlayeer', x.shape)
                    return x
class PositionalEmbedding(nn.Module):
    def __init__(self, seq_len: int = 32, d_model: int = 96):
        super().__init__()
        self.pos_emb = nn.Embedding(seq_len + 1, d_model)

    def forward(self, inputs):
        positions = (
            torch.arange(inputs.size(0), device=inputs.device)
            .expand(inputs.size(1), inputs.size(0))
            .contiguous()
            + 1
        )
        outputs = inputs + self.pos_emb(positions).permute(1, 0, 2)
        return outputs
class Motion_Transformer(nn.Module):
    def __init__(self, *, seq_len=80, num_joints=35, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels =1, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        num_patches = num_joints
        path_height = channels
        patch_width = seq_len
        patch_dim =  seq_len 
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.to_joint_embedding= PositionalEmbedding(seq_len=seq_len, d_model=dim)

        # self.to_joint_embedding = nn.Sequential(
        #     Rearrange('b t j -> b j t'),
        #     nn.Linear(patch_dim, dim),
        # )

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, motion, return_attention=False):
        x = self.to_joint_embedding(motion)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        # x += self.pos_embedding[:, :(n + 1)]
        # x = self.dropout(x)
        print('x.shape', x.shape)
        x = self.transformer(x, return_attention)
        print('transformer', x.shape)
        if return_attention:
            return x
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

class MoT_pos(nn.Module):
    def __init__(self, *, seq_len=80, num_joints=35, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels =1, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        num_patches = num_joints
        path_height = channels
        patch_width = seq_len
        patch_dim =  seq_len 
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.to_joint_embedding= PositionalEmbedding(seq_len=seq_len, d_model=dim)

        # self.to_joint_embedding = nn.Sequential(
        #     Rearrange('b t j -> b j t'),
        #     nn.Linear(patch_dim, dim),
        # )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, motion, return_attention=False):
        x = self.to_joint_embedding(motion)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        # x += self.pos_embedding[:, :(n + 1)]
        # x = self.dropout(x)
        print('x.shape', x.shape)
        x = self.transformer(x, return_attention)
        if return_attention:
            return x
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
class MoT_pos(nn.Module):
    def __init__(self, *, seq_len=80, num_joints=35, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels =1, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        num_patches = num_joints
        path_height = channels
        patch_width = seq_len
        patch_dim =  seq_len 
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.rep_to_joint = nn.Linear(3, 1)

        self.to_joint_embedding = nn.Sequential(
            Rearrange('b t j 1-> b j t'),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, motion, return_attention=False):
        motion = self.rep_to_joint(motion)
        x = self.to_joint_embedding(motion)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x, return_attention)
        if return_attention:
            return x
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
class MoT_6d(nn.Module):
    def __init__(self, *, seq_len=80, num_joints=35, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels =1, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        num_patches = num_joints
        path_height = channels
        patch_width = seq_len
        patch_dim =  seq_len 
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.rep_to_joint = nn.Linear(6, 1)

        self.to_joint_embedding = nn.Sequential(
            Rearrange('b t j 1-> b j t'),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, motion, return_attention=False):
        motion = self.rep_to_joint(motion)
        x = self.to_joint_embedding(motion)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x, return_attention)
        if return_attention:
            return x
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
class MoT_6d2(nn.Module):
    def __init__(self, *, seq_len=80, num_joints=35, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels =1, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        num_patches = num_joints
        path_height = channels
        patch_width = seq_len
        patch_dim =  seq_len 
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_joint_embedding = nn.Sequential(
            Rearrange('b t j-> b j t'),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, motion, return_attention=False):
        # print('1',motion.shape)
        # motion = self.rep_to_joint(motion)
        # print('2', motion.shape)
        x = self.to_joint_embedding(motion)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x, return_attention)
        if return_attention:
            return x
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)    
class MoT_6d3(nn.Module):
    def __init__(self, *, seq_len=80, num_joints=35, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels =1, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        num_patches = num_joints
        path_height = channels
        patch_width = seq_len
        patch_dim =  seq_len 
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        dim = patch_dim
        self.to_joint_embedding = nn.Sequential(
            Rearrange('b t j-> b j t'),
            # nn.Linear(patch_dim, dim),
            nn.Identity(),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, motion, return_attention=False):
        # motion = self.rep_to_joint(motion)
        x = self.to_joint_embedding(motion)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x, return_attention)
        if return_attention:
            return x
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)                
class MoT_pos2(nn.Module):
    def __init__(self, *, seq_len=80, num_joints=35, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels =1, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        num_patches = num_joints
        path_height = channels
        patch_width = seq_len
        patch_dim =  seq_len 
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_joint_embedding = nn.Sequential(
            Rearrange('b t j-> b j t'),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, motion, return_attention=False):
        # print('1',motion.shape)
        # motion = self.rep_to_joint(motion)
        # print('2', motion.shape)
        x = self.to_joint_embedding(motion)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x, return_attention)
        if return_attention:
            return x
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)    
class MoT_pos3(nn.Module):
    def __init__(self, *, seq_len=80, num_joints=35, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels =1, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        num_patches = num_joints
        path_height = channels
        patch_width = seq_len
        patch_dim =  seq_len 
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        dim = patch_dim
        self.to_joint_embedding = nn.Sequential(
            Rearrange('b t j-> b j t'),
            # nn.Linear(patch_dim, dim),
            nn.Identity(),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, motion, return_attention=False):
        # motion = self.rep_to_joint(motion)
        x = self.to_joint_embedding(motion)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x, return_attention)
        if return_attention:
            return x
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)                        
class  MoT(nn.Module):
    def __init__(self, *, seq_len=80, num_joints=35, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels =1, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        num_patches = num_joints
        path_height = channels
        patch_width = seq_len
        patch_dim =  seq_len 
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.to_joint_embedding = nn.Sequential(
            Rearrange('b t j-> b j t'),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, motion, return_attention=False):
        x = self.to_joint_embedding(motion)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x, return_attention)
        if return_attention:
            return x
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)        

class  MoT_patch_6d(nn.Module):
    def __init__(self, *, seq_len=80, num_joints=35, sub_seq=20, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels =1, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height = num_joints 
        image_width = seq_len        
        patch_height = 1
        patch_width = sub_seq

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, motion, return_attention=False):
        x = self.to_joint_embedding(motion)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x, return_attention)
        if return_attention:
            return x
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)                


class  MoT_patch2(nn.Module):
    def __init__(self, *, seq_len=80, num_joints=35, sub_seq=20, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels =1, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height = seq_len
        image_width = num_joints
        patch_height = sub_seq
        patch_width = 1

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, motion, return_attention=False):
        x = self.to_patch_embedding(motion)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x, return_attention)
        if return_attention:
            return x
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)             

class  MoT_patch2_seg(nn.Module):
    def __init__(self, *, seq_len=80, num_joints=35, sub_seq=20, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels =1, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height = seq_len
        image_width = num_joints*6    
        patch_height = sub_seq
        patch_width = 6
        print(image_height)
        print(image_width)
        print(patch_height)
        print(patch_width)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.pos_emb = nn.Embedding(num_patches + 1, dim)
        self.val_emb = nn.Embedding(num_patches + 1, dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, motion, valid_length, return_attention=False):
        x = self.to_patch_embedding(motion)
        device = x.device

        b, n, _ = x.shape
        pos_x = torch.arange(n+1, device=x.device).repeat((b, 1))

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        valid_segment = []
        for i in valid_length:
            valid_segment.append([1] + [1] * torch.div(i, 5, rounding_mode='floor') + [0] * (n - torch.div(i, 5, rounding_mode='floor')))
        seg_tensor = torch.IntTensor(valid_segment).to(device)
        
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_emb(pos_x) + self.val_emb(seg_tensor) #
        # x += self.pos_embedding[:, :(n + 1)] + self.val_emb(seg_tensor)
        x = self.dropout(x)
        # print('transformer', x.shape)

        x = self.transformer(x, return_attention)
        # print(return_attention)
        # print('transformer', x.shape)

        if return_attention:
            return x
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)                     

class  MoT_patch2_seg_chn(nn.Module):
    def __init__(self, *, seq_len=80, num_joints=35, sub_seq=20, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels =6, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height = seq_len
        image_width = num_joints
        patch_height = sub_seq
        patch_width = 1
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.pos_emb = nn.Embedding(num_patches + 1, dim)
        self.val_emb = nn.Embedding(num_patches + 1, dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, motion, valid_length, return_attention=False):
        x = self.to_patch_embedding(motion)
        device = x.device

        b, n, _ = x.shape
        pos_x = torch.arange(n+1, device=x.device).repeat((b, 1))

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        valid_segment = []
        for i in valid_length:
            valid_segment.append([1] + [1] * torch.div(i, 5, rounding_mode='floor') + [0] * (n - torch.div(i, 5, rounding_mode='floor')))
        seg_tensor = torch.IntTensor(valid_segment).to(device)
        
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_emb(pos_x) + self.val_emb(seg_tensor) #
        # x += self.pos_embedding[:, :(n + 1)] + self.val_emb(seg_tensor)
        x = self.dropout(x)
        # print('transformer', x.shape)

        x = self.transformer(x, return_attention)
        # print(return_attention)
        # print('transformer', x.shape)

        if return_attention:
            return x
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)    

class  MoT_patch2_seg_chn_moe(nn.Module):
    def __init__(self, *, seq_len=80, num_joints=35, sub_seq=20, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels =6, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height = seq_len
        image_width = num_joints
        patch_height = sub_seq
        patch_width = 1
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.pos_emb = nn.Embedding(num_patches + 1, dim)
        self.val_emb = nn.Embedding(num_patches + 1, dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self.gait_weights = nn.Parameter(torch.randn(heads))

    def forward(self, motion, valid_length, return_attention=False):
        x = self.to_patch_embedding(motion)
        device = x.device

        b, n, _ = x.shape
        pos_x = torch.arange(n+1, device=x.device).repeat((b, 1))

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        valid_segment = []
        for i in valid_length:
            valid_segment.append([1] + [1] * torch.div(i, 5, rounding_mode='floor') + [0] * (n - torch.div(i, 5, rounding_mode='floor')))
        seg_tensor = torch.IntTensor(valid_segment).to(device)
        
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_emb(pos_x) + self.val_emb(seg_tensor) #
        # x += self.pos_embedding[:, :(n + 1)] + self.val_emb(seg_tensor)
        x = self.dropout(x)
        # print('transformer', x.shape)

        x = self.transformer(x, return_attention)
        # print(return_attention)
        # print('transformer', x.shape)
        res = 0 
        if return_attention:
            return x, self.gait_weights
        for i in range(8):
            res += x[:,i] * self.gait_weights[i]
        # output = torch.sum(x * torch.repeat(w_gating, axis=1), dim, axis=1), axis=2)

        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        out = self.mlp_head(res)
        return out

class  MoT_seg(nn.Module):
    def __init__(self, *, seq_len=80, num_joints=35, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels =1, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        num_patches = num_joints
        path_height = channels
        patch_width = seq_len
        patch_dim =  seq_len 
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.to_joint_embedding = nn.Sequential(
            Rearrange('b t j-> b j t'),
            nn.Linear(patch_dim, dim),
        )

        self.pos_emb = nn.Embedding(num_patches + 1, dim)
        self.val_emb = nn.Embedding(num_patches + 1, dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, motion, valid_length, return_attention=False):
        x = self.to_joint_embedding(motion)
        b, n, _ = x.shape
        device = x.device
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        valid_segment = []
        for i in valid_length:
            valid_segment.append([1] + [1] * torch.div(i, 5, rounding_mode='floor') + [0] * (n - torch.div(i, 5, rounding_mode='floor')))
        seg_tensor = torch.IntTensor(valid_segment).to(device)
                
        x = torch.cat((cls_tokens, x), dim=1)
        # x += self.pos_embedding[:, :(n + 1)]
        x = x + self.pos_emb(pos_x) + self.val_emb(seg_tensor) #
        x = self.dropout(x)
        x = self.transformer(x, return_attention)
        if return_attention:
            return x
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        
        x = self.to_latent(x)
        return self.mlp_head(x)               