import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat
import sys
class SimMIM(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        masking_ratio = 0.5
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_joint, encoder_dim = encoder.pos_embedding.shape[-2:]
        # self.rep_to_patch = encoder.rep_to_joint
        self.to_patch, self.patch_to_emb = encoder.to_joint_embedding[:2]
        time_values_per_joint = self.patch_to_emb.weight.shape[-1]

        # simple linear head

        self.mask_token = nn.Parameter(torch.randn(encoder_dim))
        self.to_times = nn.Linear(encoder_dim, time_values_per_joint)

    def forward(self, img, return_attention=False):
        device = img.device

        # get patches
        # print(torch.sum(torch.isnan(img)))
        # img = self.rep_to_patch(img)
        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape
        # num_patches = self.num_joint
        # for indexing purposes

        batch_range = torch.arange(batch, device = device)[:, None]

        # get positions

        pos_emb = self.encoder.pos_embedding[:, 1:(num_patches + 1)]
        # patch to encoder tokens and add positions
        tokens = self.patch_to_emb(patches)
        tokens = tokens + pos_emb
        # prepare mask tokens
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_patches)
        mask_tokens = mask_tokens + pos_emb
        # calculate of patches needed to be masked, and get positions (indices) to be masked

        num_masked = int(self.masking_ratio * num_patches)
        masked_indices = torch.rand(batch, num_patches, device = device).topk(k = num_masked, dim = -1).indices
        masked_bool_mask = torch.zeros((batch, num_patches), device = device).scatter_(-1, masked_indices, 1).bool()

        # mask tokens

        tokens = torch.where(masked_bool_mask[..., None], mask_tokens, tokens)
        # attend with vision transformer
        encoded = self.encoder.transformer(tokens, return_attention)
        if return_attention:
            return encoded
        # get the masked tokens

        encoded_mask_tokens = encoded[batch_range, masked_indices]

        # small linear projection for predicted pixel values
        pred_time_values = self.to_times(encoded_mask_tokens)
        # get the masked patches for the final reconstruction loss

        masked_patches = patches[batch_range, masked_indices]

        # calculate reconstruction loss
        recon_loss = F.l1_loss(pred_time_values, masked_patches) / num_masked
        return recon_loss
class SimMIM_patch(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        masking_ratio = 0.5
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_joint, encoder_dim = encoder.pos_embedding.shape[-2:]
        # self.rep_to_patch = encoder.rep_to_joint
        self.to_patch, self.patch_to_emb = encoder.to_patch_embedding[:2]
        time_values_per_joint = self.patch_to_emb.weight.shape[-1]

        # simple linear head

        self.mask_token = nn.Parameter(torch.randn(encoder_dim))
        self.to_times = nn.Linear(encoder_dim, time_values_per_joint)

    def forward(self, img, return_attention=False):
        device = img.device

        # get patches
        # img = self.rep_to_patch(img)
        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape
        # num_patches = self.num_joint
        # for indexing purposes

        batch_range = torch.arange(batch, device = device)[:, None]

        # get positions

        pos_emb = self.encoder.pos_embedding[:, 1:(num_patches + 1)]
        # patch to encoder tokens and add positions
        tokens = self.patch_to_emb(patches)
        tokens = tokens + pos_emb
        # prepare mask tokens
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_patches)
        mask_tokens = mask_tokens + pos_emb

        # calculate of patches needed to be masked, and get positions (indices) to be masked

        num_masked = int(self.masking_ratio * num_patches)
        masked_indices = torch.rand(batch, num_patches, device = device).topk(k = num_masked, dim = -1).indices
        masked_bool_mask = torch.zeros((batch, num_patches), device = device).scatter_(-1, masked_indices, 1).bool()

        # mask tokens

        tokens = torch.where(masked_bool_mask[..., None], mask_tokens, tokens)

        # attend with vision transformer

        encoded = self.encoder.transformer(tokens, return_attention)
        if return_attention:
            return encoded
        # get the masked tokens

        encoded_mask_tokens = encoded[batch_range, masked_indices]

        # small linear projection for predicted pixel values

        pred_time_values = self.to_times(encoded_mask_tokens)

        # get the masked patches for the final reconstruction loss

        masked_patches = patches[batch_range, masked_indices]

        # calculate reconstruction loss

        recon_loss = F.l1_loss(pred_time_values, masked_patches) / num_masked
        return recon_loss
class SimMIM_patch2(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        masking_ratio = 0.5
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_joint, encoder_dim = encoder.pos_embedding.shape[-2:]
        # self.rep_to_patch = encoder.rep_to_joint
        self.to_patch, self.patch_to_emb = encoder.to_patch_embedding[:2]
        time_values_per_joint = self.patch_to_emb.weight.shape[-1]

        # simple linear head

        self.mask_token = nn.Parameter(torch.randn(encoder_dim))
        self.to_times = nn.Linear(encoder_dim, time_values_per_joint)

    def forward(self, img, return_attention=False):
        device = img.device

        # get patches
        # img = self.rep_to_patch(img)
        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape
        # num_patches = self.num_joint
        # for indexing purposes

        batch_range = torch.arange(batch, device = device)[:, None]

        # get positions

        pos_emb = self.encoder.pos_embedding[:, 1:(num_patches + 1)]
        # patch to encoder tokens and add positions
        # print('patches', patches.shape)
        tokens = self.patch_to_emb(patches)
        # print('tokens', tokens.shape)
        tokens = tokens + pos_emb
        # prepare mask tokens
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_patches)
        mask_tokens = mask_tokens + pos_emb

        # calculate of patches needed to be masked, and get positions (indices) to be masked

        num_masked = int(self.masking_ratio * num_patches)
        masked_indices = torch.rand(batch, num_patches, device = device).topk(k = num_masked, dim = -1).indices
        masked_bool_mask = torch.zeros((batch, num_patches), device = device).scatter_(-1, masked_indices, 1).bool()

        # mask tokens

        tokens = torch.where(masked_bool_mask[..., None], mask_tokens, tokens)

        # attend with vision transformer

        encoded = self.encoder.transformer(tokens, return_attention)
        # print('return_attention', return_attention)
        # print('encoded', encoded.shape)
        if return_attention:
            return encoded
        # get the masked tokens

        encoded_mask_tokens = encoded[batch_range, masked_indices]

        # small linear projection for predicted pixel values

        pred_time_values = self.to_times(encoded_mask_tokens)

        # get the masked patches for the final reconstruction loss

        masked_patches = patches[batch_range, masked_indices]
        # print('encoded_mask_tokens', encoded_mask_tokens.shape)
        # calculate reconstruction loss
        # print('pred_time_values', pred_time_values.shape)
        recon_loss = F.l1_loss(pred_time_values, masked_patches) / num_masked
        return recon_loss      
class SimMIM_patch2_seg(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        masking_ratio = 0.5
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_joint, encoder_dim = encoder.pos_emb.weight.shape[-2:]
        # self.rep_to_patch = encoder.rep_to_joint
        self.to_patch, self.patch_to_emb = encoder.to_patch_embedding[:2]
        time_values_per_joint = self.patch_to_emb.weight.shape[-1]

        # simple linear head

        self.mask_token = nn.Parameter(torch.randn(encoder_dim))
        self.to_times = nn.Linear(encoder_dim, time_values_per_joint)

    def forward(self, img, valid_length, return_attention=False):
        device = img.device

        # get patches
        # img = self.rep_to_patch(img)
        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape
        # num_patches = self.num_joint
        # for indexing purposes
        pos_x = torch.arange(num_patches+1, device=patches.device).repeat((batch, 1))

        batch_range = torch.arange(batch, device = device)[:, None]

        # get positions
        valid_segment = []
        for i in valid_length:
            valid_segment.append([1] + [1] * torch.floor_divide(i,5) + [0] * (num_patches - torch.floor_divide(i,5)))
        # print(len(valid_segment[0]))
        seg_tensor = torch.IntTensor(valid_segment).to(device)
        pos_emb = self.encoder.pos_emb(pos_x) + self.encoder.val_emb(seg_tensor)
        pos_emb = pos_emb[:, 1:(num_patches + 1)]

        # patch to encoder tokens and add positions
        tokens = self.patch_to_emb(patches)
        # print('tokens', tokens.shape)
        tokens = tokens + pos_emb
        # prepare mask tokens
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_patches)
        mask_tokens = mask_tokens + pos_emb

        # calculate of patches needed to be masked, and get positions (indices) to be masked

        num_masked = int(self.masking_ratio * num_patches)
        masked_indices = torch.rand(batch, num_patches, device = device).topk(k = num_masked, dim = -1).indices
        masked_bool_mask = torch.zeros((batch, num_patches), device = device).scatter_(-1, masked_indices, 1).bool()

        # mask tokens

        tokens = torch.where(masked_bool_mask[..., None], mask_tokens, tokens)

        # attend with vision transformer

        encoded = self.encoder.transformer(tokens, return_attention)
        # print('return_attention', return_attention)
        # print('encoded', encoded.shape)
        if return_attention:
            return encoded
        # get the masked tokens

        encoded_mask_tokens = encoded[batch_range, masked_indices]

        # small linear projection for predicted pixel values

        pred_time_values = self.to_times(encoded_mask_tokens)

        # get the masked patches for the final reconstruction loss

        masked_patches = patches[batch_range, masked_indices]
        # print('encoded_mask_tokens', encoded_mask_tokens.shape)
        # calculate reconstruction loss
        # print('pred_time_values', pred_time_values.shape)
        recon_loss = F.l1_loss(pred_time_values, masked_patches) / num_masked
        return recon_loss              

class SimMIM_patch2_seg_chn(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        masking_ratio = 0.5
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_joint, encoder_dim = encoder.pos_emb.weight.shape[-2:]
        # self.rep_to_patch = encoder.rep_to_joint
        self.to_patch, self.patch_to_emb = encoder.to_patch_embedding[:2]
        time_values_per_joint = self.patch_to_emb.weight.shape[-1]

        # simple linear head

        self.mask_token = nn.Parameter(torch.randn(encoder_dim))
        self.to_times = nn.Linear(encoder_dim, time_values_per_joint)

    def forward(self, img, valid_length, return_attention=False):
        device = img.device
        B, nc, h, w = img.shape
        # get patches
        # img = self.rep_to_patch(img)
        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape
        # num_patches = self.num_joint
        # for indexing purposes
        pos_x = torch.arange(num_patches+1, device=patches.device).repeat((batch, 1))

        batch_range = torch.arange(batch, device = device)[:, None]
        sub_seq = self.encoder.sub_seq
        # get positions
        valid_segment = []
        for i in valid_length:
            if i == 150 :
                i = 149
            valid_segment.append([1] + h*([1] * torch.ceil(torch.div(i, sub_seq)).int() + [0] * (int(w/sub_seq) - torch.ceil(torch.div(i, sub_seq)).int())))
        seg_tensor = torch.IntTensor(valid_segment).to(device)
        pos_emb = self.encoder.pos_emb(pos_x) + self.encoder.val_emb(seg_tensor)
        pos_emb = pos_emb[:, 1:(num_patches + 1)]

        # patch to encoder tokens and add positions
        tokens = self.patch_to_emb(patches)
        # print('tokens', tokens.shape)
        tokens = tokens + pos_emb
        # prepare mask tokens
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_patches)
        mask_tokens = mask_tokens + pos_emb

        # calculate of patches needed to be masked, and get positions (indices) to be masked

        num_masked = int(self.masking_ratio * num_patches)
        masked_indices = torch.rand(batch, num_patches, device = device).topk(k = num_masked, dim = -1).indices
        masked_bool_mask = torch.zeros((batch, num_patches), device = device).scatter_(-1, masked_indices, 1).bool()

        # mask tokens

        tokens = torch.where(masked_bool_mask[..., None], mask_tokens, tokens)

        # attend with vision transformer

        encoded = self.encoder.transformer(tokens, return_attention)
        # print('return_attention', return_attention)
        # print('encoded', encoded.shape)
        if return_attention:
            return encoded
        # get the masked tokens

        encoded_mask_tokens = encoded[batch_range, masked_indices]

        # small linear projection for predicted pixel values

        pred_time_values = self.to_times(encoded_mask_tokens)

        # get the masked patches for the final reconstruction loss

        masked_patches = patches[batch_range, masked_indices]
        # print('encoded_mask_tokens', encoded_mask_tokens.shape)
        # calculate reconstruction loss
        # print('pred_time_values', pred_time_values.shape)
        recon_loss = F.l1_loss(pred_time_values, masked_patches) / num_masked
        return recon_loss   

class SimMIM_patch2_seg_chn_sj(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        masking_ratio = 0.5
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_joint, encoder_dim = encoder.pos_emb.weight.shape[-2:]
        # self.rep_to_patch = encoder.rep_to_joint
        self.to_patch, self.patch_to_emb = encoder.to_patch_embedding[:2]
        time_values_per_joint = self.patch_to_emb.weight.shape[-1]

        # simple linear head

        self.mask_token = nn.Parameter(torch.randn(encoder_dim))
        self.to_times = nn.Linear(encoder_dim, time_values_per_joint)

    def forward(self, img, valid_length, return_attention=False):
        device = img.device

        # get patches
        # img = self.rep_to_patch(img)
        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape
        # num_patches = self.num_joint
        # for indexing purposes
        pos_x = torch.arange(num_patches+1, device=patches.device).repeat((batch, 1))

        batch_range = torch.arange(batch, device = device)[:, None]

        # get positions
        # valid_segment = []
        # for i in valid_length:
            # valid_segment.append([1] + [1] * torch.floor_divide(i,5) + [0] * (num_patches - torch.floor_divide(i,5)))
        # print(len(valid_segment[0]))
        # seg_tensor = torch.IntTensor(valid_segment).to(device)
        pos_emb = self.encoder.pos_emb(pos_x) 
        pos_emb = pos_emb[:, 1:(num_patches + 1)]

        # patch to encoder tokens and add positions
        tokens = self.patch_to_emb(patches)
        # print('tokens', tokens.shape)
        tokens = tokens + pos_emb
        # prepare mask tokens
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_patches)
        mask_tokens = mask_tokens + pos_emb

        # calculate of patches needed to be masked, and get positions (indices) to be masked

        num_masked = int(self.masking_ratio * num_patches)
        masked_indices = torch.rand(batch, num_patches, device = device).topk(k = num_masked, dim = -1).indices
        masked_bool_mask = torch.zeros((batch, num_patches), device = device).scatter_(-1, masked_indices, 1).bool()

        # mask tokens

        tokens = torch.where(masked_bool_mask[..., None], mask_tokens, tokens)

        # attend with vision transformer

        encoded = self.encoder.transformer(tokens, return_attention)
        # print('return_attention', return_attention)
        # print('encoded', encoded.shape)
        if return_attention:
            return encoded
        # get the masked tokens

        encoded_mask_tokens = encoded[batch_range, masked_indices]

        # small linear projection for predicted pixel values

        pred_time_values = self.to_times(encoded_mask_tokens)

        # get the masked patches for the final reconstruction loss

        masked_patches = patches[batch_range, masked_indices]
        # print('encoded_mask_tokens', encoded_mask_tokens.shape)
        # calculate reconstruction loss
        # print('pred_time_values', pred_time_values.shape)
        recon_loss = F.l1_loss(pred_time_values, masked_patches) / num_masked
        return recon_loss      

class SimMIM_patch2_seg_chn_moe(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        masking_ratio = 0.5
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_joint, encoder_dim = encoder.pos_emb.weight.shape[-2:]
        # self.rep_to_patch = encoder.rep_to_joint
        self.to_patch, self.patch_to_emb = encoder.to_patch_embedding[:2]
        time_values_per_joint = self.patch_to_emb.weight.shape[-1]

        # simple linear head

        self.mask_token = nn.Parameter(torch.randn(encoder_dim))
        self.to_times = nn.Linear(encoder_dim, time_values_per_joint)

    def forward(self, img, valid_length, return_attention=False):
        device = img.device

        # get patches
        # img = self.rep_to_patch(img)
        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape
        # num_patches = self.num_joint
        # for indexing purposes
        pos_x = torch.arange(num_patches+1, device=patches.device).repeat((batch, 1))

        batch_range = torch.arange(batch, device = device)[:, None]

        # get positions
        valid_segment = []
        for i in valid_length:
            valid_segment.append([1] + [1] * torch.floor_divide(i,5) + [0] * (num_patches - torch.floor_divide(i,5)))
        # print(len(valid_segment[0]))
        seg_tensor = torch.IntTensor(valid_segment).to(device)
        pos_emb = self.encoder.pos_emb(pos_x) + self.encoder.val_emb(seg_tensor)
        pos_emb = pos_emb[:, 1:(num_patches + 1)]

        # patch to encoder tokens and add positions
        tokens = self.patch_to_emb(patches)
        # print('tokens', tokens.shape)
        tokens = tokens + pos_emb
        # prepare mask tokens
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_patches)
        mask_tokens = mask_tokens + pos_emb

        # calculate of patches needed to be masked, and get positions (indices) to be masked

        num_masked = int(self.masking_ratio * num_patches)
        masked_indices = torch.rand(batch, num_patches, device = device).topk(k = num_masked, dim = -1).indices
        masked_bool_mask = torch.zeros((batch, num_patches), device = device).scatter_(-1, masked_indices, 1).bool()

        # mask tokens

        tokens = torch.where(masked_bool_mask[..., None], mask_tokens, tokens)

        # attend with vision transformer

        encoded = self.encoder.transformer(tokens, return_attention)
        # print('return_attention', return_attention)
        # print('encoded', encoded.shape)
        if return_attention:
            return encoded
        # get the masked tokens

        encoded_mask_tokens = encoded[batch_range, masked_indices]

        # small linear projection for predicted pixel values

        pred_time_values = self.to_times(encoded_mask_tokens)

        # get the masked patches for the final reconstruction loss

        masked_patches = patches[batch_range, masked_indices]
        # print('encoded_mask_tokens', encoded_mask_tokens.shape)
        # calculate reconstruction loss
        # print('pred_time_values', pred_time_values.shape)
        recon_loss = F.l1_loss(pred_time_values, masked_patches) / num_masked
        return recon_loss                              
class SimMIM_seg(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        masking_ratio = 0.5
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_joint, encoder_dim = encoder.pos_emb.weight.shape[-2:]
        # num_joint, encoder_dim = encoder.pos_embedding.shape[-2:]
        # self.rep_to_patch = encoder.rep_to_joint
        self.to_patch, self.patch_to_emb = encoder.to_joint_embedding[:2]
        time_values_per_joint = self.patch_to_emb.weight.shape[-1]

        # simple linear head

        self.mask_token = nn.Parameter(torch.randn(encoder_dim))
        self.to_times = nn.Linear(encoder_dim, time_values_per_joint)

    def forward(self, img, valid_length, return_attention=False):
        device = img.device

        # get patches
        # print(torch.sum(torch.isnan(img)))
        # img = self.rep_to_patch(img)
        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape
        # num_patches = self.num_joint
        # for indexing purposes

        batch_range = torch.arange(batch, device = device)[:, None]

        # get positions
        pos_x = torch.arange(num_patches+1, device=patches.device).repeat((batch, 1))
        batch_range = torch.arange(batch, device = device)[:, None]

        # get positions
        valid_segment = []
        for i in valid_length:
            valid_segment.append([1] + [1] * i + [0] * (num_patches - i))
        # print(len(valid_segment[0]))
        seg_tensor = torch.IntTensor(valid_segment).to(device)
        pos_emb = self.encoder.pos_emb(pos_x) + self.encoder.val_emb(seg_tensor)
        pos_emb = pos_emb[:, 1:(num_patches + 1)]


        # pos_emb = self.encoder.pos_embedding[:, 1:(num_patches + 1)]
        # patch to encoder tokens and add positions
        tokens = self.patch_to_emb(patches)
        tokens = tokens + pos_emb
        # prepare mask tokens
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_patches)
        mask_tokens = mask_tokens + pos_emb
        # calculate of patches needed to be masked, and get positions (indices) to be masked

        num_masked = int(self.masking_ratio * num_patches)
        masked_indices = torch.rand(batch, num_patches, device = device).topk(k = num_masked, dim = -1).indices
        masked_bool_mask = torch.zeros((batch, num_patches), device = device).scatter_(-1, masked_indices, 1).bool()

        # mask tokens

        tokens = torch.where(masked_bool_mask[..., None], mask_tokens, tokens)
        # attend with vision transformer
        encoded = self.encoder.transformer(tokens, return_attention)
        if return_attention:
            return encoded
        # get the masked tokens

        encoded_mask_tokens = encoded[batch_range, masked_indices]

        # small linear projection for predicted pixel values
        pred_time_values = self.to_times(encoded_mask_tokens)
        # get the masked patches for the final reconstruction loss

        masked_patches = patches[batch_range, masked_indices]

        # calculate reconstruction loss
        recon_loss = F.l1_loss(pred_time_values, masked_patches) / num_masked
        return recon_loss                        
class SimMIM_pos(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        masking_ratio = 0.5
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_joint, encoder_dim = encoder.pos_embedding.shape[-2:]
        self.rep_to_patch = encoder.rep_to_joint
        self.to_patch, self.patch_to_emb = encoder.to_joint_embedding[:2]
        time_values_per_joint = self.patch_to_emb.weight.shape[-1]

        # simple linear head

        self.mask_token = nn.Parameter(torch.randn(encoder_dim))
        self.to_times = nn.Linear(encoder_dim, time_values_per_joint)

    def forward(self, img, return_attention=False):
        device = img.device

        # get patches
        img = self.rep_to_patch(img)
        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape
        # num_patches = self.num_joint
        # for indexing purposes

        batch_range = torch.arange(batch, device = device)[:, None]

        # get positions

        pos_emb = self.encoder.pos_embedding[:, 1:(num_patches + 1)]
        # patch to encoder tokens and add positions
        tokens = self.patch_to_emb(patches)
        tokens = tokens + pos_emb
        # prepare mask tokens
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_patches)
        mask_tokens = mask_tokens + pos_emb

        # calculate of patches needed to be masked, and get positions (indices) to be masked

        num_masked = int(self.masking_ratio * num_patches)
        masked_indices = torch.rand(batch, num_patches, device = device).topk(k = num_masked, dim = -1).indices
        masked_bool_mask = torch.zeros((batch, num_patches), device = device).scatter_(-1, masked_indices, 1).bool()

        # mask tokens

        tokens = torch.where(masked_bool_mask[..., None], mask_tokens, tokens)

        # attend with vision transformer

        encoded = self.encoder.transformer(tokens, return_attention)
        if return_attention:
            return encoded
        # get the masked tokens

        encoded_mask_tokens = encoded[batch_range, masked_indices]

        # small linear projection for predicted pixel values

        pred_time_values = self.to_times(encoded_mask_tokens)

        # get the masked patches for the final reconstruction loss

        masked_patches = patches[batch_range, masked_indices]

        # calculate reconstruction loss

        recon_loss = F.l1_loss(pred_time_values, masked_patches) / num_masked
        return recon_loss        
class SimMIM_pos2(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        masking_ratio = 0.5
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_joint, encoder_dim = encoder.pos_embedding.shape[-2:]
        # self.rep_to_patch = encoder.rep_to_joint
        self.to_patch, self.patch_to_emb = encoder.to_joint_embedding[:2]
        time_values_per_joint = self.patch_to_emb.weight.shape[-1]

        # simple linear head

        self.mask_token = nn.Parameter(torch.randn(encoder_dim))
        self.to_times = nn.Linear(encoder_dim, time_values_per_joint)

    def forward(self, img, return_attention=False):
        device = img.device

        # get patches
        # img = self.rep_to_patch(img)
        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape
        # num_patches = self.num_joint
        # for indexing purposes

        batch_range = torch.arange(batch, device = device)[:, None]

        # get positions

        pos_emb = self.encoder.pos_embedding[:, 1:(num_patches + 1)]
        # patch to encoder tokens and add positions
        tokens = self.patch_to_emb(patches)
        tokens = tokens + pos_emb
        # prepare mask tokens
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_patches)
        mask_tokens = mask_tokens + pos_emb

        # calculate of patches needed to be masked, and get positions (indices) to be masked

        num_masked = int(self.masking_ratio * num_patches)
        masked_indices = torch.rand(batch, num_patches, device = device).topk(k = num_masked, dim = -1).indices
        masked_bool_mask = torch.zeros((batch, num_patches), device = device).scatter_(-1, masked_indices, 1).bool()

        # mask tokens

        tokens = torch.where(masked_bool_mask[..., None], mask_tokens, tokens)

        # attend with vision transformer

        encoded = self.encoder.transformer(tokens, return_attention)
        if return_attention:
            return encoded
        # get the masked tokens

        encoded_mask_tokens = encoded[batch_range, masked_indices]

        # small linear projection for predicted pixel values

        pred_time_values = self.to_times(encoded_mask_tokens)

        # get the masked patches for the final reconstruction loss

        masked_patches = patches[batch_range, masked_indices]

        # calculate reconstruction loss

        recon_loss = F.l1_loss(pred_time_values, masked_patches) / num_masked
        return recon_loss
class SimMIM_pos3(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        masking_ratio = 0.5
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_joint, encoder_dim = encoder.pos_embedding.shape[-2:]
        # self.rep_to_patch = encoder.rep_to_joint
        self.to_patch, self.patch_to_emb = encoder.to_joint_embedding[:2]
        time_values_per_joint = encoder_dim

        # simple linear head

        self.mask_token = nn.Parameter(torch.randn(encoder_dim))
        self.to_times = nn.Linear(encoder_dim, time_values_per_joint)

    def forward(self, img, return_attention=False):
        device = img.device

        # get patches
        # img = self.rep_to_patch(img)
        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape
        # num_patches = self.num_joint
        # for indexing purposes

        batch_range = torch.arange(batch, device = device)[:, None]

        # get positions

        pos_emb = self.encoder.pos_embedding[:, 1:(num_patches + 1)]
        # patch to encoder tokens and add positions
        tokens = self.patch_to_emb(patches)
        tokens = tokens + pos_emb
        # prepare mask tokens
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_patches)
        mask_tokens = mask_tokens + pos_emb

        # calculate of patches needed to be masked, and get positions (indices) to be masked

        num_masked = int(self.masking_ratio * num_patches)
        masked_indices = torch.rand(batch, num_patches, device = device).topk(k = num_masked, dim = -1).indices
        masked_bool_mask = torch.zeros((batch, num_patches), device = device).scatter_(-1, masked_indices, 1).bool()

        # mask tokens

        tokens = torch.where(masked_bool_mask[..., None], mask_tokens, tokens)

        # attend with vision transformer

        encoded = self.encoder.transformer(tokens, return_attention)
        if return_attention:
            return encoded
        # get the masked tokens

        encoded_mask_tokens = encoded[batch_range, masked_indices]

        # small linear projection for predicted pixel values

        pred_time_values = self.to_times(encoded_mask_tokens)

        # get the masked patches for the final reconstruction loss

        masked_patches = patches[batch_range, masked_indices]

        # calculate reconstruction loss
    
        recon_loss = F.l1_loss(pred_time_values, masked_patches) / num_masked
        return recon_loss

class SimMIM_6d2(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        masking_ratio = 0.5
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_joint, encoder_dim = encoder.pos_embedding.shape[-2:]
        # self.rep_to_patch = encoder.rep_to_joint
        self.to_patch, self.patch_to_emb = encoder.to_joint_embedding[:2]
        time_values_per_joint = self.patch_to_emb.weight.shape[-1]

        # simple linear head

        self.mask_token = nn.Parameter(torch.randn(encoder_dim))
        self.to_times = nn.Linear(encoder_dim, time_values_per_joint)

    def forward(self, img, return_attention=False):
        device = img.device
        # get patches
        # img = self.rep_to_patch(img)
        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape
        # num_patches = self.num_joint
        # for indexing purposes

        batch_range = torch.arange(batch, device = device)[:, None]

        # get positions

        pos_emb = self.encoder.pos_embedding[:, 1:(num_patches + 1)]
        # patch to encoder tokens and add positions
        tokens = self.patch_to_emb(patches)
        tokens = tokens + pos_emb
        # prepare mask tokens
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_patches)
        mask_tokens = mask_tokens + pos_emb
        # calculate of patches needed to be masked, and get positions (indices) to be masked

        num_masked = int(self.masking_ratio * num_patches)
        masked_indices = torch.rand(batch, num_patches, device = device).topk(k = num_masked, dim = -1).indices
        masked_bool_mask = torch.zeros((batch, num_patches), device = device).scatter_(-1, masked_indices, 1).bool()

        # mask tokens

        tokens = torch.where(masked_bool_mask[..., None], mask_tokens, tokens)

        # attend with vision transformer
        encoded = self.encoder.transformer(tokens, return_attention)
        if return_attention:
            return encoded
        # get the masked tokens

        encoded_mask_tokens = encoded[batch_range, masked_indices]

        # small linear projection for predicted pixel values
        pred_time_values = self.to_times(encoded_mask_tokens)

        # get the masked patches for the final reconstruction loss

        masked_patches = patches[batch_range, masked_indices]
        # calculate reconstruction loss

        recon_loss = F.l1_loss(pred_time_values, masked_patches) / num_masked
        return recon_loss
class SimMIM_6d3(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        masking_ratio = 0.5
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_joint, encoder_dim = encoder.pos_embedding.shape[-2:]
        # self.rep_to_patch = encoder.rep_to_joint
        self.to_patch, self.patch_to_emb = encoder.to_joint_embedding[:2]
        time_values_per_joint = encoder_dim

        # simple linear head

        self.mask_token = nn.Parameter(torch.randn(encoder_dim))
        self.to_times = nn.Linear(encoder_dim, time_values_per_joint)

    def forward(self, img, return_attention=False):
        device = img.device

        # get patches
        # img = self.rep_to_patch(img)
        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape
        # num_patches = self.num_joint
        # for indexing purposes

        batch_range = torch.arange(batch, device = device)[:, None]

        # get positions

        pos_emb = self.encoder.pos_embedding[:, 1:(num_patches + 1)]
        # patch to encoder tokens and add positions
        tokens = self.patch_to_emb(patches)
        tokens = tokens + pos_emb
        # prepare mask tokens
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_patches)
        mask_tokens = mask_tokens + pos_emb

        # calculate of patches needed to be masked, and get positions (indices) to be masked

        num_masked = int(self.masking_ratio * num_patches)
        masked_indices = torch.rand(batch, num_patches, device = device).topk(k = num_masked, dim = -1).indices
        masked_bool_mask = torch.zeros((batch, num_patches), device = device).scatter_(-1, masked_indices, 1).bool()

        # mask tokens

        tokens = torch.where(masked_bool_mask[..., None], mask_tokens, tokens)

        # attend with vision transformer

        encoded = self.encoder.transformer(tokens, return_attention)
        if return_attention:
            return encoded
        # get the masked tokens

        encoded_mask_tokens = encoded[batch_range, masked_indices]

        # small linear projection for predicted pixel values

        pred_time_values = self.to_times(encoded_mask_tokens)

        # get the masked patches for the final reconstruction loss

        masked_patches = patches[batch_range, masked_indices]

        # calculate reconstruction loss

        recon_loss = F.l1_loss(pred_time_values, masked_patches) / num_masked
        return recon_loss

class SimMIM_cmib(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        masking_ratio = 0.5
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_joint, encoder_dim = encoder.pos_embedding.shape[-2:]
        self.to_patch, self.patch_to_emb = encoder.to_joint_embedding[:2]
        time_values_per_joint = self.patch_to_emb.weight.shape[-1]

        # simple linear head

        self.mask_token = nn.Parameter(torch.randn(encoder_dim))
        self.to_times = nn.Linear(encoder_dim, time_values_per_joint)

    def forward(self, img, return_attention=False):
        device = img.device

        # get patches
        num_patches = self.num_joint
        # for indexing purposes


        # get positions

        pos_emb = self.encoder.pos_embedding[:, 1:(num_patches + 1)]
        batch = pos_emb.shape[0]
        batch_range = torch.arange(batch, device = device)[:, None]

        # patch to encoder tokens and add positions
        tokens = self.patch_to_emb(patches)
        tokens = tokens + pos_emb
        # prepare mask tokens
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_patches)
        mask_tokens = mask_tokens + pos_emb

        # calculate of patches needed to be masked, and get positions (indices) to be masked

        num_masked = int(self.masking_ratio * num_patches)
        masked_indices = torch.rand(batch, num_patches, device = device).topk(k = num_masked, dim = -1).indices
        masked_bool_mask = torch.zeros((batch, num_patches), device = device).scatter_(-1, masked_indices, 1).bool()

        # mask tokens

        tokens = torch.where(masked_bool_mask[..., None], mask_tokens, tokens)

        # attend with vision transformer

        encoded = self.encoder.transformer(tokens, return_attention)
        if return_attention:
            return encoded
        # get the masked tokens

        encoded_mask_tokens = encoded[batch_range, masked_indices]

        # small linear projection for predicted pixel values

        pred_time_values = self.to_times(encoded_mask_tokens)

        # get the masked patches for the final reconstruction loss

        masked_patches = patches[batch_range, masked_indices]

        # calculate reconstruction loss

        recon_loss = F.l1_loss(pred_time_values, masked_patches) / num_masked
        return recon_loss        