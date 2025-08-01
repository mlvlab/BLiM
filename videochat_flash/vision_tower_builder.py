from typing import Optional, Tuple, Union, Dict
from dataclasses import dataclass
from functools import partial, reduce
from PIL import Image
import os
from transformers.image_processing_utils import BatchFeature, get_size_dict
from transformers.image_transforms import (
    convert_to_rgb,
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    to_numpy_array,
)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from functools import partial
try:
    from flash_attn import flash_attn_qkvpacked_func
except:
    print("You need to install flash_attn")
from timm.layers import drop_path, to_2tuple, trunc_normal_



class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None,
            attn_type='flash_v2'):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        if attn_type not in ['origin', 'flash_v2']:
            raise NotImplementedError(f"Not support attn_type: {attn_type}")

        # print('umt:', f'attn_type: {attn_type}')
        
        self.attn_type = attn_type
        if attn_type == 'flash_v2':
            self.attn_drop = attn_drop
        else:
            self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        
        if self.attn_type == 'flash_v2':
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1)
            x = flash_attn_qkvpacked_func(qkv, dropout_p=self.attn_drop, softmax_scale=self.scale, causal=False).reshape(B, N, -1)
        else:
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[
                2]  # make torchscript happy (cannot use tensor as tuple)
            # B num_heads N head_dim

            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    



class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (num_frames // self.tubelet_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(
            in_channels=in_chans, out_channels=embed_dim, 
            kernel_size=(self.tubelet_size, patch_size[0], patch_size[1]), 
            stride=(self.tubelet_size, patch_size[0], patch_size[1])
        )
        # print('umt:', f'Num of patches: {num_patches}')

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    
# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid, ckpt_num_frame=-1, cur_frame=12): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 
    
    if ckpt_num_frame != -1 and ckpt_num_frame != cur_frame:
        # print('umt:', f"Interpolate position embedding")
        # print('umt:', f"Testing frame: {cur_frame}")
        # print('umt:', f"Checkpoint frame: {ckpt_num_frame}")

        T = ckpt_num_frame # checkpoint frame
        new_T = cur_frame # testing frame
        n_position = n_position // new_T * T # generate checkpoint position embedding
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 
        sinusoid_table = torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)
        # interpolate
        P = int((n_position // T) ** 0.5)
        C = d_hid
        sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
        sinusoid_table = sinusoid_table.permute(0, 2, 3, 4, 1).reshape(-1, C, T)  # BHW, C, T
        sinusoid_table = torch.nn.functional.interpolate(sinusoid_table, size=new_T, mode='linear')
        sinusoid_table = sinusoid_table.reshape(1, P, P, C, new_T).permute(0, 4, 1, 2, 3) # B, T, H, W, C
        sinusoid_table = sinusoid_table.flatten(1, 3)
        return sinusoid_table
    else:
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 
        return torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0) 
    

def get_sinusoid_encoding_table2(n_position=784, d_hid=1024, cur_frame=8, ckpt_num_frame=4, pre_n_position=784): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 
    
    # generate checkpoint position embedding
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(pre_n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 
    sinusoid_table = torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)
    
    # print(f"n_position: {n_position}")
    # print(f"pre_n_position: {pre_n_position}")
    
    if n_position != pre_n_position:
        T = ckpt_num_frame # checkpoint frame
        P = 14 # checkpoint size
        C = d_hid
        new_P = int((n_position // cur_frame) ** 0.5) # testing size
        # print(f'Pretraining uses 14x14, but current version is {new_P}x{new_P}')
        # print(f'Interpolate the position embedding')
        sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
        sinusoid_table = sinusoid_table.reshape(-1, P, P, C).permute(0, 3, 1, 2)
        sinusoid_table = torch.nn.functional.interpolate(
            sinusoid_table, size=(new_P, new_P), mode='bicubic', align_corners=False)
        # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
        sinusoid_table = sinusoid_table.permute(0, 2, 3, 1).reshape(-1, T, new_P, new_P, C)
        sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C
    
    if cur_frame != ckpt_num_frame:
        # print(f'Pretraining uses 4 frames, but current frame is {cur_frame}')
        # print(f'Interpolate the position embedding')
        T = ckpt_num_frame # checkpoint frame
        new_T = cur_frame # testing frame
        # interpolate
        P = int((n_position // cur_frame) ** 0.5) # testing size
        C = d_hid
        sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
        sinusoid_table = sinusoid_table.permute(0, 2, 3, 4, 1).reshape(-1, C, T)  # BHW, C, T
        sinusoid_table = torch.nn.functional.interpolate(sinusoid_table, size=new_T, mode='linear')
        sinusoid_table = sinusoid_table.reshape(1, P, P, C, new_T).permute(0, 4, 1, 2, 3) # B, T, H, W, C
        sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C
        
    return sinusoid_table


class PretrainVisionTransformerEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, num_frames=8, tubelet_size=1,
                 use_learnable_pos_emb=False,
                 use_checkpoint=False, checkpoint_num=0, 
                 ckpt_num_frame=-1, with_ln=True, return_index=-1
                 ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, 
            num_frames=num_frames, tubelet_size=tubelet_size
        )
        num_patches = self.patch_embed.num_patches
        self.depth = depth + return_index + 1
        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num
        # print('umt:', f"Use checkpoint: {use_checkpoint}")
        # print('umt:', f"Checkpoint number: {checkpoint_num}")
        # print('UMT:', f"Real runing depth: {self.depth}")

        # TODO: Add the cls token
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            self.img_pos_embed = nn.Parameter(torch.zeros(1, num_patches//(num_frames//tubelet_size) + 1, embed_dim))
        else:
            # sine-cosine positional embeddings 
            if img_size != 224:
                self.pos_embed = get_sinusoid_encoding_table2(num_patches, embed_dim, ckpt_num_frame=ckpt_num_frame, cur_frame=num_frames//tubelet_size)
                self.img_pos_embed = get_sinusoid_encoding_table2(num_patches//(num_frames//tubelet_size), embed_dim, cur_frame=1, ckpt_num_frame=1, pre_n_position=14*14)
            else:
                self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim, ckpt_num_frame=ckpt_num_frame, cur_frame=num_frames//tubelet_size)
                self.img_pos_embed = get_sinusoid_encoding_table(num_patches//(num_frames//tubelet_size), embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(self.depth)])
        
        if with_ln:
            self.vision_layernorm = nn.LayerNorm(embed_dim, eps=1e-12)
        else:
            self.vision_layernorm = nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x, use_image=False):
        x = self.patch_embed(x)
        
        if use_image:
            x = x + self.img_pos_embed.type_as(x).to(x.device).clone().detach()
        else:
            x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()

        B, _, C = x.shape
        x_vis = x

        for idx, blk in enumerate(self.blocks):
            if self.use_checkpoint and idx < self.checkpoint_num:
                x_vis = checkpoint.checkpoint(blk, x_vis)
            else:
                x_vis = blk(x_vis)

        # with ln ot not
        x_vis = self.vision_layernorm(x_vis)
        return x_vis

    def forward(self, x, use_image=False):
        x_vis = self.forward_features(x, use_image)
        return x_vis


class PretrainVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 img_size=224, 
                 patch_size=16, 
                 encoder_in_chans=3, 
                 encoder_embed_dim=768, 
                 encoder_depth=12,
                 encoder_num_heads=12, 
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 num_frames=8,
                 tubelet_size=1,
                 use_checkpoint=False,
                 checkpoint_num=0,
                 ckpt_num_frame=4, # the pretrained model uses 4 frames
                 return_index=-1,
                 with_ln=False
                ):
        super().__init__()

        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=encoder_in_chans, 
            embed_dim=encoder_embed_dim, 
            depth=encoder_depth,
            num_heads=encoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            use_learnable_pos_emb=use_learnable_pos_emb,
            use_checkpoint=use_checkpoint,
            checkpoint_num=checkpoint_num,
            ckpt_num_frame=ckpt_num_frame,
            with_ln=with_ln,
            return_index=return_index
        )
        # print('umt:', f'With LN: {with_ln}')
        # print('UMT:', f'Total {encoder_depth} layer')
        # print('UMT:', f'Return {encoder_depth+return_index+1}-th layer')

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'clip_pos_embed'}

    def forward(self, x, use_image=False):
        T = x.shape[2]
        x_vis = self.encoder(x, use_image) # [B, N_vis, C_e]
        B, TL, C = x_vis.shape
        x_vis = x_vis.view(B, T, TL // T, C)

        return x_vis



 



class UMTImageProcessor:
    def __init__(self, image_mean=(0.485, 0.456, 0.406), image_std=(0.229, 0.224, 0.225), size=(224, 224), crop_size: Dict[str, int] = None, resample=PILImageResampling.BICUBIC, rescale_factor=1 / 255, data_format=ChannelDimension.FIRST):
        crop_size = crop_size if crop_size is not None else {"height": 224, "width": 224}
        crop_size = get_size_dict(crop_size, default_to_square=True, param_name="crop_size")

        self.image_mean = image_mean
        self.image_std = image_std
        self.size = size
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.data_format = data_format
        self.crop_size = crop_size

    def preprocess(self, images, return_tensors, target_size=None):
        if isinstance(images, Image.Image):
            images = [images]
        else:
            # to adapt video data
            images = [to_numpy_array(image) for image in images]
            assert isinstance(images, list)

        if target_size is None:
            target_size = self.size
            
        transforms = [
            convert_to_rgb,
            to_numpy_array,
            partial(resize, size=target_size, resample=self.resample, data_format=self.data_format),
            partial(rescale, scale=self.rescale_factor, data_format=self.data_format),
            partial(normalize, mean=self.image_mean, std=self.image_std, data_format=self.data_format),
            partial(to_channel_dimension_format, channel_dim=self.data_format, input_channel_dim=self.data_format),
        ]

        images = reduce(lambda x, f: [*map(f, x)], transforms, images)
        data = {"pixel_values": images}

        return BatchFeature(data=data, tensor_type=return_tensors)


class UMTVisionConfig:
    model_type = "umt_vision_model"

    def __init__(
        self,
        num_frames=4,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_channels=3,
        image_size=224,
        patch_size=16,
        return_idx=-2
        # **kwargs,
    ):
        # super().__init__(**kwargs)
        self.num_frames = num_frames
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.return_idx = return_idx


def build_vit(config, pt_type='origin'):
    model = PretrainVisionTransformer(
        img_size=config.image_size, 
        patch_size=16, 
        encoder_embed_dim=1024, 
        encoder_depth=24,
        encoder_num_heads=16, 
        drop_path_rate=0., 
        num_frames=config.num_frames,
        tubelet_size=1,
        use_checkpoint=False,
        checkpoint_num=24,
        return_index=config.return_idx,
        with_ln=True, # merge vision_layernorm in it
    )
    
    # no need to load pt

    return model



class UMTVisionTower(nn.Module):
    def __init__(self, vision_tower, vision_tower_cfg, delay_load=False, pt_type='origin', image_size=224):
        super().__init__()

        self.is_loaded = False
        self.pt_type = pt_type

        self.config = UMTVisionConfig(num_frames=vision_tower_cfg.mm_local_num_frames, return_idx=vision_tower_cfg.mm_vision_select_layer, image_size=image_size)

        self.vision_tower_name = vision_tower

        self.image_processor = UMTImageProcessor(size=(image_size, image_size))

        if not delay_load:
            print(f"Loading vision tower: {vision_tower}")
            self.load_model()
        elif getattr(vision_tower_cfg, "unfreeze_mm_vision_tower", False):
            # TODO: better detector is needed.
            print(f"The checkpoint seems to contain `vision_tower` weights: `unfreeze_mm_vision_tower`: True.")
            self.load_model()
        elif hasattr(vision_tower_cfg, "mm_tunable_parts") and "mm_vision_tower" in vision_tower_cfg.mm_tunable_parts:
            print(f"The checkpoint seems to contain `vision_tower` weights: `mm_tunable_parts` contains `mm_vision_tower`.")
            self.load_model()
        else:
            self.cfg_only = self.config

    def load_model(self, device_map=None):
        if self.is_loaded:
            print("{} is already loaded, `load_model` called again, skipping.".format(self.vision_tower_name))
            return

        self.vision_tower = build_vit(self.config, pt_type=self.pt_type)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def forward(self, images):
        if type(images) is list:
            raise NotImplementedError
        else:
            # input: B T C H W
            # output: B T*L C
            T = images.shape[1]
            images = images.permute(0, 2, 1, 3, 4)
            image_embeds = self.vision_tower(images, use_image=(T == 1))
            B, T, L, C = image_embeds.shape
            image_embeds = image_embeds.reshape(B, -1, C)

        return image_embeds

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        for p in self.vision_tower.parameters():
            return p.dtype

    @property
    def device(self):
        for p in self.vision_tower.parameters():
            return p.device

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def image_size(self):
        return self.config.image_size


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, "mm_vision_tower", getattr(vision_tower_cfg, "vision_tower", None))


    if "umt-hd" in vision_tower:
        return UMTVisionTower(vision_tower, vision_tower_cfg=vision_tower_cfg, image_size=448, **kwargs)
    elif "umt" in vision_tower:
        raise NotImplementedError
        return UMTVisionTower(vision_tower, vision_tower_cfg=vision_tower_cfg, **kwargs)

    raise ValueError(f"Unknown vision tower: {vision_tower}")