"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2023 . All Rights Reserved.
"""

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def inverse_sigmoid(x: torch.Tensor, eps: float=1e-5) -> torch.Tensor:
    x = x.clip(min=0., max=1.)
    return torch.log(x.clip(min=eps) / (1 - x).clip(min=eps))


def bias_init_with_prob(prior_prob=0.01):
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-math.log((1 - prior_prob) / prior_prob))
    return bias_init


def deformable_attention_core_func(value, value_spatial_shapes, sampling_locations, attention_weights):
    """
    Args:
        value (Tensor): [bs, value_length, n_head, c]
        value_spatial_shapes (Tensor|List): [n_levels, 2]
        value_level_start_index (Tensor|List): [n_levels]
        sampling_locations (Tensor): [bs, query_length, n_head, n_levels, n_points, 2]
        attention_weights (Tensor): [bs, query_length, n_head, n_levels, n_points]

    Returns:
        output (Tensor): [bs, Length_{query}, C]
    """
    bs, _, n_head, c = value.shape
    _, Len_q, _, n_levels, n_points, _ = sampling_locations.shape

    split_shape = [h * w for h, w in value_spatial_shapes]
    value_list = value.split(split_shape, dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[level].flatten(2).permute(
            0, 2, 1).reshape(bs * n_head, c, h, w)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].permute(
            0, 2, 1, 3, 4).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_*M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(
        bs * n_head, 1, Len_q, n_levels * n_points)
    output = (torch.stack(
        sampling_value_list, dim=-2).flatten(-2) *
              attention_weights).sum(-1).reshape(bs, n_head * c, Len_q)

    return output.permute(0, 2, 1)



def deformable_attention_core_func_v2(\
    value: torch.Tensor,
    value_spatial_shapes,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    num_points_list: List[int],
    method='default',
    value_shape='default',
    ):
    """
    Args:
        value (Tensor): [bs, value_length, n_head, c]
        value_spatial_shapes (Tensor|List): [n_levels, 2]
        value_level_start_index (Tensor|List): [n_levels]
        sampling_locations (Tensor): [bs, query_length, n_head, n_levels * n_points, 2]
        attention_weights (Tensor): [bs, query_length, n_head, n_levels * n_points]

    Returns:
        output (Tensor): [bs, Length_{query}, C]
    """
    if value_shape == 'default':
        bs, n_head, c, _ = value[0].shape
    elif value_shape == 'reshape':   # reshape following RT-DETR
        bs, _, n_head, c = value.shape
        split_shape = [h * w for h, w in value_spatial_shapes]
        value = value.permute(0, 2, 3, 1).flatten(0, 1).split(split_shape, dim=-1)
    _, Len_q, _, _, _ = sampling_locations.shape

    # sampling_offsets [8, 480, 8, 12, 2]
    if method == 'default':
        sampling_grids = 2 * sampling_locations - 1

    elif method == 'discrete':
        sampling_grids = sampling_locations

    sampling_grids = sampling_grids.permute(0, 2, 1, 3, 4).flatten(0, 1)
    sampling_locations_list = sampling_grids.split(num_points_list, dim=-2)

    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        value_l = value[level].reshape(bs * n_head, c, h, w)
        sampling_grid_l: torch.Tensor = sampling_locations_list[level]

        if method == 'default':
            sampling_value_l = F.grid_sample(
                value_l,
                sampling_grid_l,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False)

        elif method == 'discrete':
            # n * m, seq, n, 2
            sampling_coord = (sampling_grid_l * torch.tensor([[w, h]], device=value_l.device) + 0.5).to(torch.int64)

            # FIX ME? for rectangle input
            sampling_coord = sampling_coord.clamp(0, h - 1)
            sampling_coord = sampling_coord.reshape(bs * n_head, Len_q * num_points_list[level], 2)

            s_idx = torch.arange(sampling_coord.shape[0], device=value_l.device).unsqueeze(-1).repeat(1, sampling_coord.shape[1])
            sampling_value_l: torch.Tensor = value_l[s_idx, :, sampling_coord[..., 1], sampling_coord[..., 0]] # n l c

            sampling_value_l = sampling_value_l.permute(0, 2, 1).reshape(bs * n_head, c, Len_q, num_points_list[level])

        sampling_value_list.append(sampling_value_l)

    attn_weights = attention_weights.permute(0, 2, 1, 3).reshape(bs * n_head, 1, Len_q, sum(num_points_list))
    weighted_sample_locs = torch.concat(sampling_value_list, dim=-1) * attn_weights
    output = weighted_sample_locs.sum(-1).reshape(bs, n_head * c, Len_q)

    return output.permute(0, 2, 1)


def get_activation(act: str, inpace: bool=True):
    """get activation
    """
    if act is None:
        return nn.Identity()

    elif isinstance(act, nn.Module):
        return act

    act = act.lower()

    if act == 'silu' or act == 'swish':
        m = nn.SiLU()

    elif act == 'relu':
        m = nn.ReLU()

    elif act == 'leaky_relu':
        m = nn.LeakyReLU()

    elif act == 'silu':
        m = nn.SiLU()

    elif act == 'gelu':
        m = nn.GELU()

    elif act == 'hardsigmoid':
        m = nn.Hardsigmoid()

    else:
        raise RuntimeError('')

    if hasattr(m, 'inplace'):
        m.inplace = inpace

    return m


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel)


def de_parallel(model) -> nn.Module:
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model

# Helper functions for load_tuning_state
def _matched_state(state: dict, params: dict):
    missed_list = []
    unmatched_list = []
    matched_state = {}
    for k, v in state.items():
        if k in params:
            if v.shape == params[k].shape:
                matched_state[k] = params[k]
            else:
                unmatched_list.append(k)
        else:
            missed_list.append(k)
    return matched_state, {'missed': missed_list, 'unmatched': unmatched_list}


def _map_class_weights(cur_tensor, pretrain_tensor, obj365_ids):
    if pretrain_tensor.size() == cur_tensor.size():
        return pretrain_tensor
    
    adjusted_tensor = cur_tensor.clone()
    adjusted_tensor.requires_grad = False
    
    if pretrain_tensor.size() > cur_tensor.size():
        # Map from obj365 (larger) to coco (smaller)
        for coco_id, obj_id in enumerate(obj365_ids):
            if coco_id < len(adjusted_tensor) and (obj_id + 1) < len(pretrain_tensor):
                adjusted_tensor[coco_id] = pretrain_tensor[obj_id + 1]
    else:
        # Map from coco (smaller) to obj365 (larger)
        for coco_id, obj_id in enumerate(obj365_ids):
            if (obj_id + 1) < len(adjusted_tensor) and coco_id < len(pretrain_tensor):
                adjusted_tensor[obj_id + 1] = pretrain_tensor[coco_id]
    
    return adjusted_tensor


def _adjust_head_parameters(cur_state_dict, pretrain_state_dict, obj365_ids):
    if 'decoder.denoising_class_embed.weight' in pretrain_state_dict and \
       'decoder.denoising_class_embed.weight' in cur_state_dict and \
       pretrain_state_dict['decoder.denoising_class_embed.weight'].size() != cur_state_dict['decoder.denoising_class_embed.weight'].size():
        del pretrain_state_dict['decoder.denoising_class_embed.weight']

    head_param_names = [
        'decoder.enc_score_head.weight',
        'decoder.enc_score_head.bias'
    ]
    # Assuming max 8 decoder layers from original repo
    for i in range(8):
        head_param_names.append(f'decoder.dec_score_head.{i}.weight')
        head_param_names.append(f'decoder.dec_score_head.{i}.bias')

    for param_name in head_param_names:
        if param_name in cur_state_dict and param_name in pretrain_state_dict:
            cur_tensor = cur_state_dict[param_name]
            pretrain_tensor = pretrain_state_dict[param_name]
            adjusted_tensor = _map_class_weights(cur_tensor, pretrain_tensor, obj365_ids)
            if adjusted_tensor is not None:
                pretrain_state_dict[param_name] = adjusted_tensor
    
    return pretrain_state_dict

# This is the obj365_ids list from the reference solver
OBJ365_IDS = [
    0, 46, 5, 58, 114, 55, 116, 65, 21, 40, 176, 127, 249, 24, 56, 139, 92, 78, 99, 96,
    144, 295, 178, 180, 38, 39, 13, 43, 120, 219, 148, 173, 165, 154, 137, 113, 145, 146,
    204, 8, 35, 10, 88, 84, 93, 26, 112, 82, 265, 104, 141, 152, 234, 143, 150, 97, 2,
    50, 25, 75, 98, 153, 37, 73, 115, 132, 106, 61, 163, 134, 277, 81, 133, 18, 94, 30,
    169, 70, 328, 226
]


def load_tuning_state(model, path: str):
    """Load model for tuning and adjust mismatched head parameters"""
    if path.startswith('http'):
        state = torch.hub.load_state_dict_from_url(path, map_location='cpu')
    else:
        state = torch.load(path, map_location='cpu')

    module = de_parallel(model)

    # Load the appropriate state dict
    if 'ema' in state:
        pretrain_state_dict = state['ema']['module']
    elif 'model' in state:
        pretrain_state_dict = state['model']
    else:
        pretrain_state_dict = state

    # Adjust head parameters between datasets
    try:
        # Pass obj365_ids for remapping
        adjusted_state_dict = _adjust_head_parameters(module.state_dict(), pretrain_state_dict, OBJ365_IDS)
        stat, infos = _matched_state(module.state_dict(), adjusted_state_dict)
    except Exception:
        stat, infos = _matched_state(module.state_dict(), pretrain_state_dict)

    module.load_state_dict(stat, strict=False)
    print(f'Load model.state_dict, {infos}')
    return model