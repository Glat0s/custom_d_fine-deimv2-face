"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
"""

import math
import copy
import functools
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import List

from src.d_fine.denoising import get_contrastive_denoising_training_group
from src.d_fine.utils import deformable_attention_core_func_v2, get_activation, inverse_sigmoid, bias_init_with_prob

from src.d_fine.arch.dfine_decoder import MSDeformableAttention, LQE, Integral
from src.d_fine.arch.utils import weighting_function, distance2bbox
from src.d_fine.deim_utils import RMSNorm, SwiGLUFFN, Gate, MLP


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation='relu',
                 n_levels=4,
                 n_points=4,
                 cross_attn_method='default',
                 layer_scale=None,
                 use_gateway=False,
                 ):
        super(TransformerDecoderLayer, self).__init__()

        if layer_scale is not None:
            dim_feedforward = round(layer_scale * dim_feedforward)
            d_model = round(layer_scale * d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = RMSNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points, method=cross_attn_method)
        self.dropout2 = nn.Dropout(dropout)

        self.use_gateway = use_gateway
        if use_gateway:
            self.gateway = Gate(d_model, use_rmsnorm=True)
        else:
            self.norm2 = RMSNorm(d_model)

        # ffn
        self.swish_ffn = SwiGLUFFN(d_model, dim_feedforward // 2, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = RMSNorm(d_model)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self,
                target,
                reference_points,
                value,
                spatial_shapes,
                attn_mask=None,
                query_pos_embed=None):

        # self attention
        q = k = self.with_pos_embed(target, query_pos_embed)

        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
        target = target + self.dropout1(target2)
        target = self.norm1(target)

        # cross attention
        target2 = self.cross_attn(\
            self.with_pos_embed(target, query_pos_embed),
            reference_points,
            value,
            spatial_shapes)

        if self.use_gateway:
            target = self.gateway(target, self.dropout2(target2))
        else:
            target = target + self.dropout2(target2)
            target = self.norm2(target)

        # ffn
        target2 = self.swish_ffn(target)
        target = target + self.dropout4(target2)
        target = self.norm3(target.clamp(min=-65504, max=65504))

        return target


class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, decoder_layer, decoder_layer_wide, num_layers, num_head, reg_max, reg_scale, up,
                 eval_idx=-1, layer_scale=2, act='relu'):
        super(TransformerDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer_scale = layer_scale
        self.num_head = num_head
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.up, self.reg_scale, self.reg_max = up, reg_scale, reg_max
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(self.eval_idx + 1)] \
                    + [copy.deepcopy(decoder_layer_wide) for _ in range(num_layers - self.eval_idx - 1)])
        self.lqe_layers = nn.ModuleList([copy.deepcopy(LQE(4, 64, 2, reg_max, act=act)) for _ in range(num_layers)])

    def value_op(self, memory, value_proj, value_scale, memory_mask, memory_spatial_shapes):
        value = value_proj(memory) if value_proj is not None else memory
        value = F.interpolate(memory, size=value_scale) if value_scale is not None else value
        if memory_mask is not None:
            value = value * memory_mask.to(value.dtype).unsqueeze(-1)
        value = value.reshape(value.shape[0], value.shape[1], self.num_head, -1)
        split_shape = [h * w for h, w in memory_spatial_shapes]
        return value.permute(0, 2, 3, 1).split(split_shape, dim=-1)

    def convert_to_deploy(self):
        self.project = weighting_function(self.reg_max, self.up, self.reg_scale, deploy=True)
        self.layers = self.layers[:self.eval_idx + 1]
        self.lqe_layers = nn.ModuleList([nn.Identity()] * (self.eval_idx) + [self.lqe_layers[self.eval_idx]])

    def forward(self,
                target,
                ref_points_unact,
                memory,
                spatial_shapes,
                bbox_head,
                score_head,
                query_pos_head,
                pre_bbox_head,
                integral,
                up,
                reg_scale,
                attn_mask=None,
                memory_mask=None,
                dn_meta=None):
        output = target
        output_detach = pred_corners_undetach = 0
        value = self.value_op(memory, None, None, memory_mask, spatial_shapes)

        dec_out_bboxes, dec_out_logits, dec_out_pred_corners, dec_out_refs = [], [], [], []
        project = weighting_function(self.reg_max, up, reg_scale) if not hasattr(self, 'project') else self.project

        ref_points_detach = F.sigmoid(ref_points_unact)
        query_pos_embed = query_pos_head(ref_points_detach).clamp(min=-10, max=10)

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)

            if i >= self.eval_idx + 1 and self.layer_scale > 1:
                query_pos_embed = F.interpolate(query_pos_embed, scale_factor=self.layer_scale)
                value = self.value_op(memory, None, query_pos_embed.shape[-1], memory_mask, spatial_shapes)
                output = F.interpolate(output, size=query_pos_embed.shape[-1])
                output_detach = output.detach()

            output = layer(output, ref_points_input, value, spatial_shapes, attn_mask, query_pos_embed)

            if i == 0:
                pre_bboxes = F.sigmoid(pre_bbox_head(output) + inverse_sigmoid(ref_points_detach))
                pre_scores = score_head[0](output)
                ref_points_initial = pre_bboxes.detach()

            pred_corners = bbox_head[i](output + output_detach) + pred_corners_undetach
            inter_ref_bbox = distance2bbox(ref_points_initial, integral(pred_corners, project), reg_scale)

            if self.training or i == self.eval_idx:
                scores = self.lqe_layers[i](score_head[i](output), pred_corners)
                dec_out_logits.append(scores)
                dec_out_bboxes.append(inter_ref_bbox)
                dec_out_pred_corners.append(pred_corners)
                dec_out_refs.append(ref_points_initial)
                if not self.training: break

            pred_corners_undetach = pred_corners
            ref_points_detach = inter_ref_bbox.detach()
            output_detach = output.detach()

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits), \
               torch.stack(dec_out_pred_corners), torch.stack(dec_out_refs), pre_bboxes, pre_scores


class DEIMTransformer(nn.Module):
    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,
                 num_points=4,
                 nhead=8,
                 num_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learn_query_content=False,
                 eval_spatial_size=None,
                 eval_idx=-1,
                 eps=1e-2,
                 aux_loss=True,
                 cross_attn_method='default',
                 query_select_method='default',
                 reg_max=32,
                 reg_scale=4.,
                 layer_scale=1,
                 mlp_act='relu',
                 use_gateway=True,
                 share_bbox_head=False,
                 share_score_head=False,
                 ):
        super().__init__()
        assert len(feat_channels) <= num_levels and len(feat_strides) == len(feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim, self.nhead, self.feat_strides, self.num_levels = hidden_dim, nhead, feat_strides, num_levels
        self.num_classes, self.num_queries, self.eps, self.num_layers = num_classes, num_queries, eps, num_layers
        self.eval_spatial_size, self.aux_loss, self.reg_max = eval_spatial_size, aux_loss, reg_max
        self.cross_attn_method, self.query_select_method = cross_attn_method, query_select_method
        
        self._build_input_proj_layer(feat_channels)
        
        scaled_dim = round(layer_scale * hidden_dim)
        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)
        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels, num_points, cross_attn_method, use_gateway=use_gateway)
        decoder_layer_wide = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels, num_points, cross_attn_method, layer_scale, use_gateway=use_gateway)
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, decoder_layer_wide, num_layers, nhead, reg_max, self.reg_scale, self.up, eval_idx, layer_scale, act=activation)
        
        self.num_denoising, self.label_noise_ratio, self.box_noise_scale = num_denoising, label_noise_ratio, box_noise_scale
        if num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(num_classes + 1, hidden_dim, padding_idx=num_classes)
            init.normal_(self.denoising_class_embed.weight[:-1])
            
        self.learn_query_content = learn_query_content
        if learn_query_content:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)

        self.enc_score_head = nn.Linear(hidden_dim, 1 if query_select_method == 'agnostic' else num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, act=mlp_act)
        self.query_pos_head = MLP(4, hidden_dim, hidden_dim, 3, act=mlp_act)
        self.pre_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, act=mlp_act)
        self.integral = Integral(self.reg_max)

        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        dec_score_head = nn.Linear(hidden_dim, num_classes)
        self.dec_score_head = nn.ModuleList([dec_score_head if share_score_head else copy.deepcopy(dec_score_head) for _ in range(self.eval_idx + 1)] + [copy.deepcopy(dec_score_head) for _ in range(num_layers - self.eval_idx - 1)])
        
        dec_bbox_head = MLP(hidden_dim, hidden_dim, 4 * (self.reg_max+1), 3, act=mlp_act)
        self.dec_bbox_head = nn.ModuleList([dec_bbox_head if share_bbox_head else copy.deepcopy(dec_bbox_head) for _ in range(self.eval_idx + 1)] + [MLP(scaled_dim, scaled_dim, 4 * (self.reg_max+1), 3, act=mlp_act) for _ in range(num_layers - self.eval_idx - 1)])

        if self.eval_spatial_size:
            self.anchors, self.valid_mask = self._generate_anchors()

        self._reset_parameters(feat_channels)

    def convert_to_deploy(self):
        self.dec_score_head = nn.ModuleList([nn.Identity()] * self.eval_idx + [self.dec_score_head[self.eval_idx]])
        self.dec_bbox_head = nn.ModuleList([self.dec_bbox_head[i] if i <= self.eval_idx else nn.Identity() for i in range(len(self.dec_bbox_head))])

    def _reset_parameters(self, feat_channels):
        bias = bias_init_with_prob(0.01)
        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)
        init.constant_(self.pre_bbox_head.layers[-1].weight, 0)
        init.constant_(self.pre_bbox_head.layers[-1].bias, 0)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(cls_.bias, bias)
            if hasattr(reg_, 'layers'):
                init.constant_(reg_.layers[-1].weight, 0)
                init.constant_(reg_.layers[-1].bias, 0)

        if self.learn_query_content: init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        init.xavier_uniform_(self.query_pos_head.layers[-1].weight)
        for m, in_channels in zip(self.input_proj, feat_channels):
            if in_channels != self.hidden_dim:
                init.xavier_uniform_(m[0].weight)

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            self.input_proj.append(nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)),
                ('norm', nn.BatchNorm2d(self.hidden_dim))])) if in_channels != self.hidden_dim else nn.Identity())
        
        for _ in range(self.num_levels - len(feat_channels)):
            self.input_proj.append(nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, 2, padding=1, bias=False)),
                ('norm', nn.BatchNorm2d(self.hidden_dim))])))

    def _get_encoder_input(self, feats: List[torch.Tensor]):
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            for i in range(len(proj_feats), self.num_levels):
                proj_feats.append(self.input_proj[i](proj_feats[-1]))
        
        feat_flatten = [feat.flatten(2).permute(0, 2, 1) for feat in proj_feats]
        spatial_shapes = [[feat.shape[2], feat.shape[3]] for feat in proj_feats]
        return torch.concat(feat_flatten, 1), spatial_shapes

    def _generate_anchors(self, spatial_shapes=None, grid_size=0.05, dtype=torch.float32, device='cpu'):
        if spatial_shapes is None:
            spatial_shapes = [[int(self.eval_spatial_size[0] / s), int(self.eval_spatial_size[1] / s)] for s in self.feat_strides]
        
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], -1)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor([w, h], dtype=dtype, device=device)
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
            anchors.append(torch.cat([grid_xy, wh], -1).reshape(-1, h * w, 4))
        
        anchors = torch.cat(anchors, 1).to(device)
        valid_mask = ((anchors > self.eps) & (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        return torch.where(valid_mask, anchors, torch.inf), valid_mask

    def _get_decoder_input(self, memory, spatial_shapes, denoising_logits=None, denoising_bbox_unact=None):
        anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device) if self.training or not hasattr(self, 'anchors') else (self.anchors, self.valid_mask)
        memory = valid_mask.to(memory.dtype) * memory
        enc_outputs_logits = self.enc_score_head(memory)
        
        topk_memory, topk_logits, topk_anchors = self._select_topk(memory, enc_outputs_logits, anchors, self.num_queries)
        enc_topk_bbox_unact = self.enc_bbox_head(topk_memory) + topk_anchors
        
        enc_topk_bboxes_list, enc_topk_logits_list = [F.sigmoid(enc_topk_bbox_unact)], [topk_logits] if self.training else [], []
        
        content = self.tgt_embed.weight.unsqueeze(0).expand(memory.shape[0], -1, -1) if self.learn_query_content else topk_memory.detach()
        
        if denoising_bbox_unact is not None:
            enc_topk_bbox_unact = torch.cat([denoising_bbox_unact, enc_topk_bbox_unact.detach()], 1)
            content = torch.cat([denoising_logits, content], 1)
        
        return content, enc_topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_list

    def _select_topk(self, memory, logits, anchors, topk):
        _, topk_ind = torch.topk(logits.max(-1).values, topk, -1)
        
        topk_anchors = anchors.gather(1, topk_ind.unsqueeze(-1).repeat(1, 1, anchors.shape[-1]))
        topk_logits = logits.gather(1, topk_ind.unsqueeze(-1).repeat(1, 1, logits.shape[-1])) if self.training else None
        topk_memory = memory.gather(1, topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1]))
        return topk_memory, topk_logits, topk_anchors

    def forward(self, feats, targets=None):
        memory, spatial_shapes = self._get_encoder_input(feats)
        dn_meta = None
        if self.training and self.num_denoising > 0:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = get_contrastive_denoising_training_group(targets, self.num_classes, self.num_queries, self.denoising_class_embed, num_denoising=self.num_denoising)
        else:
            denoising_logits, denoising_bbox_unact, attn_mask = None, None, None

        content, ref_points_unact, enc_bboxes, enc_logits = self._get_decoder_input(memory, spatial_shapes, denoising_logits, denoising_bbox_unact)

        out_bboxes, out_logits, out_corners, out_refs, pre_bboxes, pre_logits = self.decoder(
            content, ref_points_unact, memory, spatial_shapes, self.dec_bbox_head, self.dec_score_head,
            self.query_pos_head, self.pre_bbox_head, self.integral, self.up, self.reg_scale, attn_mask=attn_mask, dn_meta=dn_meta
        )

        out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1]}
        if self.training:
            out.update({'pred_corners': out_corners[-1], 'ref_points': out_refs[-1], 'up': self.up, 'reg_scale': self.reg_scale})
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1], out_corners[:-1], out_refs[:-1], out_corners[-1], out_logits[-1])
                out['enc_aux_outputs'] = self._set_aux_loss(enc_logits, enc_bboxes)
                out['pre_outputs'] = {'pred_logits': pre_logits, 'pred_boxes': pre_bboxes}
                out['enc_meta'] = {'class_agnostic': self.query_select_method == 'agnostic'}

            if dn_meta:
                dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
                dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)
                dn_out_corners, out_corners = torch.split(out_corners, dn_meta['dn_num_split'], dim=2)
                dn_out_refs, _ = torch.split(out_refs, dn_meta['dn_num_split'], dim=2)
                dn_pre_logits, _ = torch.split(pre_logits, dn_meta['dn_num_split'], dim=1)
                dn_pre_bboxes, _ = torch.split(pre_bboxes, dn_meta['dn_num_split'], dim=1)

                out['dn_outputs'] = self._set_aux_loss(dn_out_logits, dn_out_bboxes, dn_out_corners, dn_out_refs, dn_out_corners[-1], dn_out_logits[-1])
                out['dn_pre_outputs'] = {'pred_logits': dn_pre_logits, 'pred_boxes': dn_pre_bboxes}
                out['dn_meta'] = dn_meta
        
        return out

    @torch.jit.unused
    def _set_aux_loss(self, logits, boxes, corners=None, refs=None, teacher_corners=None, teacher_logits=None):
        if corners is None:
            return [{'pred_logits': l, 'pred_boxes': b} for l, b in zip(logits, boxes)]
        return [{'pred_logits': l, 'pred_boxes': b, 'pred_corners': c, 'ref_points': r, 'teacher_corners': teacher_corners, 'teacher_logits': teacher_logits} 
                for l, b, c, r in zip(logits, boxes, corners, refs)]
