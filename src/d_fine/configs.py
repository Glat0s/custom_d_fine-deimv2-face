from copy import deepcopy

base_cfg = {
    "HGNetv2": {
        "pretrained": False,
        "local_model_dir": "weight/hgnetv2/",
        "freeze_stem_only": True,
        "act": "silu",  # DEIMv2 uses silu
    },
    "DINOv3STAs": {
        "name": "dinov3_vits16",
        "weights_path": "./ckpts/dinov3_vits16.pth",
        "interaction_indexes": [5, 8, 11],
        "finetune": True,
        "conv_inplane": 32,
        "hidden_dim": 224,
    },
    "HybridEncoder": {
        "num_encoder_layers": 1,
        "nhead": 8,
        "dropout": 0.0,
        "enc_act": "gelu",
        "act": "silu",
        "version": "deim",
        "csp_type": "csp2",
        "fuse_op": "sum",
    },
    "LiteEncoder": {
        "act": "silu",
        "csp_type": "csp2",
    },
    "DEIMTransformer": {
        "eval_idx": -1,
        "num_queries": 300,
        "num_denoising": 100,
        "label_noise_ratio": 0.5,
        "box_noise_scale": 1.0,
        "reg_max": 32,
        "layer_scale": 1,
        "cross_attn_method": "default",
        "query_select_method": "default",
        "activation": "silu",
        "mlp_act": "silu",
        "use_gateway": True,
        "share_bbox_head": False,
        "share_score_head": False,
    },
    "DEIMCriterion": {
        "weight_dict": {
            "loss_mal": 1,
            "loss_bbox": 5,
            "loss_giou": 2,
            "loss_fgl": 0.15,
            "loss_ddf": 1.5,
        },
        "losses": ["mal", "boxes", "local"],
        "gamma": 1.5,
        "alpha": 0.75,
        "reg_max": 32,
        "use_uni_set": True,
    },
    "DFINECriterion": {
        "weight_dict": {
            "loss_vfl": 1,
            "loss_bbox": 5,
            "loss_giou": 2,
            "loss_fgl": 0.15,
            "loss_ddf": 1.5,
        },
        "losses": ["vfl", "boxes", "local"],
        "alpha": 0.75,
        "gamma": 2.0,
        "reg_max": 32,
    },
    "matcher": {
        "weight_dict": {"cost_class": 2, "cost_bbox": 5, "cost_giou": 2},
        "alpha": 0.25,
        "gamma": 2.0,
        "use_focal_loss": True,
        "change_matcher": True,
        "iou_order_alpha": 4.0,
        "matcher_change_epoch": 45,
    },
}

sizes_cfg = {
    "atto": {
        "HGNetv2": {"name": "Atto", "return_idx": [2]},
        "LiteEncoder": {"in_channels": [256], "feat_strides": [16], "hidden_dim": 64, "expansion": 0.34, "depth_mult": 0.5},
        "DEIMTransformer": {
            "feat_channels": [64, 64], "feat_strides": [16, 32], "hidden_dim": 64, "num_levels": 2,
            "num_layers": 3, "num_queries": 100, "dim_feedforward": 160, "num_points": [4, 2],
            "share_bbox_head": True, "use_gateway": False,
        },
    },
    "femto": {
        "HGNetv2": {"name": "Femto", "return_idx": [2]},
        "LiteEncoder": {"in_channels": [512], "feat_strides": [16], "hidden_dim": 96, "expansion": 0.34, "depth_mult": 0.5},
        "DEIMTransformer": {
            "feat_channels": [96, 96], "feat_strides": [16, 32], "hidden_dim": 96, "num_levels": 2,
            "num_layers": 3, "num_queries": 150, "dim_feedforward": 256, "num_points": [4, 2],
            "share_bbox_head": True, "use_gateway": False,
        },
    },
    "pico": {
        "HGNetv2": {"name": "Pico", "return_idx": [2]},
        "LiteEncoder": {"in_channels": [512], "feat_strides": [16], "hidden_dim": 112, "expansion": 0.34, "depth_mult": 0.5},
        "DEIMTransformer": {
            "feat_channels": [112, 112], "feat_strides": [16, 32], "hidden_dim": 112, "num_levels": 2,
            "num_layers": 3, "num_queries": 200, "dim_feedforward": 320, "num_points": [4, 2],
            "share_bbox_head": True, "use_gateway": False,
        },
    },
    "n": {
        "HGNetv2": {"name": "B0", "return_idx": [2, 3], "use_lab": True},
        "HybridEncoder": {
            "in_channels": [512, 1024], "feat_strides": [16, 32], "hidden_dim": 128,
            "use_encoder_idx": [1], "dim_feedforward": 512, "expansion": 0.34, "depth_mult": 0.5,
            "version": "dfine", # n-model uses dfine encoder
        },
        "DEIMTransformer": {
            "feat_channels": [128, 128], "feat_strides": [16, 32], "hidden_dim": 128,
            "num_levels": 2, "num_layers": 3, "dim_feedforward": 512, "num_points": [6, 6],
        },
    },
    "s": {
        "DINOv3STAs": { "name": "vit_tiny", "embed_dim": 192, "num_heads": 3, "weights_path": "./ckpts/vitt_distill.pt", "interaction_indexes": [3, 7, 11], "hidden_dim": 192}, # Added hidden_dim for consistency
        "HybridEncoder": { "in_channels": [192, 192, 192], "hidden_dim": 192, "depth_mult": 0.67, "expansion": 0.34, "dim_feedforward": 512},
        "DEIMTransformer": {"feat_channels": [192, 192, 192], "hidden_dim": 192, "num_layers": 4, "dim_feedforward": 512, "num_points": [3, 6, 3],},
    },
    "m": {
        "DINOv3STAs": { "name": "vit_tinyplus", "embed_dim": 256, "num_heads": 4, "weights_path": "./ckpts/vittplus_distill.pt", "interaction_indexes": [3, 7, 11], "hidden_dim": 256}, # Added hidden_dim for consistency
        "HybridEncoder": {"in_channels": [256, 256, 256], "hidden_dim": 256, "depth_mult": 1.0, "expansion": 0.67, "dim_feedforward": 512},
        "DEIMTransformer": {"feat_channels": [256, 256, 256], "hidden_dim": 256, "num_layers": 4, "dim_feedforward": 512, "num_points": [3, 6, 3],},
    },
    "l": {
        "DINOv3STAs": { "name": "dinov3_vits16", "weights_path": "./ckpts/dinov3_vits16.pth", "interaction_indexes": [5, 8, 11], "hidden_dim": 224, "conv_inplane": 32},
        "HybridEncoder": {"in_channels": [224, 224, 224], "hidden_dim": 224, "dim_feedforward": 896},
        "DEIMTransformer": {"feat_channels": [224, 224, 224], "hidden_dim": 224, "num_layers": 4, "dim_feedforward": 1792, "num_points": [3, 6, 3],},
    },
    "x": {
        "DINOv3STAs": { "name": "dinov3_vits16plus", "weights_path": "./ckpts/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth", "interaction_indexes": [5, 8, 11], "hidden_dim": 256, "conv_inplane": 64},
        "HybridEncoder": {"in_channels": [256, 256, 256], "hidden_dim": 256, "dim_feedforward": 1024, "expansion": 1.25, "depth_mult": 1.37},
        "DEIMTransformer": {"feat_channels": [256, 256, 256], "hidden_dim": 256, "num_layers": 6, "dim_feedforward": 2048, "num_points": [3, 6, 3],},
    },
}

def merge_configs(base, size_specific):
    result = {**base}
    for key, value in size_specific.items():
        if key in result and isinstance(result[key], dict):
            # Recursively merge dictionaries
            result[key] = {**result[key], **value}
        else:
            result[key] = value
    return result

models = {size: merge_configs(deepcopy(base_cfg), config) for size, config in sizes_cfg.items()}