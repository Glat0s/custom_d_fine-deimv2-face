# src/d_fine/deim.py

from pathlib import Path
import re
from copy import deepcopy
import inspect

import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf

# Import new DEIM components
from src.d_fine.deim_criterion import DEIMCriterion
from .arch.deim_decoder import DEIMTransformer
from .arch.hgnetv2 import HGNetv2
from .arch.dinov3_adapter import DINOv3STAs
from .arch.hybrid_encoder import HybridEncoder
from .arch.lite_encoder import LiteEncoder
from .configs import models
from .matcher import HungarianMatcher
from .utils import load_tuning_state

__all__ = ["DEIM"]


class DEIM(nn.Module):
    __inject__ = [
        "backbone",
        "encoder",
        "decoder",
    ]

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder

    def forward(self, x, targets=None):
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x, targets)
        return x

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self


def build_model(model_name, num_classes, device, img_size=None, pretrained_model_path=None, deim_transformer_cfg=None, hybrid_encoder_cfg=None, lite_encoder_cfg=None, dinov3_stas_cfg=None):
    model_cfg = deepcopy(models[model_name])
    
    # Merge overrides from the main config.yaml into the model-specific config
    if deim_transformer_cfg:
        model_cfg["DEIMTransformer"].update(deim_transformer_cfg)
    if hybrid_encoder_cfg:
        model_cfg["HybridEncoder"].update(hybrid_encoder_cfg)
    if lite_encoder_cfg:
        model_cfg["LiteEncoder"].update(lite_encoder_cfg)
    if dinov3_stas_cfg:
        model_cfg["DINOv3STAs"].update(dinov3_stas_cfg)


    if "DINOv3STAs" in model_cfg and model_name in ['s', 'm', 'l', 'x']:
        print("Building DEIMv2 with DINOv3/ViT backbone.")

        backbone_cfg = model_cfg["DINOv3STAs"]

        backbone = DINOv3STAs(**backbone_cfg)
        
        # DINOv3STAs uses hidden_dim for all output feature maps
        encoder_in_channels = [backbone_cfg["hidden_dim"]] * 3
    else:
        print("Building DEIMv2 with HGNetv2 backbone.")
        backbone = HGNetv2(**model_cfg["HGNetv2"])
        # HGNetv2 has a list of output channels for its return_idx
        encoder_in_channels = [backbone._out_channels[i] for i in backbone.return_idx]
        model_cfg["HybridEncoder"]["in_channels"] = encoder_in_channels

    if "LiteEncoder" in model_cfg and model_name in ['atto', 'femto', 'pico']:
        print("Building with LiteEncoder.")
        model_cfg["LiteEncoder"]["in_channels"] = [backbone._out_channels[i] for i in backbone.return_idx]
        encoder_cfg = model_cfg["LiteEncoder"]
        encoder_cfg["eval_spatial_size"] = img_size
        encoder = LiteEncoder(**encoder_cfg)
    elif "HybridEncoder" in model_cfg:
        print("Building with HybridEncoder.")
        model_cfg["HybridEncoder"]["in_channels"] = encoder_in_channels
        encoder_cfg = model_cfg["HybridEncoder"]
        encoder_cfg["eval_spatial_size"] = img_size
        encoder = HybridEncoder(**encoder_cfg)
    else:
        raise ValueError(f"No valid encoder configuration found for model size '{model_name}'.")

    # Decoder configuration
    decoder_cfg = model_cfg["DEIMTransformer"]
    decoder_cfg["eval_spatial_size"] = img_size

    # Filter decoder kwargs to only include valid arguments for DEIMTransformer
    sig = inspect.signature(DEIMTransformer.__init__)
    valid_kwargs = {k: v for k, v in decoder_cfg.items() if k in sig.parameters}
    decoder = DEIMTransformer(num_classes=num_classes, **valid_kwargs) 

    model = DEIM(backbone, encoder, decoder)

    if pretrained_model_path:
        if not Path(pretrained_model_path).exists():
            raise FileNotFoundError(f"{pretrained_model_path} does not exist")
        model = load_tuning_state(model, str(pretrained_model_path))

    return model.to(device)

def build_loss(model_name, num_classes, deim_criterion_cfg):
    model_cfg = models[model_name]
    # Correctly merge matcher configs
    matcher_config = {**model_cfg["matcher"], **deim_criterion_cfg['matcher']}

    # Ensure use_focal_loss is set correctly for the matcher
    matcher_config['use_focal_loss'] = True 
    matcher = HungarianMatcher(**matcher_config)

    # Convert DictConfig to a regular dict to allow deletion
    criterion_params = OmegaConf.to_container(deim_criterion_cfg, resolve=True)
    if 'matcher' in criterion_params:
        del criterion_params['matcher']
    
    # Use DEIMCriterion
    loss_fn = DEIMCriterion(
        matcher=matcher,
        num_classes=num_classes,
        **criterion_params,
    )
    return loss_fn


def build_optimizer(model, optimizer_cfg):
    """
    Builds an optimizer with flexible parameter grouping based on regex patterns
    from the configuration file, matching the DEIMv2 methodology.
    """
    param_groups = []
    visited_names = set()
    all_params = dict(model.named_parameters())

    # Create parameter groups from the config's 'params' list
    if 'params' in optimizer_cfg:
        for pg_cfg in optimizer_cfg['params']:
            pattern = pg_cfg['params']
            
            # Find parameters that match the pattern AND have not been visited yet
            params_in_group = {
                name: param for name, param in all_params.items()
                if param.requires_grad and re.search(pattern, name) and name not in visited_names
            }
            
            # Create a new dict for this group's config, excluding the 'params' key
            group_cfg = {k: v for k, v in pg_cfg.items() if k != 'params'}
            group_cfg['params'] = list(params_in_group.values()) # Use list() for safety
            
            if group_cfg['params']:
                param_groups.append(group_cfg)
                visited_names.update(params_in_group.keys())

    # Add remaining parameters to a default group
    remaining_params = [
        param for name, param in all_params.items()
        if param.requires_grad and name not in visited_names
    ]
    
    if remaining_params:
        # This group will use the default optimizer settings (lr, weight_decay)
        param_groups.append({'params': remaining_params})

    # Create optimizer with the gathered parameter groups
    optimizer_type = optimizer_cfg.get('type', 'AdamW')
    if optimizer_type == 'AdamW':
        return optim.AdamW(
            param_groups,
            lr=optimizer_cfg['lr'],
            betas=optimizer_cfg.get('betas', [0.9, 0.999]),
            weight_decay=optimizer_cfg.get('weight_decay', 0.0001)
        )
    else:
        raise NotImplementedError(f"Optimizer {optimizer_type} not implemented")

