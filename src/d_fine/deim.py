from pathlib import Path
import re
from copy import deepcopy
import inspect

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from omegaconf import OmegaConf

# Import new DEIM components
from src.d_fine.deim_criterion import DEIMCriterion
from .arch.deim_decoder import DEIMTransformer 
from .arch.hgnetv2 import HGNetv2
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


def build_model(model_name, num_classes, device, img_size=None, pretrained_model_path=None, deim_transformer_cfg=None, hybrid_encoder_cfg=None, lite_encoder_cfg=None):
    model_cfg = deepcopy(models[model_name])
    
    # Merge DEIMv2 specific configs if provided
    if deim_transformer_cfg:
        model_cfg["DEIMTransformer"].update(deim_transformer_cfg)

    # Determine which encoder to use based on config
    if "LiteEncoder" in model_cfg or lite_encoder_cfg is not None:
         if "LiteEncoder" not in model_cfg: model_cfg["LiteEncoder"] = {}
         if lite_encoder_cfg: model_cfg["LiteEncoder"].update(lite_encoder_cfg)
         model_cfg["LiteEncoder"]["eval_spatial_size"] = img_size
         encoder = LiteEncoder(**model_cfg["LiteEncoder"])
    else:
        if hybrid_encoder_cfg:
            model_cfg["HybridEncoder"].update(hybrid_encoder_cfg)
        model_cfg["HybridEncoder"]["eval_spatial_size"] = img_size
        encoder = HybridEncoder(**model_cfg["HybridEncoder"])

    model_cfg["DEIMTransformer"]["eval_spatial_size"] = img_size

    backbone = HGNetv2(**model_cfg["HGNetv2"])

    # Filter decoder kwargs to only include valid arguments for DEIMTransformer
    decoder_args = model_cfg["DEIMTransformer"]
    sig = inspect.signature(DEIMTransformer.__init__)
    valid_kwargs = {k: v for k, v in decoder_args.items() if k in sig.parameters}
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

    # Create parameter groups from the config's 'params' list
    if 'params' in optimizer_cfg:
        for pg_cfg in optimizer_cfg['params']:
            pattern = pg_cfg['params']
            params_in_group = {
                name: param for name, param in model.named_parameters()
                if param.requires_grad and re.search(pattern, name)
            }
            
            # Create a new dict for this group's config, excluding the 'params' key
            group_cfg = {k: v for k, v in pg_cfg.items() if k != 'params'}
            group_cfg['params'] = params_in_group.values()
            param_groups.append(group_cfg)
            
            visited_names.update(params_in_group.keys())

    # Add remaining parameters to a default group
    remaining_params = [
        param for name, param in model.named_parameters()
        if param.requires_grad and name not in visited_names
    ]
    
    if remaining_params:
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
