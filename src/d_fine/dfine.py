from pathlib import Path
import torch.nn as nn
import torch.optim as optim

from src.d_fine.dfine_criterion import DFINECriterion

from .arch.dfine_decoder import DFINETransformer
from .arch.hgnetv2 import HGNetv2
from .arch.hybrid_encoder import HybridEncoder
from .configs import models
from .matcher import HungarianMatcher
from .utils import load_tuning_state

__all__ = ["DFINE"]


class DFINE(nn.Module):
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


def build_model(model_name, num_classes, device, img_size=None, pretrained_model_path=None):
    model_cfg = models[model_name]
    model_cfg["HybridEncoder"]["eval_spatial_size"] = img_size
    model_cfg["DFINETransformer"]["eval_spatial_size"] = img_size

    backbone = HGNetv2(**model_cfg["HGNetv2"])
    encoder = HybridEncoder(**model_cfg["HybridEncoder"])
    decoder = DFINETransformer(num_classes=num_classes, **model_cfg["DFINETransformer"])

    model = DFINE(backbone, encoder, decoder)

    if pretrained_model_path:
        if not Path(pretrained_model_path).exists():
            raise FileNotFoundError(f"{pretrained_model_path} does not exist")
        model = load_tuning_state(model, str(pretrained_model_path))
    return model.to(device)


def build_loss(model_name, num_classes, label_smoothing):
    model_cfg = models[model_name]
    matcher = HungarianMatcher(**model_cfg["matcher"])
    loss_fn = DFINECriterion(
        matcher,
        num_classes=num_classes,
        label_smoothing=label_smoothing,
        **model_cfg["DFINECriterion"],
    )
    return loss_fn

def build_optimizer(model, optimizer_cfg):
    """
    Builds a simple AdamW optimizer for the D-FINE model.
    """
    return optim.AdamW(
        model.parameters(),
        lr=optimizer_cfg["lr"],
        betas=optimizer_cfg.get("betas", [0.9, 0.999]),
        weight_decay=optimizer_cfg.get("weight_decay", 0.0001),
    )