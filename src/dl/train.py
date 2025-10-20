import math
import time
from copy import deepcopy
from pathlib import Path
from shutil import rmtree
from typing import Dict, List, Tuple

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dl.dataset import Loader
from src.dl.lr_scheduler import FlatCosineLRScheduler
from src.dl.utils import get_model_builder, calculate_remaining_time, filter_preds, get_vram_usage, log_metrics_locally, process_boxes, save_metrics, set_seeds, visualize, wandb_logger
from src.dl.validator import Validator

class ModelEMA:
    def __init__(self, student, ema_momentum):
        self.model = deepcopy(student).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.ema_momentum = ema_momentum
        self.ema_scheduler = lambda x: self.ema_momentum * (1 - math.exp(-x / 2000))

    def update(self, iters, student):
        student = student.state_dict()
        with torch.no_grad():
            momentum = self.ema_scheduler(iters)
            for name, param in self.model.state_dict().items():
                if param.dtype.is_floating_point:
                    param.mul_(momentum).add_((1.0 - momentum) * student[name].detach())


class Trainer:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.device = cfg.train.device
        self.conf_thresh = cfg.train.conf_thresh
        self.iou_thresh = cfg.train.iou_thresh
        self.epochs = cfg.train.epochs
        self.path_to_save = Path(cfg.train.path_to_save)
        self.to_visualize_eval = cfg.train.to_visualize_eval
        self.amp_enabled = cfg.train.amp_enabled
        self.clip_max_norm = cfg.train.clip_max_norm
        self.b_accum_steps = max(cfg.train.b_accum_steps, 1)
        self.keep_ratio = cfg.train.keep_ratio
        self.early_stopping = cfg.train.early_stopping
        self.use_wandb = cfg.train.use_wandb
        self.label_to_name = cfg.train.label_to_name
        self.num_labels = len(cfg.train.label_to_name)

        self.debug_img_path = Path(self.cfg.train.debug_img_path)
        self.eval_preds_path = Path(self.cfg.train.eval_preds_path)
        self.init_dirs()

        if self.use_wandb:
            wandb.init(project=cfg.project_name, name=cfg.exp, config=OmegaConf.to_container(cfg, resolve=True))
        
        log_file = self.path_to_save / "train_log.txt"
        log_file.unlink(missing_ok=True)
        logger.add(log_file, format="{message}", level="INFO")

        set_seeds(cfg.train.seed, cfg.train.cudnn_fixed)
        
        if 'DEIMCriterion' in cfg.train:
            from src.d_fine.deim import build_loss, build_optimizer
        else:
            # Import the correct optimizer builder for the D-FINE model path
            from src.d_fine.dfine import build_loss, build_optimizer
        
        build_model = get_model_builder(cfg)

        self.stop_epoch = self.epochs - self.cfg.train.lr_scheduler.no_aug_epochs

        self.base_loader = Loader(
            root_path=Path(cfg.train.data_path), img_size=tuple(cfg.train.img_size),
            batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers,
            cfg=cfg, debug_img_processing=cfg.train.debug_img_processing,
            base_size_repeat=cfg.train.dataloader.get("base_size_repeat"),
            stop_epoch_multiscale=self.stop_epoch,
        )
        self.train_loader, self.val_loader, self.test_loader = self.base_loader.build_dataloaders()
        
        if 'DEIMCriterion' in cfg.train:
            self.model = build_model(
                cfg.model_name, self.num_labels, self.device,
                img_size=cfg.train.img_size,
                pretrained_model_path=cfg.train.pretrained_model_path,
                deim_transformer_cfg=cfg.train.get('DEIMTransformer'),
                hybrid_encoder_cfg=cfg.train.get('HybridEncoder'),
                lite_encoder_cfg=cfg.train.get('LiteEncoder'),
                dinov3_stas_cfg=cfg.train.get('DINOv3STAs')  # <-- FIX
            )
        else:
            self.model = build_model(
                cfg.model_name, self.num_labels, self.device,
                img_size=cfg.train.img_size,
                pretrained_model_path=cfg.train.pretrained_model_path
            )

        self.ema_model = ModelEMA(self.model, cfg.train.ema_momentum) if cfg.train.use_ema else None
        
        self.loss_fn = build_loss(
            cfg.model_name, self.num_labels, deim_criterion_cfg=cfg.train.get('DEIMCriterion')
        ) if 'DEIMCriterion' in cfg.train else build_loss(
            cfg.model_name, self.num_labels, cfg.train.label_smoothing
        )

        self.optimizer = build_optimizer(self.model, cfg.train.optimizer)

        # Store initial LRs for all schedulers
        for pg in self.optimizer.param_groups:
            pg['initial_lr'] = pg['lr']

        # Scheduler Selection
        self.self_lr_scheduler = False
        scheduler_cfg = cfg.train.lr_scheduler
        if scheduler_cfg.type == 'flatcosine':
            logger.info("Using FlatCosineLRScheduler")
            iter_per_epoch = len(self.train_loader) // self.b_accum_steps
            self.lr_scheduler = FlatCosineLRScheduler(
                self.optimizer,
                lr_gamma=scheduler_cfg.lr_gamma,
                iter_per_epoch=iter_per_epoch,
                total_epochs=cfg.train.epochs,
                warmup_iter=scheduler_cfg.warmup_iter,
                flat_epochs=scheduler_cfg.flat_epochs,
                no_aug_epochs=scheduler_cfg.no_aug_epochs
            )
            self.self_lr_scheduler = True
        elif scheduler_cfg.type == 'onecycle':
            logger.info("Using OneCycleLR Scheduler")
            max_lr_groups = [pg['initial_lr'] * 2 for pg in self.optimizer.param_groups]

            self.scheduler = OneCycleLR(
                self.optimizer, max_lr=max_lr_groups, epochs=cfg.train.epochs,
                steps_per_epoch=len(self.train_loader) // self.b_accum_steps,
                pct_start=scheduler_cfg.cycler_pct_start, cycle_momentum=False
            )
        else:
            self.scheduler = None

        self.scaler = GradScaler() if self.amp_enabled else None
        if self.use_wandb: wandb.watch(self.model)

    def init_dirs(self):
        for path in [self.debug_img_path, self.eval_preds_path]:
            if path.exists():
                rmtree(path)
            path.mkdir(exist_ok=True, parents=True)

        self.path_to_save.mkdir(exist_ok=True, parents=True)
        with open(self.path_to_save / "config.yaml", "w") as f:
            OmegaConf.save(config=self.cfg, f=f)

    def preds_postprocess(self, inputs, outputs, orig_sizes, num_top_queries=300, use_focal_loss=True) -> List[Dict[str, torch.Tensor]]:
        logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
        boxes = process_boxes(
            boxes, inputs.shape[2:], orig_sizes, self.keep_ratio, inputs.device
        )  # B x TopQ x 4

        if use_focal_loss:
            scores = torch.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), num_top_queries, dim=-1)
            labels = index - index // self.num_labels * self.num_labels
            index = index // self.num_labels
            boxes = boxes.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, boxes.shape[-1]))
        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > num_top_queries:
                scores, index = torch.topk(scores, num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(
                    boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1])
                )

        results = []
        for lab, box, sco in zip(labels, boxes, scores):
            result = dict(
                labels=lab.detach().cpu(), boxes=box.detach().cpu(), scores=sco.detach().cpu()
            )
            results.append(result)
        return results

    def gt_postprocess(self, inputs, targets, orig_sizes):
        results = []
        for idx, target in enumerate(targets):
            lab = target["labels"]
            box = process_boxes(
                target["boxes"][None],
                inputs[idx].shape[1:],
                orig_sizes[idx][None],
                self.keep_ratio,
                inputs.device,
            )
            result = dict(labels=lab.detach().cpu(), boxes=box.squeeze(0).detach().cpu())
            results.append(result)
        return results

    @torch.no_grad()
    def get_preds_and_gt(self, val_loader: DataLoader) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]:
        all_gt, all_preds = [], []
        model = self.ema_model.model if self.ema_model else self.model
        model.eval()
        for idx, (inputs, targets, img_paths) in enumerate(val_loader):
            if inputs is None: continue
            inputs = inputs.to(self.device)
            if self.amp_enabled:
                with autocast(self.device, cache_enabled=True):
                    raw_res = model(inputs)
            else:
                raw_res = model(inputs)

            targets = [{k: v.to(self.device) if hasattr(v, "to") else v for k, v in t.items()} for t in targets]
            orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).float().to(self.device)
            preds = self.preds_postprocess(inputs, raw_res, orig_sizes)
            gt = self.gt_postprocess(inputs, targets, orig_sizes)
            all_preds.extend(preds)
            all_gt.extend(gt)

            if self.to_visualize_eval and idx <= 5:
                visualize(
                    img_paths, gt, filter_preds(preds, self.conf_thresh),
                    dataset_path=Path(self.cfg.train.data_path) / "images",
                    path_to_save=self.eval_preds_path,
                    label_to_name=self.label_to_name,
                )
        return all_gt, all_preds

    @staticmethod
    def get_metrics(gt, preds, conf_thresh: float, iou_thresh: float, extended: bool, path_to_save=None, mode=None):
        validator = Validator(gt, preds, conf_thresh=conf_thresh, iou_thresh=iou_thresh)
        metrics = validator.compute_metrics(extended=extended)
        if path_to_save:
            validator.save_plots(path_to_save / "plots" / mode)
        return metrics

    def evaluate(self, val_loader: DataLoader, conf_thresh: float, iou_thresh: float, path_to_save: Path, extended: bool, mode: str = None) -> Dict[str, float]:
        gt, preds = self.get_preds_and_gt(val_loader=val_loader)
        metrics = self.get_metrics(gt, preds, conf_thresh, iou_thresh, extended=extended, path_to_save=path_to_save, mode=mode)
        return metrics

    def train(self) -> None:
        best_metric, best_metric_stage1, cur_iter, ema_iter = 0, 0, 0, 0
        self.early_stopping_steps = 0
        one_epoch_time = None

        def optimizer_step():
            nonlocal ema_iter
            if self.amp_enabled:
                if self.clip_max_norm:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_max_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if self.clip_max_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_max_norm)
                self.optimizer.step()

            # OneCycleLR is stepped after each optimizer update
            if not self.self_lr_scheduler and self.scheduler:
                self.scheduler.step()

            self.optimizer.zero_grad()
            if self.ema_model:
                ema_iter += 1
                self.ema_model.update(ema_iter, self.model)

        for epoch in range(1, self.epochs + 1):
            epoch_start_time = time.time()
            self.model.train()
            self.loss_fn.train()
            losses = []
            
            # Inform dataloader about the current epoch
            self.base_loader.set_epoch(epoch)

            # Added: Handle ignore_background_epochs
            if self.cfg.train.ignore_background_epochs > 0:
                if epoch <= self.cfg.train.ignore_background_epochs:
                    self.train_loader.dataset.ignore_background = True
                    if epoch == 1:
                         logger.info(f"Ignoring background images for the first {self.cfg.train.ignore_background_epochs} epochs.")
                elif epoch == self.cfg.train.ignore_background_epochs + 1:
                    self.train_loader.dataset.ignore_background = False
                    logger.info("Resuming use of background images.")

            # EMA restart logic
            if epoch == self.stop_epoch:
                logger.info(f"--- End of Stage 1 at epoch {epoch}. Reloading best model and resetting EMA. ---")
                if (self.path_to_save / "best_stg1.pt").exists():
                    state_dict = torch.load(self.path_to_save / "best_stg1.pt", map_location=self.device)
                    self.model.load_state_dict(state_dict)
                    
                    if self.ema_model:
                        self.ema_model = ModelEMA(self.model, self.cfg.train.ema_restart_decay)
                        ema_iter = 0
                    
                    best_metric = 0
                else:
                    logger.warning("best_stg1.pt not found. Continuing without reloading.")

            with tqdm(self.train_loader, unit="batch") as tepoch:
                for batch_idx, (inputs, targets, _) in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch}/{self.epochs}")
                    if inputs is None: continue
                    cur_iter += 1

                    # FlatCosineLRScheduler is stepped each iteration (before optimizer step)
                    if self.self_lr_scheduler:
                        self.lr_scheduler.step(cur_iter, self.optimizer)

                    inputs = inputs.to(self.device)
                    targets = [{k: v.to(self.device) if hasattr(v, "to") else v for k, v in t.items()} for t in targets]

                    lr = self.optimizer.param_groups[-1]["lr"]
                    
                    if self.amp_enabled:
                        with autocast(self.device, cache_enabled=True):
                            output = self.model(inputs, targets=targets)
                        with autocast(self.device, enabled=False):
                            loss_dict = self.loss_fn(output, targets, epoch=epoch)
                        loss = sum(loss_dict.values()) / self.b_accum_steps
                        self.scaler.scale(loss).backward()
                    else:
                        output = self.model(inputs, targets=targets)
                        loss_dict = self.loss_fn(output, targets, epoch=epoch)
                        loss = sum(loss_dict.values()) / self.b_accum_steps
                        loss.backward()

                    if (batch_idx + 1) % self.b_accum_steps == 0:
                        optimizer_step()

                    losses.append(loss.item())

                    tepoch.set_postfix(
                        loss=np.mean(losses) * self.b_accum_steps,
                        eta=calculate_remaining_time(one_epoch_time, epoch_start_time, epoch, self.epochs, batch_idx + 1, len(self.train_loader)),
                        vram=f"{get_vram_usage()}%",
                    )
            
            # A final optimizer step if the number of batches is not a multiple of accumulation steps
            if (len(self.train_loader)) % self.b_accum_steps != 0:
                optimizer_step()

            if self.use_wandb:
                wandb.log({"lr": lr, "epoch": epoch})

            metrics = self.evaluate(
                val_loader=self.val_loader, conf_thresh=self.conf_thresh, iou_thresh=self.iou_thresh,
                extended=False, path_to_save=None
            )

           # Model saving and early stopping logic
            model_to_save = self.ema_model.model if self.ema_model else self.model
            self.path_to_save.mkdir(parents=True, exist_ok=True)
            torch.save(model_to_save.state_dict(), self.path_to_save / "last.pt")
            decision_metric = (metrics.get("mAP_50", 0) + metrics.get("f1", 0)) / 2

            if epoch < self.stop_epoch:  # Stage 1: Training with heavy augmentations
                if decision_metric > best_metric_stage1:
                    best_metric_stage1 = decision_metric
                    logger.info("Saving new best stage 1 model 🔥")
                    torch.save(model_to_save.state_dict(), self.path_to_save / "best_stg1.pt")
            else:  # Stage 2: Fine-tuning with light/no augmentations
                if decision_metric > best_metric:
                    best_metric = decision_metric
                    logger.info("Saving new best model (stage 2) 🔥")
                    torch.save(model_to_save.state_dict(), self.path_to_save / "model.pt")
                    self.early_stopping_steps = 0
                else:
                    self.early_stopping_steps += 1
                    logger.warning(f"Metric did not improve. Early stopping counter: {self.early_stopping_steps}/{self.early_stopping}")

            save_metrics({}, metrics, np.mean(losses) * self.b_accum_steps, epoch, path_to_save=None, use_wandb=self.use_wandb)

            one_epoch_time = time.time() - epoch_start_time

            if self.early_stopping and self.early_stopping_steps >= self.early_stopping:
                logger.info("Early stopping")
                break

@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig) -> None:
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()