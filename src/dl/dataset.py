import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
from copy import deepcopy

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from src.dl.torchvision_transforms import RandomZoomOut
from src.dl.utils import (
    LetterboxRect,
    abs_xyxy_to_norm_xywh,
    get_mosaic_coordinate,
    norm_xywh_to_abs_xyxy,
    random_affine,
    seed_worker,
    vis_one_box,
)


def generate_scales(base_size, base_size_repeat):
    scale_repeat = (base_size - int(base_size * 0.75 / 32) * 32) // 32
    scales = [int(base_size * 0.75 / 32) * 32 + i * 32 for i in range(scale_repeat)]
    scales += [base_size] * base_size_repeat
    scales += [int(base_size * 1.25 / 32) * 32 - i * 32 for i in range(scale_repeat)]
    return scales

class CustomDataset(Dataset):
    def __init__(
        self,
        img_size: Tuple[int, int],  # h, w
        root_path: Path,
        split: pd.DataFrame,
        debug_img_processing: bool,
        mode: str,
        cfg: DictConfig,
    ) -> None:
        self.project_path = Path(cfg.train.root)
        self.root_path = root_path
        self.split = split
        self.target_h, self.target_w = img_size
        self.norm = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.debug_img_processing = debug_img_processing
        self.mode = mode
        self.ignore_background = False
        self.label_to_name = cfg.train.label_to_name

        self.mosaic_prob = cfg.train.mosaic_augs.mosaic_prob
        self.mosaic_scale = cfg.train.mosaic_augs.mosaic_scale
        self.degrees = cfg.train.mosaic_augs.degrees
        self.translate = cfg.train.mosaic_augs.translate
        self.shear = cfg.train.mosaic_augs.shear
        self.keep_ratio = cfg.train.keep_ratio
        self.use_one_class = cfg.train.use_one_class
        self.cases_to_debug = 20
        self.epoch = 0

        self.aug_policy_ops = cfg.train.aug_policy.ops
        self.aug_policy_epochs = cfg.train.aug_policy.epoch
        self.cfg_augs = cfg.train.augs

        self.normalize_transform = A.Compose(
            [
                A.Normalize(mean=self.norm[0], std=self.norm[1]),
                ToTensorV2(),
            ]
        )

        self.debug_img_path = Path(cfg.train.debug_img_path)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _debug_image(
        self, idx, image: torch.Tensor, boxes: torch.Tensor, classes: torch.Tensor, img_path: Path
    ) -> None:
        # The image is now a tensor, so we handle it directly
        image_np = image.cpu().numpy()
        
        # Denormalize if it was normalized
        if image_np.max() <= 1.0 and image_np.min() < 0:
            mean = np.array(self.norm[0]).reshape(-1, 1, 1)
            std = np.array(self.norm[1]).reshape(-1, 1, 1)
            image_np = (image_np * std) + mean

        image_np = np.transpose(image_np, (1, 2, 0))
        image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
        image_np = np.ascontiguousarray(image_np)

        # Convert boxes from xyxy tensor to numpy int for drawing
        boxes_np = boxes.cpu().numpy().astype(int)
        classes_np = classes.cpu().numpy()
        for box, class_id in zip(boxes_np, classes_np):
            vis_one_box(image_np, box, class_id, mode="gt", label_to_name=self.label_to_name)

        save_dir = self.debug_img_path / self.mode
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{idx}_idx_{img_path.stem}_debug.jpg"
        cv2.imwrite(str(save_path), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

    def _get_data(self, idx: int) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
        image_path = Path(self.split.iloc[idx].values[0])
        image = cv2.imread(str(self.root_path / "images" / f"{image_path}"))
        assert image is not None, f"Image wasn't loaded: {image_path}"

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape
        orig_size = torch.tensor([height, width])

        labels_path = self.root_path / "labels" / f"{image_path.stem}.txt"
        if labels_path.exists() and labels_path.stat().st_size:
            targets = np.loadtxt(labels_path)
            if targets.ndim == 1:
                targets = targets.reshape(1, -1)

            if self.use_one_class:
                targets[:, 0] = 0

            abs_boxes = norm_xywh_to_abs_xyxy(targets[:, 1:], height, width).astype(np.float32)
            areas = (abs_boxes[:, 2] - abs_boxes[:, 0]) * (abs_boxes[:, 3] - abs_boxes[:, 1])
            targets = np.concatenate([targets[:, :1], abs_boxes, areas[:, None]], axis=1)
            return image, targets, orig_size

        targets = np.zeros((0, 6), dtype=np.float32)
        return image, targets, orig_size

    def _load_mosaic(self, idx):
        mosaic_targets = []
        yc = int(random.uniform(self.target_h * 0.6, self.target_h * 1.4))
        xc = int(random.uniform(self.target_w * 0.6, self.target_w * 1.4))
        indices = [idx] + [random.randint(0, self.__len__() - 1) for _ in range(3)]

        for i_mosaic, m_idx in enumerate(indices):
            img, targets, _ = self._get_data(m_idx)
            (h, w, c) = img.shape[:3]
            scale = min(self.target_h / h, self.target_w / w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            (h, w, c) = img.shape[:3]

            if i_mosaic == 0:
                mosaic_img = np.full(
                    (self.target_h * 2, self.target_w * 2, c), 114, dtype=np.uint8
                )

            (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
                mosaic_img, i_mosaic, xc, yc, w, h, self.target_h, self.target_w
            )
            mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]
            padw, padh = l_x1 - s_x1, l_y1 - s_y1

            targets = targets.copy()
            if targets.size > 0:
                targets[:, 1:5] *= scale
                targets[:, 1] += padw
                targets[:, 2] += padh
                targets[:, 3] += padw
                targets[:, 4] += padh
                targets[:, 5] *= scale * scale
            mosaic_targets.append(targets)

        if len(mosaic_targets) > 0:
            mosaic_targets = np.concatenate(mosaic_targets, 0)
            mosaic_targets[:, 1:5] = np.clip(
                mosaic_targets[:, 1:5], 0, 2 * max(self.target_h, self.target_w)
            )
        else:
            mosaic_targets = np.zeros((0, 6), dtype=np.float32)

        mosaic_img, mosaic_targets = random_affine(
            mosaic_img,
            mosaic_targets,
            segments=[],
            target_size=(self.target_w, self.target_h),
            degrees=self.degrees,
            translate=self.translate,
            scales=self.mosaic_scale,
            shear=self.shear,
        )

        box_heights = mosaic_targets[:, 4] - mosaic_targets[:, 2]
        box_widths = mosaic_targets[:, 3] - mosaic_targets[:, 1]
        # Ensure boxes have a positive area
        mosaic_targets = mosaic_targets[(box_widths > 0) & (box_heights > 0)]

        # Return numpy arrays and lists, not tensors
        labels = mosaic_targets[:, 0].tolist() if mosaic_targets.shape[0] > 0 else []
        # Convert the numpy array to a list to be consistent
        boxes = mosaic_targets[:, 1:5].tolist() if mosaic_targets.shape[0] > 0 else []
        
        # We don't need to return area, as it can be recalculated after transforms
        return mosaic_img, labels, boxes, np.array([self.target_h, self.target_w])

    def _get_transform(self, mosaic_applied=False):
        # Common components for all modes
        to_tensor = [ToTensorV2()]
        resize = [A.Resize(self.target_h, self.target_w, interpolation=cv2.INTER_AREA)]

        # The normalization and ToTensor steps are now part of the main pipeline
        normalize_and_tensor = [
            A.Normalize(mean=self.norm[0], std=self.norm[1]),
            ToTensorV2(),
        ]

        if self.mode == "train":
            # Check if strong augmentations are active for the current epoch
            use_strong_augs = self.aug_policy_epochs[0] <= self.epoch < self.aug_policy_epochs[2]

            # Start with basic augmentations
            augs = [A.HorizontalFlip(p=self.cfg_augs.left_right_flip)]

            if use_strong_augs:
                policy_ops = self.aug_policy_ops
                if "RandomPhotometricDistort" in policy_ops:
                    augs.append(A.ColorJitter(p=self.cfg_augs.photometric_distort_p))

                # Ensure IoUCrop and ZoomOut are never applied with Mosaic
                if not mosaic_applied:
                    if "RandomIoUCrop" in policy_ops:
                        augs.append(
                            A.RandomSizedBBoxSafeCrop(
                                height=self.target_h,
                                width=self.target_w,
                                p=self.cfg_augs.iou_crop_p,
                                erosion_rate=0.2,
                            )
                        )
                    if "RandomZoomOut" in policy_ops:
                        augs.append(RandomZoomOut(side_range=(1.0, 4.0), p=self.cfg_augs.zoom_out_p))

            return A.Compose(
                augs + resize + normalize_and_tensor,
                bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
            )
        else:  # val, test, bench modes
            return A.Compose(
                resize + normalize_and_tensor,
                bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
            )

    def __getitem__(self, idx: int):
        image_path = Path(self.split.iloc[idx].values[0])
        mosaic_applied = False

        mosaic_active = (
            "Mosaic" in self.aug_policy_ops
            and self.aug_policy_epochs[0] <= self.epoch < self.aug_policy_epochs[2]
        )

        if self.mode == "train" and mosaic_active and random.random() < self.mosaic_prob:
            image, class_labels, bboxes, orig_size_np = self._load_mosaic(idx)
            orig_size = torch.from_numpy(orig_size_np)
            mosaic_applied = True
        else:
            image, targets, orig_size = self._get_data(idx)

            if self.ignore_background and targets.shape[0] == 0 and self.mode == "train":
                return None

            bboxes = targets[:, 1:5].tolist() if targets.shape[0] > 0 else []
            class_labels = targets[:, 0].tolist() if targets.shape[0] > 0 else []

        # robust filtering
        if bboxes:
            bboxes_np = np.array(bboxes, dtype=np.float32)
            labels_np = np.array(class_labels, dtype=np.int64)

            # Ensure that x_max > x_min and y_max > y_min. This is the crucial fix.
            keep = (bboxes_np[:, 2] > bboxes_np[:, 0]) & (bboxes_np[:, 3] > bboxes_np[:, 1])
            
            bboxes = bboxes_np[keep].tolist()
            class_labels = labels_np[keep].tolist()

        # Get the appropriate transform pipeline
        transform = self._get_transform(mosaic_applied=mosaic_applied)
        
        # Apply transforms to image, boxes, and labels
        transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        
        image = transformed["image"]

        boxes = (
            torch.tensor(transformed["bboxes"], dtype=torch.float32)
            if len(transformed["bboxes"]) > 0
            else torch.zeros((0, 4), dtype=torch.float32)
        )
        labels = torch.tensor(transformed["class_labels"], dtype=torch.int64)

        # This secondary check is good practice but the primary fix is above
        valid_boxes_mask = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes, labels = boxes[valid_boxes_mask], labels[valid_boxes_mask]
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        if self.debug_img_processing and idx <= self.cases_to_debug:
            # Note: For debugging, image is now a tensor. We adjust _debug_image accordingly.
            self._debug_image(idx, image, boxes, labels, image_path)

        boxes = abs_xyxy_to_norm_xywh(boxes.numpy(), image.shape[1], image.shape[2])
        boxes = torch.tensor(boxes, dtype=torch.float32)

        return image, labels, boxes, areas, image_path, orig_size

    def __len__(self):
        return len(self.split)


class Loader:
    def __init__(
        self,
        root_path: Path,
        img_size: Tuple[int, int],
        batch_size: int,
        num_workers: int,
        cfg: DictConfig,
        debug_img_processing: bool = False,
    ) -> None:
        self.root_path = root_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cfg = cfg
        self.use_one_class = cfg.train.use_one_class
        self.debug_img_processing = debug_img_processing
        self.base_size = cfg.train.img_size[0]
        self._get_splits()
        self.class_names = list(cfg.train.label_to_name.values())

        self.base_size_repeat = cfg.train.dataloader.get("base_size_repeat")
        self.stop_epoch_multiscale = cfg.train.lr_scheduler.get("no_aug_epochs") # Or a dedicated param
        self.scales = generate_scales(self.base_size, self.base_size_repeat) if self.base_size_repeat is not None else None

        # Store mixup and copyblend configs from the main config
        self.mixup_cfg = cfg.train.get("mixup_augs", {"mixup_prob": 0.0, "mixup_epochs": [0, 0]})
        self.copyblend_cfg = cfg.train.copyblend_augs
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        if hasattr(self, "train_loader"):
            self.train_loader.dataset.set_epoch(epoch)

    def _get_splits(self) -> None:
        self.splits = {"train": None, "val": None, "test": None}
        for split_name in self.splits.keys():
            if (self.root_path / f"{split_name}.csv").exists():
                logger.info("Reading train/val csv files")
                self.splits[split_name] = pd.read_csv(
                    self.root_path / f"{split_name}.csv", header=None
                )
            else:
                logger.info("csv for train/val do not exist")
                self.splits[split_name] = []

    def _get_label_stats(self) -> Dict:
        if self.use_one_class:
            classes = {"target": 0}
        else:
            classes = {class_name: 0 for class_name in self.class_names}
        for split in self.splits.values():
            if not len(split):
                continue
            for image_path in split.iloc[:, 0]:
                labels_path = self.root_path / "labels" / f"{Path(image_path).stem}.txt"
                if not (labels_path.exists() and labels_path.stat().st_size):
                    continue
                targets = np.loadtxt(labels_path)
                if targets.ndim == 1:
                    targets = targets.reshape(1, -1)
                labels = targets[:, 0]
                for class_id in labels:
                    if self.use_one_class:
                        classes["target"] += 1
                    else:
                        classes[self.class_names[int(class_id)]] += 1
        return classes

    def _get_amount_of_background(self):
        labels = set()
        for label_path in (self.root_path / "labels").iterdir():
            if not label_path.stat().st_size:
                label_path.unlink()  # remove empty txt files
            elif not (label_path.stem.startswith(".") and label_path.name == "labels.txt"):
                labels.add(label_path.stem)

        raw_split_images = set()
        for split in self.splits.values():
            if len(split):
                raw_split_images.update(split.iloc[:, 0].values)

        split_images = []
        for split_image in raw_split_images:
            split_images.append(Path(split_image).stem)

        images = {
            f.stem for f in (self.root_path / "images").iterdir() if not f.stem.startswith(".")
        }
        images = images.intersection(split_images)
        return len(images - labels)

    def _build_dataloader_impl(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        collate_fn = self.val_collate_fn if dataset.mode != "train" else self.train_collate_fn
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            collate_fn=collate_fn,
            worker_init_fn=seed_worker,
            prefetch_factor=4,
            pin_memory=True,
        )

    def build_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_ds = CustomDataset(
            self.img_size,
            self.root_path,
            self.splits["train"],
            self.debug_img_processing,
            "train",
            self.cfg,
        )
        val_ds = CustomDataset(
            self.img_size, self.root_path, self.splits["val"], self.debug_img_processing, "val", self.cfg
        )

        self.train_loader = self._build_dataloader_impl(train_ds, shuffle=True)
        val_loader = self._build_dataloader_impl(val_ds)

        test_loader, test_ds_len = None, 0
        if len(self.splits["test"]):
            test_ds = CustomDataset(
                self.img_size,
                self.root_path,
                self.splits["test"],
                self.debug_img_processing,
                "test",
                self.cfg,
            )
            test_loader = self._build_dataloader_impl(test_ds)
            test_ds_len = len(test_ds)

        logger.info(
            f"Images in train: {len(train_ds)}, val: {len(val_ds)}, test: {test_ds_len}"
        )
        logger.info(f"Objects count: {self._get_label_stats()}")
        logger.info(f"Background images: {self._get_amount_of_background()}")
        return self.train_loader, val_loader, test_loader

    def _collate_fn(self, batch) -> Tuple:
        batch = [item for item in batch if item is not None]
        if not batch:
            return None, None, None

        images, targets, img_paths = [], [], []
        for item in batch:
            target_dict = {
                "boxes": item[2],
                "labels": item[1],
                "area": item[3],
                "orig_size": item[5],
            }
            images.append(item[0])
            targets.append(target_dict)
            img_paths.append(item[4])

        images = torch.stack(images, dim=0)
        
        return images, targets, img_paths

    def val_collate_fn(self, batch) -> Tuple:
        return self._collate_fn(batch)

    def train_collate_fn(self, batch) -> Tuple:
        batch = [item for item in batch if item is not None]
        if not batch:
            return None, None, None

        images, targets, img_paths = [], [], []
        for item in batch:
            target_dict = {
                "boxes": item[2],
                "labels": item[1],
                "area": item[3],
                "orig_size": item[5],
            }
            images.append(item[0])
            targets.append(target_dict)
            img_paths.append(item[4])

        images = torch.stack(images, dim=0)

        # DEIMv2 Augmentations
        # Check if Mixup/CopyBlend should be active for the current epoch
        mixup_active = (
            self.mixup_cfg.get("mixup_prob", 0.0) > 0 and
            self.mixup_cfg.get("mixup_epochs", [0, 0])[0] <= self.epoch < self.mixup_cfg.get("mixup_epochs", [0, 0])[1]
        )
        copyblend_active = (
            self.copyblend_cfg.copyblend_prob > 0 and
            self.copyblend_cfg.copyblend_epochs[0]
            <= self.epoch
            < self.copyblend_cfg.copyblend_epochs[1]
        )

        r = random.random()
        mixup_prob = self.mixup_cfg["mixup_prob"] if mixup_active else 0.0
        copyblend_prob = self.copyblend_cfg.copyblend_prob if copyblend_active else 0.0

        if r < mixup_prob:

            beta = random.uniform(0.8, 1.0) # Or another range, e.g., 0.45, 0.55 as in CopyBlend

            # Shift images for mixup
            shifted_images = images.roll(shifts=1, dims=0)
            images = images.mul_(beta).add_(shifted_images.mul_(1.0 - beta))

            # Shift targets list
            shifted_targets = targets[-1:] + targets[:-1]
            
            # Create a new list of updated targets to avoid modifying originals in place
            new_targets = []
            for i in range(len(targets)):
                new_target_dict = {
                    "boxes": torch.cat([targets[i]["boxes"], shifted_targets[i]["boxes"]], dim=0),
                    "labels": torch.cat([targets[i]["labels"], shifted_targets[i]["labels"]], dim=0),
                    "area": torch.cat([targets[i]["area"], shifted_targets[i]["area"]], dim=0),
                    "orig_size": targets[i]["orig_size"] # Keep original size from the primary image
                }
                new_targets.append(new_target_dict)
            targets = new_targets

        elif r < mixup_prob + copyblend_prob:

            objects_pool = defaultdict(list)
            img_height, img_width = images.shape[-2:]
            beta = random.uniform(0.45, 0.55)

            for i in range(len(images)):
                valid_indices = [
                    idx
                    for idx, area in enumerate(targets[i]["area"])
                    if area >= self.cfg.train.copyblend_augs.area_threshold
                ]
                for idx in valid_indices:
                    objects_pool["boxes"].append(targets[i]["boxes"][idx])
                    objects_pool["labels"].append(targets[i]["labels"][idx])
                    objects_pool["image_idx"].append(i)

            if len(objects_pool["boxes"]) > 0:
                updated_images = images.clone()
                updated_targets = deepcopy(targets)
                for i in range(len(images)):
                    if self.cfg.train.copyblend_augs.random_num_objects:
                        num_to_blend = random.randint(1, self.cfg.train.copyblend_augs.num_objects)
                    else:
                        num_to_blend = self.cfg.train.copyblend_augs.num_objects
                    
                    num_to_blend = min(num_to_blend, len(objects_pool["boxes"]))
                    if num_to_blend == 0:
                        continue
                    
                    selected_indices = random.sample(range(len(objects_pool["boxes"])), num_to_blend)

                    blend_mixup_ratios = []
                    for idx in selected_indices:
                        box, label, source_idx = (
                            objects_pool["boxes"][idx],
                            objects_pool["labels"][idx],
                            objects_pool["image_idx"][idx],
                        )
                        cx, cy, w, h = box
                        x1_src, y1_src = int((cx - w / 2) * img_width), int((cy - h / 2) * img_height)
                        x2_src, y2_src = int((cx + w / 2) * img_width), int((cy + h / 2) * img_height)

                        patch_w, patch_h = x2_src - x1_src, y2_src - y1_src
                        if patch_w <= 0 or patch_h <= 0:
                            continue

                        copy_patch = images[source_idx, :, y1_src:y2_src, x1_src:x2_src]

                        x1_dst = random.randint(0, img_width - patch_w) if patch_w < img_width else 0
                        y1_dst = random.randint(0, img_height - patch_h) if patch_h < img_height else 0

                        if self.cfg.train.copyblend_augs.with_expand:
                            alpha = random.uniform(*self.cfg.train.copyblend_augs.expand_ratios)
                            expand_w, expand_h = int(patch_w * alpha), int(patch_h * alpha)

                            # Calculate how much we can actually expand on each side for both source and destination
                            # This prevents going out of bounds.
                            src_expand_left = min(x1_src, expand_w)
                            src_expand_top = min(y1_src, expand_h)
                            src_expand_right = min(img_width - x2_src, expand_w)
                            src_expand_bottom = min(img_height - y2_src, expand_h)

                            dst_expand_left = min(x1_dst, expand_w)
                            dst_expand_top = min(y1_dst, expand_h)
                            dst_expand_right = min(img_width - (x1_dst + patch_w), expand_w)
                            dst_expand_bottom = min(img_height - (y1_dst + patch_h), expand_h)
                            
                            # Take the minimum possible expansion to ensure slices match
                            final_expand_left = min(src_expand_left, dst_expand_left)
                            final_expand_top = min(src_expand_top, dst_expand_top)
                            final_expand_right = min(src_expand_right, dst_expand_right)
                            final_expand_bottom = min(src_expand_bottom, dst_expand_bottom)

                            # Define the final source and destination slices, which are now guaranteed to be the same size
                            x1_src_exp, y1_src_exp = x1_src - final_expand_left, y1_src - final_expand_top
                            x2_src_exp, y2_src_exp = x2_src + final_expand_right, y2_src + final_expand_bottom

                            x1_dst_exp, y1_dst_exp = x1_dst - final_expand_left, y1_dst - final_expand_top
                            x2_dst_exp, y2_dst_exp = x1_dst + patch_w + final_expand_right, y1_dst + patch_h + final_expand_bottom
                            
                            copy_patch_expanded = images[source_idx, :, y1_src_exp:y2_src_exp, x1_src_exp:x2_src_exp]
                            
                            if self.copyblend_cfg.get("copyblend_type") == "blend":
                                blended_patch = (
                                    updated_images[i, :, y1_dst_exp:y2_dst_exp, x1_dst_exp:x2_dst_exp]
                                    * beta
                                    + copy_patch_expanded * (1 - beta)
                                )
                                updated_images[
                                    i, :, y1_dst_exp:y2_dst_exp, x1_dst_exp:x2_dst_exp
                                ] = blended_patch
                            else:  # 'paste'
                                updated_images[
                                    i, :, y1_dst_exp:y2_dst_exp, x1_dst_exp:x2_dst_exp
                                ] = copy_patch_expanded
                        else: # Original non-expanded logic
                            if self.copyblend_cfg.get("copyblend_type") == "blend":
                                blended_patch = (
                                    updated_images[i, :, y1_dst : y1_dst + patch_h, x1_dst : x1_dst + patch_w]
                                    * beta
                                    + copy_patch * (1 - beta)
                                )
                                updated_images[
                                    i, :, y1_dst : y1_dst + patch_h, x1_dst : x1_dst + patch_w
                                ] = blended_patch
                            else:  # 'paste'
                                updated_images[
                                    i, :, y1_dst : y1_dst + patch_h, x1_dst : x1_dst + patch_w
                                ] = copy_patch

                        new_cx, new_cy = (x1_dst + patch_w / 2) / img_width, (
                            y1_dst + patch_h / 2
                        ) / img_height
                        new_w, new_h = patch_w / img_width, patch_h / img_height
                        blend_mixup_ratios.append(1.0 - beta)

                        # The criterion doesn't support mixup weights, so we only add the new boxes/labels.
                        new_box = torch.tensor([new_cx, new_cy, new_w, new_h], device=images.device)
                        updated_targets[i]["boxes"] = torch.cat([updated_targets[i]["boxes"], new_box.unsqueeze(0)])
                        updated_targets[i]["labels"] = torch.cat([updated_targets[i]["labels"], label.unsqueeze(0)])
                        updated_targets[i]["area"] = torch.cat([updated_targets[i]["area"], torch.tensor([patch_w * patch_h], device=images.device)])

                images = updated_images
                targets = updated_targets

        # Multiscale training - active except for the last `no_aug_epochs`
        if self.scales is not None and self.epoch < self.stop_epoch_multiscale:
            sz = random.choice(self.scales)
            # Only resize if the size is different to avoid unnecessary operations
            if images.shape[-1] != sz:
                images = F.interpolate(images, size=sz, mode='bilinear', align_corners=False)

        # Apply normalization as the final step
        #images = self.normalize_transform(image=images.permute(0, 2, 3, 1).numpy())["image"]

        return images, targets, img_paths
