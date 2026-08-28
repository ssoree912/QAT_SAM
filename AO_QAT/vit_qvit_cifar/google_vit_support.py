from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Final

import torch
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, create_transform
from timm.utils import accuracy
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from quant_vision_transformer import lowbit_VisionTransformer, resize_pos_embed
from qsam_training import QsamTrainMode


@dataclass(frozen=True)
class VitVariant:
    patch_size: int
    embed_dim: int
    depth: int
    num_heads: int
    checkpoint_url: str


@dataclass(frozen=True)
class ResumeState:
    start_epoch: int
    best_acc: float
    loaded_scheduler: bool


@dataclass(frozen=True)
class ResumeTarget:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: LRScheduler
    device: torch.device


@dataclass(frozen=True)
class CheckpointPayload:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: LRScheduler
    epoch: int
    best_acc: float


VARIANTS: Final = {
    "B_16": VitVariant(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        checkpoint_url=(
            "https://github.com/rwightman/pytorch-image-models/releases/download/"
            "v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth"
        ),
    ),
    "B_32": VitVariant(
        patch_size=32,
        embed_dim=768,
        depth=12,
        num_heads=12,
        checkpoint_url=(
            "https://github.com/rwightman/pytorch-image-models/releases/download/"
            "v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth"
        ),
    ),
}

METHOD_TO_TRAIN_MODE: Final = {
    "qvit": QsamTrainMode.BASELINE,
    "qvit_sam": QsamTrainMode.SAM,
    "qvit_qsam_twopass": QsamTrainMode.TWO_PASS,
}


def dataset_num_classes(data_set: str) -> int:
    if data_set == "CIFAR100":
        return 100
    if data_set == "IMNET":
        return 1000
    raise RuntimeError(f"unsupported dataset: {data_set}")


def build_model(args: argparse.Namespace) -> nn.Module:
    variant = VARIANTS[args.variant]
    return lowbit_VisionTransformer(
        nbits=args.nbits,
        img_size=args.input_size,
        patch_size=variant.patch_size,
        embed_dim=variant.embed_dim,
        depth=variant.depth,
        num_heads=variant.num_heads,
        mlp_ratio=4,
        qkv_bias=True,
        num_classes=dataset_num_classes(args.data_set),
        distilled=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )


def checkpoint_source(args: argparse.Namespace) -> str:
    if args.no_finetune:
        return ""
    if args.finetune == "auto":
        return VARIANTS[args.variant].checkpoint_url
    return args.finetune


def load_pretrained_checkpoint(model: nn.Module, source: str) -> None:
    if not source:
        print("[finetune] disabled")
        return
    if source.startswith("https://") or source.startswith("http://"):
        checkpoint = torch.hub.load_state_dict_from_url(source, map_location="cpu")
    else:
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    checkpoint_model = checkpoint.get("model", checkpoint)
    checkpoint_model = checkpoint_model.get("state_dict", checkpoint_model)
    state_dict = model.state_dict()
    for key in ("head.weight", "head.bias", "head_dist.weight", "head_dist.bias"):
        if key in checkpoint_model and key in state_dict and checkpoint_model[key].shape != state_dict[key].shape:
            del checkpoint_model[key]
    if "pos_embed" in checkpoint_model and checkpoint_model["pos_embed"].shape != state_dict["pos_embed"].shape:
        checkpoint_model["pos_embed"] = resize_pos_embed(checkpoint_model["pos_embed"], state_dict["pos_embed"])
    result = model.load_state_dict(checkpoint_model, strict=False)
    print(f"[finetune] init from {source}")
    print(f"[finetune] missing ({len(result.missing_keys)}): {result.missing_keys[:12]}")
    print(f"[finetune] unexpected ({len(result.unexpected_keys)}): {result.unexpected_keys[:12]}")


def resolve_resume_path(args: argparse.Namespace, output_dir: Path | None) -> Path | None:
    if args.fresh_start or not args.resume:
        return None
    if args.resume == "auto":
        if output_dir is None:
            return None
        checkpoint_path = output_dir / "checkpoint.pth"
        if checkpoint_path.exists():
            return checkpoint_path
        return None
    checkpoint_path = Path(args.resume)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"resume checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def load_training_checkpoint(target: ResumeTarget, checkpoint_path: Path) -> ResumeState:
    checkpoint = torch.load(checkpoint_path, map_location=target.device, weights_only=False)
    checkpoint_model = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    result = target.model.load_state_dict(checkpoint_model, strict=False)
    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    best_acc = float(checkpoint.get("best_acc1", 0.0))
    loaded_optimizer = "optimizer" in checkpoint
    loaded_scheduler = "scheduler" in checkpoint
    if loaded_optimizer:
        target.optimizer.load_state_dict(checkpoint["optimizer"])
    if loaded_scheduler:
        target.scheduler.load_state_dict(checkpoint["scheduler"])
    print(
        f"[resume] loaded {checkpoint_path} epoch={start_epoch - 1} "
        f"start_epoch={start_epoch} best={best_acc:.3f} "
        f"optimizer={loaded_optimizer} scheduler={loaded_scheduler}"
    )
    print(f"[resume] missing ({len(result.missing_keys)}): {result.missing_keys[:12]}")
    print(f"[resume] unexpected ({len(result.unexpected_keys)}): {result.unexpected_keys[:12]}")
    return ResumeState(start_epoch=start_epoch, best_acc=best_acc, loaded_scheduler=loaded_scheduler)


def fast_forward_scheduler(scheduler: LRScheduler, steps: int) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step\\(\\)`")
        for _ in range(steps):
            scheduler.step()


def save_training_checkpoint(payload: CheckpointPayload, checkpoint_path: Path) -> None:
    torch.save(
        {
            "model": payload.model.state_dict(),
            "optimizer": payload.optimizer.state_dict(),
            "scheduler": payload.scheduler.state_dict(),
            "epoch": payload.epoch,
            "best_acc1": payload.best_acc,
        },
        checkpoint_path,
    )


def build_transform(is_train: bool, input_size: int):
    if is_train:
        return create_transform(
            input_size=input_size,
            is_training=True,
            color_jitter=0.4,
            auto_augment="rand-m9-mstd0.5-inc1",
            interpolation="bicubic",
            re_prob=0.25,
            re_mode="pixel",
            re_count=1,
        )
    size = int((256 / 224) * input_size)
    return transforms.Compose(
        [
            transforms.Resize(size, interpolation=3),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    )


def build_loader(args: argparse.Namespace, is_train: bool) -> DataLoader:
    transform = build_transform(is_train, args.input_size)
    if args.data_set == "CIFAR100":
        dataset = datasets.CIFAR100(
            args.data_path,
            train=is_train,
            transform=transform,
            download=args.download,
        )
    elif args.data_set == "IMNET":
        split = "train" if is_train else "val"
        dataset = datasets.ImageFolder(Path(args.data_path) / split, transform=transform)
    else:
        raise RuntimeError(f"unsupported dataset: {args.data_set}")
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=is_train,
        num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=is_train,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc1 = 0.0
    total_acc5 = 0.0
    total_seen = 0
    for step, (images, targets) in enumerate(loader):
        if args.debug_val_batches and step >= args.debug_val_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        total_acc1 += acc1.item() * batch_size
        total_acc5 += acc5.item() * batch_size
        total_seen += batch_size
    return {
        "loss": total_loss / total_seen,
        "acc1": total_acc1 / total_seen,
        "acc5": total_acc5 / total_seen,
    }
