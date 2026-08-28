from dataclasses import dataclass
from enum import Enum
from typing import Dict

import torch
import torch.nn as nn
from timm.utils import accuracy
from torch.utils.data import DataLoader

from Quant import Conv2dQ, LinearQ


class QsamTrainMode(Enum):
    BASELINE = "baseline"
    SAM = "sam"
    TWO_PASS = "two_pass"


@dataclass(frozen=True)
class TrainControls:
    debug_train_batches: int
    print_freq: int
    sam_rho: float = 0.05


@dataclass(frozen=True)
class TrainRuntime:
    model: nn.Module
    loader: DataLoader
    optimizer: torch.optim.Optimizer
    criterion: nn.Module
    device: torch.device


def train_one_epoch(
    runtime: TrainRuntime,
    controls: TrainControls,
    mode: QsamTrainMode,
) -> Dict[str, float]:
    trainers = {
        QsamTrainMode.BASELINE: _train_one_epoch_baseline,
        QsamTrainMode.SAM: _train_one_epoch_vanilla_sam,
        QsamTrainMode.TWO_PASS: _train_one_epoch_two_pass_qsam,
    }
    return trainers[mode](runtime, controls)


def _train_one_epoch_baseline(runtime: TrainRuntime, controls: TrainControls) -> Dict[str, float]:
    return _train_one_epoch_single_pass(runtime, controls)


def _train_one_epoch_vanilla_sam(
    runtime: TrainRuntime,
    controls: TrainControls,
) -> Dict[str, float]:
    runtime.model.train()
    sam_params = _collect_sam_weight_params(runtime.model)
    total_loss = 0.0
    total_acc = 0.0
    total_seen = 0
    for step, (images, targets) in enumerate(runtime.loader):
        if controls.debug_train_batches and step >= controls.debug_train_batches:
            break
        images = images.to(runtime.device, non_blocking=True)
        targets = targets.to(runtime.device, non_blocking=True)

        runtime.optimizer.zero_grad()
        outputs = runtime.model(images)
        clean_loss = runtime.criterion(outputs, targets)
        clean_loss.backward()
        perturbations = _add_vanilla_sam_perturbation(sam_params, controls.sam_rho)

        runtime.optimizer.zero_grad()
        try:
            perturbed_outputs = runtime.model(images)
            perturbed_loss = runtime.criterion(perturbed_outputs, targets)
            perturbed_loss.backward()
        finally:
            _restore_vanilla_sam_perturbation(perturbations)
        runtime.optimizer.step()

        acc1 = accuracy(outputs, targets, topk=(1,))[0]
        batch_size = images.shape[0]
        total_loss += clean_loss.item() * batch_size
        total_acc += acc1.item() * batch_size
        total_seen += batch_size
        if step % controls.print_freq == 0:
            print(
                f"train step={step} loss={clean_loss.item():.4f} "
                f"sam_loss={perturbed_loss.item():.4f} "
                f"avg_loss={total_loss / total_seen:.4f} acc1={acc1.item():.2f}"
            )
    return {"loss": total_loss / total_seen, "acc1": total_acc / total_seen}


def _train_one_epoch_single_pass(
    runtime: TrainRuntime,
    controls: TrainControls,
) -> Dict[str, float]:
    runtime.model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_seen = 0
    for step, (images, targets) in enumerate(runtime.loader):
        if controls.debug_train_batches and step >= controls.debug_train_batches:
            break
        images = images.to(runtime.device, non_blocking=True)
        targets = targets.to(runtime.device, non_blocking=True)
        runtime.optimizer.zero_grad()
        outputs = runtime.model(images)
        loss = runtime.criterion(outputs, targets)
        loss.backward()
        runtime.optimizer.step()
        acc1 = accuracy(outputs, targets, topk=(1,))[0]
        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        total_acc += acc1.item() * batch_size
        total_seen += batch_size
        if step % controls.print_freq == 0:
            print(
                f"train step={step} loss={loss.item():.4f} "
                f"avg_loss={total_loss / total_seen:.4f} acc1={acc1.item():.2f}"
            )
    return {"loss": total_loss / total_seen, "acc1": total_acc / total_seen}


def _train_one_epoch_two_pass_qsam(
    runtime: TrainRuntime,
    controls: TrainControls,
) -> Dict[str, float]:
    runtime.model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_seen = 0
    for step, (images, targets) in enumerate(runtime.loader):
        if controls.debug_train_batches and step >= controls.debug_train_batches:
            break
        images = images.to(runtime.device, non_blocking=True)
        targets = targets.to(runtime.device, non_blocking=True)

        _set_qsam_on(runtime.model, False)
        runtime.optimizer.zero_grad()
        outputs = runtime.model(images)
        clean_loss = runtime.criterion(outputs, targets)
        clean_loss.backward()
        _save_qsam_grads(runtime.model)

        _set_qsam_on(runtime.model, True)
        runtime.optimizer.zero_grad()
        perturbed_outputs = runtime.model(images)
        perturbed_loss = runtime.criterion(perturbed_outputs, targets)
        perturbed_loss.backward()
        runtime.optimizer.step()

        acc1 = accuracy(outputs, targets, topk=(1,))[0]
        batch_size = images.shape[0]
        total_loss += clean_loss.item() * batch_size
        total_acc += acc1.item() * batch_size
        total_seen += batch_size
        if step % controls.print_freq == 0:
            print(
                f"train step={step} loss={clean_loss.item():.4f} "
                f"sam_loss={perturbed_loss.item():.4f} "
                f"avg_loss={total_loss / total_seen:.4f} acc1={acc1.item():.2f}"
            )
    return {"loss": total_loss / total_seen, "acc1": total_acc / total_seen}


def _save_qsam_grads(model: nn.Module) -> None:
    for module in model.modules():
        save_grad = getattr(module, "save_grad_for_qsam", None)
        if save_grad is not None:
            save_grad()


def _set_qsam_on(model: nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if getattr(module, "use_qsam", False):
            module.qsam_on = enabled


def _collect_sam_weight_params(model: nn.Module):
    params = []
    for name, module in model.named_modules():
        if name.endswith("head") or name.endswith("head_dist"):
            continue
        if isinstance(module, (LinearQ, Conv2dQ)) and module.weight.requires_grad:
            params.append(module.weight)
    return params


def _add_vanilla_sam_perturbation(params, rho: float):
    norm_sq = None
    for param in params:
        if param.grad is None:
            continue
        grad = param.grad.detach()
        term = torch.sum(grad * grad)
        norm_sq = term if norm_sq is None else norm_sq + term
    if norm_sq is None:
        return []

    norm = torch.sqrt(norm_sq)
    if norm.item() == 0.0:
        return []

    perturbations = []
    scale = rho / (norm + 1e-12)
    with torch.no_grad():
        for param in params:
            if param.grad is None:
                continue
            delta = param.grad.detach() * scale
            param.add_(delta)
            perturbations.append((param, delta))
    return perturbations


def _restore_vanilla_sam_perturbation(perturbations) -> None:
    with torch.no_grad():
        for param, delta in perturbations:
            param.sub_(delta)
