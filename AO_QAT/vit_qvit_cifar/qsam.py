"""qSAM enabler for the Q-ViT quantized model.

qSAM (quantized Sharpness-Aware Minimization) is grafted onto Q-ViT's
LSQ layers (Quant.LinearQ / Quant.Conv2dQ). The perturbation math lives in
Quant._qsam_perturbation and the per-layer forward; this module only flips it on and
allocates the gradient buffer, *after* the pretrained weights are loaded so the buffer
matches the initialized weights.

Contract (mirrors AO_QAT/quan/func.py + resnet_cifar100/train.py):
  - forward: w_q = w_q + rho * alpha * sign(g_prev) * mask   (top-K |g_prev|, dequantized domain)
  - S2 loop: after optimizer.step(), call save_grad_for_qsam() for the next step.
  - 2-pass loop: call save_grad_for_qsam() after the first backward, then forward again.
"""
import torch

from Quant import LinearQ, Conv2dQ


def enable_qsam(model, ratio, rho, include_head=False, warmup_epochs=0):
    """Turn on qSAM for every quantized weight layer (LinearQ / Conv2dQ).

    Args:
        model:  the (already pretrained-initialized) network.
        ratio:  K/n fraction of weights perturbed per layer (e.g. 1e-3).
        rho:    perturbation step scale (<=1); one integer level shift at rho=1.
        include_head: also perturb the 8-bit classifier head (default: skip it).
        warmup_epochs: if >0, layers start gated off (qsam_on=False); the training loop
                       flips them on via set_qsam_on() once epoch >= warmup_epochs so
                       alpha/weights settle first.
    Returns:
        list of the enabled module names (for logging / sanity checks).
    """
    enabled = []
    for name, m in model.named_modules():
        if isinstance(m, (LinearQ, Conv2dQ)):
            if not include_head and (name.endswith('head') or name.endswith('head_dist')):
                continue
            if getattr(m, 'alpha', None) is None:   # a "fake"/disabled quantizer (nbits<0)
                continue
            m.use_qsam = True
            m.qsam_ratio = ratio
            m.qsam_rho = rho
            m.qsam_on = (warmup_epochs <= 0)
            m.register_buffer('g_prev', torch.zeros_like(m.weight.detach()))
            enabled.append(name)
    return enabled


def set_qsam_on(model, on):
    """Gate the perturbation on/off at runtime (used for warmup)."""
    for m in model.modules():
        if isinstance(m, (LinearQ, Conv2dQ)) and getattr(m, 'use_qsam', False):
            m.qsam_on = bool(on)
