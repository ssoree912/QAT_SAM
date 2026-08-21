import sys
import torch
import torch as t
from torch.nn import functional as F
from torch import Tensor

sys.path.append("..")
from globalVal import globalVal


def _get_w_quant_step(quan_w_fn):
    """Effective Δ (quantization step) of a weight quantizer, detached."""
    # Lazy import to avoid circular dependency on .quantizer
    from .quantizer import LSQ, AOQ

    if isinstance(quan_w_fn, LSQ):
        return quan_w_fn.step_size.detach().view(-1)[0]
    if isinstance(quan_w_fn, AOQ):
        e = float(globalVal.epoch)
        scale = (
            quan_w_fn.init_interval
            if e <= quan_w_fn.stage1_end_epoch
            else quan_w_fn.step_size
        )
        step = (scale * quan_w_fn.alpha).detach().view(-1)[0]
        return step
    return torch.tensor(1.0, device=next(quan_w_fn.parameters()).device)


def _qsam_perturbation(weight, g_prev, delta_step, ratio, rho):
    """Single-step quantized SAM perturbation.

    Geometry follows quantizedSAM RA-qSAM, p=2 (Eq. 26): a top-K signed integer
    level-shift in the *quantized-weight* domain on the K most sensitive coords.

      K   = max(1, round(ratio * n))   (K = κ^2 budget expressed as a fraction)
      I   = top-K indices of |g|
      δ_i = ρ · Δ · sign(g_i)  for i ∈ I, else 0

    The paper's lattice form uses a one-level shift (ρ=1). Because we follow
    S²-SAM and reuse the *stale* prior-step gradient g_prev (not a fresh g_q),
    that full level can be too aggressive, so ρ ≤ 1 tempers the step size.

    Efficiency follows S²-SAM (sam2.pdf, Eq. 5-7): instead of a second
    forward/backward to obtain a fresh g_q, we reuse the prior step's gradient
    g_prev (saved by save_grad_for_qsam before the optimizer step). This keeps
    the update single-step / zero-extra-cost. g_prev is the STE proxy m(w)⊙g_q.
    """
    if g_prev.abs().sum().item() == 0:
        return None
    n = weight.numel()
    K = max(1, int(round(ratio * n)))
    g_flat = g_prev.detach().reshape(-1)
    _, topk_idx = torch.topk(g_flat.abs(), K)
    mask = torch.zeros(n, device=weight.device, dtype=weight.dtype)
    mask[topk_idx] = 1.0
    mask = mask.view_as(weight)
    perturb = rho * delta_step.to(weight.dtype) * torch.sign(g_prev.detach()) * mask
    return perturb.detach()


class QuanConv2d(t.nn.Conv2d):
    def __init__(
        self,
        m: t.nn.Conv2d,
        quan_w_fn=None,
        quan_a_fn=None,
        use_qsam: bool = False,
        qsam_ratio: float = 0.001,
        qsam_rho: float = 1.0,
    ):
        assert type(m) == t.nn.Conv2d
        super().__init__(
            m.in_channels,
            m.out_channels,
            m.kernel_size,
            stride=m.stride,
            padding=m.padding,
            dilation=m.dilation,
            groups=m.groups,
            bias=True if m.bias is not None else False,
            padding_mode=m.padding_mode,
        )
        self.quan_w_fn = quan_w_fn
        self.quan_a_fn = quan_a_fn

        self.weight = t.nn.Parameter(m.weight.detach())
        self.quan_w_fn.init_from(m.weight.detach())
        if m.bias is not None:
            self.bias = t.nn.Parameter(m.bias.detach())

        self.use_qsam = use_qsam
        self.qsam_ratio = qsam_ratio
        self.qsam_rho = qsam_rho
        # Runtime gate: train loop turns this off during warmup and during the
        # BN-recalibration pass (so BN stats are gathered on unperturbed q).
        self.qsam_on = True
        if use_qsam:
            self.register_buffer(
                "g_prev", torch.zeros_like(m.weight.detach())
            )

    def save_grad_for_qsam(self):
        if self.use_qsam and self.weight.grad is not None:
            self.g_prev.copy_(self.weight.grad.detach())

    def forward(self, x: Tensor) -> Tensor:
        quantized_weight = self.quan_w_fn(self.weight)
        if self.use_qsam and self.training and self.qsam_on:
            delta_step = _get_w_quant_step(self.quan_w_fn).to(self.weight.device)
            delta = _qsam_perturbation(
                self.weight, self.g_prev, delta_step, self.qsam_ratio, self.qsam_rho
            )
            if delta is not None:
                quantized_weight = quantized_weight + delta
        quantized_act = self.quan_a_fn(x)
        return self._conv_forward(quantized_act, quantized_weight, bias=self.bias)


class QuanLinear(t.nn.Linear):
    def __init__(
        self,
        m: t.nn.Linear,
        quan_w_fn=None,
        quan_a_fn=None,
        use_qsam: bool = False,
        qsam_ratio: float = 0.001,
        qsam_rho: float = 1.0,
    ):
        assert type(m) == t.nn.Linear
        super().__init__(
            m.in_features, m.out_features, bias=(m.bias is not None)
        )
        self.quan_w_fn = quan_w_fn
        self.quan_a_fn = quan_a_fn
        self.weight = t.nn.Parameter(m.weight.detach())
        self.quan_w_fn.init_from(m.weight.detach())
        if m.bias is not None:
            self.bias = t.nn.Parameter(m.bias.detach())

        self.use_qsam = use_qsam
        self.qsam_ratio = qsam_ratio
        self.qsam_rho = qsam_rho
        self.qsam_on = True
        if use_qsam:
            self.register_buffer(
                "g_prev", torch.zeros_like(m.weight.detach())
            )

    def save_grad_for_qsam(self):
        if self.use_qsam and self.weight.grad is not None:
            self.g_prev.copy_(self.weight.grad.detach())

    def forward(self, x: Tensor) -> Tensor:
        quantized_weight = self.quan_w_fn(self.weight)
        if self.use_qsam and self.training and self.qsam_on:
            delta_step = _get_w_quant_step(self.quan_w_fn).to(self.weight.device)
            delta = _qsam_perturbation(
                self.weight, self.g_prev, delta_step, self.qsam_ratio, self.qsam_rho
            )
            if delta is not None:
                quantized_weight = quantized_weight + delta
        quantized_act = self.quan_a_fn(x)
        return F.linear(quantized_act, quantized_weight, bias=self.bias)


QuanModuleMapping = {
    t.nn.Conv2d: QuanConv2d,
    t.nn.Linear: QuanLinear,
}
