import torch
import torch.nn as nn
import sys
import numpy as np

sys.path.append("..")
from globalVal import globalVal


device = torch.device(globalVal.device)


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad


def grad_scale(x, scale):
    """LSQ step-size gradient scale (Esser et al., 2020, Sec 2.2 / Function 1).
    Forward returns x unchanged; backward multiplies the gradient by `scale`.
    """
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


class LTQ(nn.Module):
    """Learnable threshold quantizer (kept for backward compatibility)."""

    def __init__(self, num_bits, quantized_object):
        super(LTQ, self).__init__()
        init_range = 2.0
        self.num_bits = num_bits
        self.register_buffer("n_val", torch.tensor([2**num_bits - 1]))
        self.register_buffer("interval", torch.tensor([init_range / self.n_val]))
        self.register_buffer("zero", torch.tensor([0.0]))
        self.register_buffer("two", torch.tensor([2.0]))
        self.quantized_object = quantized_object
        self.eps = nn.Parameter(torch.tensor([1e-6]), requires_grad=False)
        self.start = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.register_buffer("start_detach", torch.tensor([0.0]))
        self.a = nn.Parameter(torch.tensor([self.interval] * self.n_val), requires_grad=True)
        self.scale1 = nn.Parameter(torch.tensor([1.0]))
        self.scale2 = nn.Parameter(torch.tensor([2.0]))

    def init_from(self, x):
        pass

    def forward(self, x):
        if self.quantized_object != "act":
            return x
        x = x * self.scale1
        self.start_detach = self.start.clone().detach()
        a_pos = torch.where(self.a > self.eps, self.a, self.eps)
        step_right = self.zero + 0.0
        thre_forward = self.start + a_pos[0] / 2
        thre_backward = self.start + 0.0
        for i in range(self.n_val):
            step_right = step_right + self.interval
            if i == 0:
                x_forward = torch.where(x > thre_forward, step_right, self.zero)
                x_backward = torch.where(
                    x > thre_backward,
                    self.interval / a_pos[i] * (x - thre_backward) + step_right - self.interval,
                    self.zero,
                )
            else:
                thre_forward = thre_forward + a_pos[i - 1] / 2 + a_pos[i] / 2
                thre_backward = thre_backward + a_pos[i - 1]
                x_forward = torch.where(x > thre_forward, step_right, x_forward)
                x_backward = torch.where(
                    x > thre_backward,
                    self.interval / a_pos[i] * (x - thre_backward) + step_right - self.interval,
                    x_backward,
                )
        thre_backward = thre_backward + a_pos[i]
        x_backward = torch.where(x > thre_backward, self.two, x_backward)
        out = x_forward.detach() + x_backward - x_backward.detach()
        return out * self.scale2


class LSQ(nn.Module):
    """LSQ quantizer for weight or activation. Supports arbitrary bit-width.

    Weight:      signed, range [-2^(b-1), 2^(b-1)-1]
    Activation:  unsigned, range [0, 2^b - 1] (assumes ReLU input)
    act_signed:  signed activation, range [-2^(b-1), 2^(b-1)-1]
                 (e.g. the first-layer input image, which is not post-ReLU)
    """

    def __init__(self, num_bits, quantized_object):
        super(LSQ, self).__init__()
        self.num_bits = num_bits
        self.quantized_object = quantized_object
        self.step_size = nn.Parameter(torch.tensor([1.0]))
        if quantized_object == "act":
            self.thd_neg = 0
            self.thd_pos = 2**num_bits - 1
        elif quantized_object in ("weight", "act_signed"):
            self.thd_neg = -(2 ** (num_bits - 1))
            self.thd_pos = 2 ** (num_bits - 1) - 1
        else:
            raise ValueError(f"unknown quantized_object: {quantized_object}")
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    def init_from(self, x):
        denom = max(self.thd_pos, 1) ** 0.5
        s_init = x.detach().abs().mean() * 2.0 / denom
        s_init = s_init.clamp(min=1e-8)
        self.step_size.data = s_init.view(1).to(self.step_size.device)
        self.initialized.fill_(1)

    def forward(self, x):
        if self.initialized.item() == 0:
            self.init_from(x)
        # Step-size gradient scale g = 1/sqrt(N * Qp) (LSQ Sec 2.2). For weights
        # N = #weights in the layer; for activations N = #features per sample
        # (NF, batch-independent, per the paper). Qp = thd_pos.
        qp = max(self.thd_pos, 1)
        if self.quantized_object in ("act", "act_signed"):
            n = x.numel() / x.shape[0] if x.dim() > 0 and x.shape[0] > 0 else x.numel()
        else:
            n = x.numel()
        g = 1.0 / ((float(n) * qp) ** 0.5)
        s = grad_scale(self.step_size, g)
        x_q = torch.clamp(x / s, self.thd_neg, self.thd_pos)
        x_q = round_pass(x_q)
        return x_q * s


class AOQ(nn.Module):
    """Allowing Oscillation Quantization for weights. Arbitrary bit-width.

    Stage 1 (epoch <= stage1_end): alpha shrinks via cosine from 1.0 -> 0.30,
        levels computed from fixed init_interval (s_th == s_le).
    Stage 2 (stage1_end < epoch < stage3_start): alpha frozen,
        levels computed from learnable step_size (s_le), thresholds stay on
        init_interval (s_th decoupled).
    Stage 3 (epoch >= stage3_start): same as Stage 2, plus dampening loss
        accumulated into globalVal.loss.
    """

    def __init__(self, num_bits, quantized_object="weight", stage1_end_epoch=30):
        super(AOQ, self).__init__()
        assert quantized_object == "weight", "AOQ is for weight quantization only"
        self.num_bits = num_bits
        self.num_levels = 2**num_bits
        self.stage1_end_epoch = float(max(stage1_end_epoch, 1))

        center = (self.num_levels - 1) / 2.0
        offsets = torch.arange(self.num_levels, dtype=torch.float32) - center
        self.register_buffer("level_offsets", offsets)
        self.register_buffer("thresh_offsets", offsets[:-1] + 0.5)

        self.step_size = nn.Parameter(torch.tensor([1.0]))
        self.init_interval = nn.Parameter(torch.tensor([1.0]), requires_grad=False)
        self.alpha = nn.Parameter(torch.tensor([1.0]), requires_grad=False)

    def init_from(self, x):
        s = torch.std(x.detach()) / (2 ** (self.num_bits - 2))
        s = s.clamp(min=1e-8)
        self.step_size.data = s.view(1).to(self.step_size.device)
        self.init_interval.data = s.view(1).to(self.init_interval.device)

    def forward(self, x):
        e = float(globalVal.epoch)
        s1 = self.stage1_end_epoch

        # Stage 1: cosine schedule -- alpha goes 1.0 (epoch 0) -> 0.30 (epoch s1)
        if e <= s1:
            new_alpha = 0.35 * float(np.cos(np.pi * e / s1)) + 0.65
            self.alpha.data.fill_(new_alpha)

        # Stage 1 uses fixed init_interval for level spacing (s_th == s_le);
        # Stage 2/3 lets s_le grow via learned step_size while s_th stays on init_interval.
        scale = self.init_interval if e <= s1 else self.step_size
        alpha = self.alpha

        levels = self.level_offsets * scale * alpha
        thresholds = self.thresh_offsets * self.init_interval * alpha
        cluster = self.level_offsets * self.init_interval * alpha

        # Eq. (6): round to nearest level via threshold comparisons
        x_forward = levels[0].expand_as(x)
        x_cluster = cluster[0].expand_as(x)
        for i in range(self.num_levels - 1):
            mask = x >= thresholds[i]
            x_forward = torch.where(mask, levels[i + 1].expand_as(x), x_forward)
            x_cluster = torch.where(mask, cluster[i + 1].expand_as(x), x_cluster)

        # STE: forward = x_forward, backward = identity through x
        out = x + (x_forward - x).detach()

        # Eq. (9) dampening: ||quantized - clip(w, Qn, Qp)||_F^2
        lvl_min = levels[0].detach()
        lvl_max = levels[-1].detach()
        w_clipped = torch.clamp(x, lvl_min, lvl_max)
        dampen = ((x_cluster.detach() - w_clipped) ** 2).sum()
        globalVal.loss = globalVal.loss + dampen

        return out
