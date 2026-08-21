from .func import *
from .quantizer import *
import torch.nn as nn


def find_modules_to_quantize(
    model,
    n_bit,
    weight_quantizer="lsq",
    stage1_end_epoch=30,
    use_qsam=False,
    qsam_ratio=0.001,
    qsam_rho=1.0,
    first_conv_name="conv1",
    last_fc_name="fc",
    first_last_n_bit=0,
    quantize_downsample=True,
):
    """Wrap conv/linear layers with quantized versions.

    Default policy (LSQ-style, `first_last_n_bit=0`): the first conv (`conv1`)
    and the final fc Linear stay full-precision; all other Conv2d are quantized
    to `n_bit`, and qSAM perturbs only those convs.

    LSQ-paper policy (`first_last_n_bit=8`): the first conv and last fc are
    additionally quantized, but at `first_last_n_bit` bits (the paper keeps
    first/last at 8-bit while the rest are 2/3/4-bit). The first conv's input is
    the (signed) image, so it uses a signed activation quantizer; the fc input is
    post-ReLU (avgpool of layer4), so it uses the usual unsigned quantizer.

    weight_quantizer: 'lsq' or 'aoq' (applies to the main n_bit convs)
    use_qsam:        if True, add the single-step S2-SAM perturbation in forward.
    qsam_ratio:      fraction K/d of top-|g| weight coords to shift per step.
    qsam_rho:        scale rho<=1 applied to the one-level shift (rho*Delta).
    first_last_n_bit: 0 -> first conv & fc full-precision; >0 -> quantize them
                      at this bit-width (LSQ paper uses 8). first/last always use
                      the LSQ quantizer regardless of weight_quantizer.
    quantize_downsample: False leaves shortcut/downsample Conv2d layers full-precision.
    """
    def make_w_fn(bits, force_lsq=False):
        if force_lsq or weight_quantizer == "lsq":
            return LSQ(num_bits=bits, quantized_object="weight")
        elif weight_quantizer == "aoq":
            return AOQ(num_bits=bits, quantized_object="weight",
                       stage1_end_epoch=stage1_end_epoch)
        raise ValueError(f"unknown weight_quantizer: {weight_quantizer}")

    replaced_modules = dict()
    for name, module in model.named_modules():
        is_first = name == first_conv_name
        is_last = name == last_fc_name

        if isinstance(module, nn.Conv2d):
            if not quantize_downsample and ".downsample." in name:
                continue
            if is_first:
                if first_last_n_bit <= 0:
                    continue  # first conv stays full-precision
                w_fn = make_w_fn(first_last_n_bit, force_lsq=True)
                # first-conv input is the signed image -> signed activation
                a_fn = LSQ(num_bits=first_last_n_bit, quantized_object="act_signed")
                uq, qr = False, qsam_ratio  # never perturb the first conv
            else:
                w_fn = make_w_fn(n_bit)
                a_fn = LSQ(num_bits=n_bit, quantized_object="act")
                uq, qr = use_qsam, qsam_ratio
            replaced_modules[name] = QuanModuleMapping[type(module)](
                module, quan_w_fn=w_fn, quan_a_fn=a_fn,
                use_qsam=uq, qsam_ratio=qr, qsam_rho=qsam_rho,
            )
        elif isinstance(module, nn.Linear) and is_last and first_last_n_bit > 0:
            # last fc: quantize at first_last_n_bit; input is post-ReLU -> unsigned
            w_fn = make_w_fn(first_last_n_bit, force_lsq=True)
            a_fn = LSQ(num_bits=first_last_n_bit, quantized_object="act")
            replaced_modules[name] = QuanModuleMapping[type(module)](
                module, quan_w_fn=w_fn, quan_a_fn=a_fn,
                use_qsam=False, qsam_ratio=qsam_ratio, qsam_rho=qsam_rho,
            )
    return replaced_modules


def replace_module_by_names(model, modules_to_replace):
    def helper(child: t.nn.Module):
        for n, c in child.named_children():
            if type(c) in QuanModuleMapping.keys():
                for full_name, m in model.named_modules():
                    if c is m and full_name in modules_to_replace.keys():
                        child.add_module(n, modules_to_replace.pop(full_name))
                        break
            else:
                helper(c)

    helper(model)
    return model
