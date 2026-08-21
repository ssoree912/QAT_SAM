import torch


def collect_sam_params(model, module_types):
    params = []
    for module in model.modules():
        if isinstance(module, module_types) and module.weight.requires_grad:
            params.append(module.weight)
    return params


def init_prev_grads(params):
    return [torch.zeros_like(param.detach()) for param in params]


def add_sam_perturbation(params, prev_grads, rho):
    norm_sq = None
    for grad in prev_grads:
        term = torch.sum(grad.detach() * grad.detach())
        norm_sq = term if norm_sq is None else norm_sq + term
    if norm_sq is None:
        return []

    norm = torch.sqrt(norm_sq)
    if norm.item() == 0.0:
        return []

    scale = rho / (norm + 1e-12)
    perturbations = []
    with torch.no_grad():
        for param, grad in zip(params, prev_grads):
            delta = grad.detach() * scale
            param.add_(delta)
            perturbations.append((param, delta))
    return perturbations


def restore_sam_perturbation(perturbations):
    with torch.no_grad():
        for param, delta in perturbations:
            param.sub_(delta)


def save_current_grads(params, prev_grads):
    for param, prev_grad in zip(params, prev_grads):
        if param.grad is not None:
            prev_grad.copy_(param.grad.detach())
