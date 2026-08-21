import os
import sys
import time
import random
import logging
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

sys.path.append("..")
from utils.utils import (
    AverageMeter,
    ProgressMeter,
    save_checkpoint,
    accuracy,
)
from torchvision import datasets, transforms
import quan
from globalVal import globalVal
from resnet import resnet18_cifar, resnet50_cifar
from s2_sam import (
    add_sam_perturbation,
    collect_sam_params,
    init_prev_grads,
    restore_sam_perturbation,
    save_current_grads,
)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser("aoq-qsam-cifar100")
parser.add_argument(
    "--method",
    choices=["lsq", "lsq_sam", "lsq_aoq", "lsq_qsam", "lsq_aoq_qsam"],
    required=True,
    help="Which combination to run.",
)
parser.add_argument("--model", choices=["resnet18", "resnet50"], default="resnet18")
parser.add_argument("--n_bit", type=int, default=4)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument(
    "--qsam_ratio",
    type=float,
    default=0.001,
    help="Fraction of weights perturbed per layer per step (qSAM).",
)
parser.add_argument(
    "--sam_rho",
    type=float,
    default=0.05,
    help="S2-SAM perturbation radius for continuous full-precision weights.",
)
parser.add_argument(
    "--bn_recal_batches",
    type=int,
    default=100,
    help="BN recalibration batches on unperturbed q before each validate (qSAM only, 0=off).",
)
parser.add_argument(
    "--lambda_dampen",
    type=float,
    default=1e-3,
    help="Weight of AOQ Eq.(9) dampening loss in Stage 3.",
)
parser.add_argument("--data", default="./data")
parser.add_argument("--save", default="./models")
parser.add_argument("--workers", type=int, default=4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--pretrained",
    type=str,
    default="",
    help="Path to FP CIFAR-100 checkpoint. Empty -> use ImageNet pretrained backbone.",
)
args = parser.parse_args()


USE_AOQ = "aoq" in args.method
USE_QSAM = "qsam" in args.method
USE_SAM = args.method == "lsq_sam"

CLASSES = 100
STAGE1_END = max(args.epochs // 5, 1)
STAGE3_START = max(args.epochs * 3 // 5, STAGE1_END + 1)

RUN_TAG = f"{args.method}_{args.model}_{args.n_bit}bit_seed{args.seed}"
log_dir = f"log/{RUN_TAG}"
os.makedirs(log_dir, exist_ok=True)

log_format = "%(asctime)s %(message)s"
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format=log_format,
    datefmt="%m/%d %I:%M:%S %p",
)
fh = logging.FileHandler(os.path.join(log_dir, "log.txt"))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

device = torch.device(globalVal.device)
SAM_PARAMS = []
SAM_PREV_GRADS = []


def main():
    setup_seed(args.seed)
    if not torch.cuda.is_available():
        sys.exit("CUDA required")
    cudnn.benchmark = True
    logging.info("args = %s", args)
    logging.info(
        "method=%s | use_aoq=%s | use_qsam=%s | use_sam=%s | stages: [0,%d)->[%d,%d)->[%d,%d)",
        args.method, USE_AOQ, USE_QSAM, USE_SAM,
        STAGE1_END, STAGE1_END, STAGE3_START, STAGE3_START, args.epochs,
    )

    # Backbone init: ImageNet pretrained layer1~4 unless a CIFAR-100 FP ckpt given.
    model_fn = resnet50_cifar if args.model == "resnet50" else resnet18_cifar
    model = model_fn(
        num_classes=CLASSES, pretrained_imagenet=(args.pretrained == "")
    )
    if args.pretrained:
        ckpt = torch.load(args.pretrained, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        logging.info(
            "loaded FP ckpt %s | missing=%d unexpected=%d",
            args.pretrained, len(missing), len(unexpected),
        )

    weight_quantizer = "aoq" if USE_AOQ else "lsq"
    modules_to_replace = quan.find_modules_to_quantize(
        model,
        args.n_bit,
        weight_quantizer=weight_quantizer,
        stage1_end_epoch=STAGE1_END,
        use_qsam=USE_QSAM,
        qsam_ratio=args.qsam_ratio,
    )
    model = quan.replace_module_by_names(model, modules_to_replace)
    model = model.to(device)
    logging.info("model:\n%s", model)

    quan_modules = [
        m for m in model.modules()
        if isinstance(m, (quan.QuanConv2d, quan.QuanLinear)) and getattr(m, "use_qsam", False)
    ]
    logging.info("qSAM: %s | tracked modules: %d", USE_QSAM, len(quan_modules))

    global SAM_PARAMS, SAM_PREV_GRADS
    if USE_SAM:
        SAM_PARAMS = collect_sam_params(model, (quan.QuanConv2d, quan.QuanLinear))
        SAM_PREV_GRADS = init_prev_grads(SAM_PARAMS)
    logging.info(
        "S2-SAM: %s | tracked params: %d | rho=%.4f",
        USE_SAM, len(SAM_PARAMS), args.sam_rho,
    )

    criterion = nn.CrossEntropyLoss().to(device)

    weight_params, other_params = [], []
    for pname, p in model.named_parameters():
        if p.ndimension() == 4 and "bias" not in pname:
            weight_params.append(p)
        else:
            other_params.append(p)
    logging.info(
        "weight params=%d | other params=%d", len(weight_params), len(other_params)
    )

    optimizer = torch.optim.Adam(
        [
            {"params": other_params, "lr": args.learning_rate},
            {
                "params": weight_params,
                "weight_decay": args.weight_decay,
                "lr": args.learning_rate,
            },
        ],
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: max(1.0 - step / args.epochs, 0.0)
    )

    # CIFAR-100 data
    normalize = transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408],
        std=[0.2675, 0.2565, 0.2761],
    )
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_tf = transforms.Compose([transforms.ToTensor(), normalize])

    from cifar_local import CIFAR100Local
    train_set = CIFAR100Local(args.data, train=True, transform=train_tf)
    val_set = CIFAR100Local(args.data, train=False, transform=val_tf)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    save_dir = os.path.join(args.save, RUN_TAG)
    os.makedirs(save_dir, exist_ok=True)

    # Resume from checkpoint if present (so a stopped run continues, not restarts).
    start_epoch = 0
    best_acc = 0.0
    checkpoint_tar = os.path.join(save_dir, "checkpoint.pth.tar")
    if os.path.exists(checkpoint_tar):
        ckpt = torch.load(checkpoint_tar, map_location=device)
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt["best_top1_acc"]
        model.load_state_dict(ckpt["state_dict"], strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        logging.info("resumed from epoch %d, best=%.3f", ckpt["epoch"], best_acc)
    # LambdaLR is deterministic in step count: fast-forward to match a fresh run
    # (train_one_epoch steps it once more at the start of each epoch).
    for _ in range(start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, args.epochs):
        globalVal.epoch = float(epoch)
        train_one_epoch(epoch, train_loader, model, criterion, optimizer, scheduler)
        # qSAM: recompute BN stats on unperturbed q so eval matches deployed model.
        if USE_QSAM and args.bn_recal_batches > 0:
            recalibrate_bn(model, train_loader, quan_modules, args.bn_recal_batches)
        acc1 = validate(epoch, val_loader, model, criterion)
        is_best = acc1 > best_acc
        if is_best:
            best_acc = acc1
        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_top1_acc": best_acc,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            save_dir,
        )
        logging.info("epoch %d | top1=%.3f | best=%.3f", epoch, acc1, best_acc)
    logging.info("done. best top1 = %.3f", best_acc)


def _stage_of(epoch):
    if epoch < STAGE1_END:
        return 1
    if epoch < STAGE3_START:
        return 2
    return 3


def train_one_epoch(epoch, loader, model, criterion, optimizer, scheduler):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(loader), [batch_time, losses, top1, top5], prefix=f"Epoch:[{epoch}]"
    )

    model.train()
    scheduler.step()
    for pg in optimizer.param_groups:
        cur_lr = pg["lr"]
    stage = _stage_of(epoch)
    logging.info("epoch %d lr=%.5f stage=%d", epoch, cur_lr, stage)

    end = time.time()
    for i, (img, tgt) in enumerate(loader):
        img = img.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        sam_perturbations = []
        if USE_SAM:
            sam_perturbations = add_sam_perturbation(
                SAM_PARAMS, SAM_PREV_GRADS, args.sam_rho
            )

        globalVal.loss = 0.0
        logits = model(img)
        loss_task = criterion(logits, tgt)
        if USE_AOQ and stage == 3:
            loss = loss_task + args.lambda_dampen * globalVal.loss
        else:
            loss = loss_task
        globalVal.loss = 0.0

        prec1, prec5 = accuracy(logits, tgt, topk=(1, 5))
        n = img.size(0)
        losses.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        optimizer.zero_grad()
        loss.backward()
        if sam_perturbations:
            restore_sam_perturbation(sam_perturbations)
        optimizer.step()

        # qSAM: snapshot the just-computed weight grads as g_prev for the *next* step.
        if USE_QSAM:
            for m in model.modules():
                if hasattr(m, "save_grad_for_qsam"):
                    m.save_grad_for_qsam()
        if USE_SAM:
            save_current_grads(SAM_PARAMS, SAM_PREV_GRADS)

        batch_time.update(time.time() - end)
        end = time.time()
        if i % 50 == 0:
            progress.display(i)

    return losses.avg, top1.avg


def recalibrate_bn(model, loader, quan_modules, num_batches):
    """Recompute BN running stats on the unperturbed quantized model (qSAM off)."""
    bn_layers = [m for m in model.modules()
                 if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bn_layers or num_batches <= 0:
        return
    saved_on = [m.qsam_on for m in quan_modules]
    for m in quan_modules:
        m.qsam_on = False
    saved_mom = []
    for bn in bn_layers:
        bn.reset_running_stats()
        saved_mom.append(bn.momentum)
        bn.momentum = None
    model.train()
    seen = 0
    with torch.no_grad():
        for img, _ in loader:
            if seen >= num_batches:
                break
            model(img.to(device, non_blocking=True))
            seen += 1
    for bn, mom in zip(bn_layers, saved_mom):
        bn.momentum = mom
    for m, s in zip(quan_modules, saved_on):
        m.qsam_on = s


def validate(epoch, loader, model, criterion):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(loader), [batch_time, losses, top1, top5], prefix="Test:"
    )
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (img, tgt) in enumerate(loader):
            img = img.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            logits = model(img)
            loss = criterion(logits, tgt)
            prec1, prec5 = accuracy(logits, tgt, topk=(1, 5))
            n = img.size(0)
            losses.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
            batch_time.update(time.time() - end)
            end = time.time()
            if i % 50 == 0:
                progress.display(i)
    logging.info("Val: top1=%.3f top5=%.3f", top1.avg, top5.avg)
    return top1.avg


if __name__ == "__main__":
    main()
