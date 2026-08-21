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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append("..")
from utils.utils import (
    AverageMeter,
    ProgressMeter,
    CrossEntropyLabelSmooth,
    save_checkpoint,
    accuracy,
)
from utils import KD_loss
from torchvision import datasets, transforms
import torchvision.models as models
import quan
from globalVal import globalVal


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser("aoq-imagenet")
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--epochs", type=int, default=150)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--save", type=str, default="./models")
parser.add_argument("--data", metavar="DIR", required=True)
parser.add_argument("--label_smooth", type=float, default=0.1)
parser.add_argument("--teacher", type=str, default="resnet101")
parser.add_argument(
    "--use_kd",
    type=str,
    default="False",
    help="knowledge distillation from a frozen full-precision teacher (off saves GPU memory)",
)
parser.add_argument("--student", type=str, default="resnet50")
parser.add_argument("--n_bit", type=int, default=4)
parser.add_argument(
    "--optimizer", type=str, default="adam", choices=["adam", "sgd"],
    help="adam (current harness) or sgd momentum (LSQ paper)",
)
parser.add_argument(
    "--lr_scheduler", type=str, default="linear", choices=["linear", "cosine"],
    help="linear decay (current harness) or cosine decay without restarts (LSQ paper)",
)
parser.add_argument(
    "--first_last_n_bit", type=int, default=0,
    help="0: first conv & fc full-precision (default). >0: quantize them at this "
         "bit-width (LSQ paper uses 8 for first/last).",
)
parser.add_argument(
    "--use_qsam",
    type=str,
    default="False",
    help="enable single-step quantized SAM (S2-SAM efficiency + RA-qSAM p=2 geometry)",
)
parser.add_argument(
    "--qsam_ratio",
    type=float,
    default=0.001,
    help="fraction K/d of top-|g| weight coords shifted by quantized SAM",
)
parser.add_argument(
    "--qsam_rho",
    type=float,
    default=1.0,
    help="level-shift scale: delta = rho*Delta*sign(g_prev); rho=1 is one level",
)
parser.add_argument(
    "--qsam_warmup_epochs",
    type=int,
    default=0,
    help="run pure LSQ for this many epochs before turning qSAM on",
)
parser.add_argument(
    "--bn_recal_batches",
    type=int,
    default=200,
    help="batches for BN recalibration on unperturbed q before each validate (0=off)",
)
parser.add_argument("--quantize_downsample", type=str, default="True")
parser.add_argument("-j", "--workers", default=8, type=int)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

CLASSES = 1000

if not os.path.exists("log"):
    os.makedirs("log", exist_ok=True)

log_format = "%(asctime)s %(message)s"
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format=log_format,
    datefmt="%m/%d %I:%M:%S %p",
)
fh = logging.FileHandler(os.path.join("log/log.txt"))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

device = torch.device(globalVal.device)


def main():
    setup_seed(args.seed)
    if not torch.cuda.is_available():
        sys.exit("CUDA not available")
    start_t = time.time()

    cudnn.benchmark = True
    cudnn.enabled = True
    logging.info("args = %s", args)

    use_kd = args.use_kd in ("True", "true", "1", 1, True)

    # Teacher (full-precision, frozen) -- only built when KD is enabled, so the
    # no-KD baseline avoids the extra teacher model + forward (less GPU memory).
    model_teacher = None
    if use_kd:
        model_teacher = models.__dict__[args.teacher](weights="IMAGENET1K_V1")
        model_teacher = model_teacher.to(device)
        for p in model_teacher.parameters():
            p.requires_grad = False
        model_teacher.eval()

    quantize_downsample = args.quantize_downsample in ("True", "true", "1", 1, True)

    use_qsam = args.use_qsam in ("True", "true", "1", 1, True)

    # Student initialized from full-precision pretrained, then wrap convs with a
    # faithful LSQ weight/act quantizer (LSQ baseline). qSAM, when enabled, adds
    # the single-step S2-SAM perturbation inside each quantized conv's forward.
    model_student = models.__dict__[args.student](weights="IMAGENET1K_V1")
    modules_to_replace = quan.find_modules_to_quantize(
        model_student,
        args.n_bit,
        weight_quantizer="lsq",
        use_qsam=use_qsam,
        qsam_ratio=args.qsam_ratio,
        qsam_rho=args.qsam_rho,
        first_last_n_bit=args.first_last_n_bit,
        quantize_downsample=quantize_downsample,
    )
    model_student = quan.replace_module_by_names(model_student, modules_to_replace)
    model_student = model_student.to(device)
    logging.info("student model:\n%s", model_student)

    # Quantized modules that own a g_prev buffer; we refresh it each step (S2-SAM).
    quan_modules = [
        m for m in model_student.modules()
        if isinstance(m, (quan.QuanConv2d, quan.QuanLinear)) and m.use_qsam
    ]
    logging.info("qSAM: %s | quantized modules tracked: %d",
                 use_qsam, len(quan_modules))

    criterion = nn.CrossEntropyLoss().to(device)
    criterion_kd = KD_loss.DistributionLoss() if use_kd else None
    logging.info("KD: %s (teacher=%s)", use_kd, args.teacher if use_kd else "none")

    # Parameter groups: weight params vs others (step_size, BN, FC, etc.).
    weight_parameters, other_parameters = [], []
    for pname, p in model_student.named_parameters():
        if p.ndimension() == 4 and "bias" not in pname:
            weight_parameters.append(p)
        else:
            other_parameters.append(p)
    logging.info(
        "weight params: %d  |  other params: %d",
        len(weight_parameters), len(other_parameters),
    )

    param_groups = [
        {"params": other_parameters, "lr": args.learning_rate},
        {
            "params": weight_parameters,
            "weight_decay": args.weight_decay,
            "lr": args.learning_rate,
        },
    ]
    if args.optimizer == "sgd":
        # LSQ paper: SGD momentum 0.9.
        optimizer = torch.optim.SGD(param_groups, momentum=args.momentum)
    else:
        optimizer = torch.optim.Adam(param_groups, betas=(0.9, 0.999))
    logging.info("optimizer: %s", args.optimizer)
    if args.lr_scheduler == "cosine":
        # LSQ paper: cosine decay without restarts.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
    else:
        # Linear LR decay.
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: max(1.0 - step / args.epochs, 0.0), last_epoch=-1
        )
    logging.info("lr_scheduler: %s", args.lr_scheduler)

    run_tag = (
        f"{args.student}_{args.n_bit}bit_qd_{quantize_downsample}_fl{args.first_last_n_bit}"
        f"_opt_{args.optimizer}_sched_{args.lr_scheduler}_lr{args.learning_rate:g}"
        f"_ep{args.epochs}_wd{args.weight_decay:g}_qsam_{use_qsam}_kd_{use_kd}"
    )
    if use_qsam:
        run_tag += f"_w{args.qsam_warmup_epochs}_r{args.qsam_ratio:g}_rho{args.qsam_rho:g}"
    run_tag += f"_seed{args.seed}"
    save_dir = os.path.join(args.save, run_tag)
    os.makedirs(save_dir, exist_ok=True)

    start_epoch = 0
    best_top1_acc = 0
    checkpoint_tar = os.path.join(save_dir, "checkpoint.pth.tar")
    loaded_scheduler = False
    if os.path.exists(checkpoint_tar):
        logging.info("loading checkpoint %s", checkpoint_tar)
        checkpoint = torch.load(checkpoint_tar, map_location=device)
        start_epoch = checkpoint["epoch"] + 1
        best_top1_acc = checkpoint["best_top1_acc"]
        model_student.load_state_dict(checkpoint["state_dict"], strict=False)
        # Restore optimizer (Adam moments) so resume is continuous, not a restart
        # of the moment estimates. Guarded for older checkpoints without it.
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
            loaded_scheduler = True
        logging.info(
            "resumed: epoch %d, best_top1 = %.3f", checkpoint["epoch"], best_top1_acc
        )

    for _ in range(0 if loaded_scheduler else start_epoch):
        scheduler.step()

    # Data
    traindir = os.path.join(args.data, "train")
    valdir = os.path.join(args.data, "val")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    train_dataset = datasets.ImageFolder(traindir, transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            valdir,
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    if use_kd:
        print("Teacher accuracy:")
        validate(-2, val_loader, model_teacher, criterion, args)

    epoch = start_epoch
    while epoch < args.epochs:
        globalVal.epoch = float(epoch)

        # Warmup: keep qSAM off until step sizes / act quantizers stabilize.
        qsam_active = use_qsam and epoch >= args.qsam_warmup_epochs
        for m in quan_modules:
            m.qsam_on = qsam_active
        if use_qsam:
            logging.info(
                "epoch %d  qSAM %s", epoch,
                "ON" if qsam_active else f"off (warmup<{args.qsam_warmup_epochs})",
            )

        train(epoch, train_loader, model_student, model_teacher,
              criterion_kd, criterion, optimizer, quan_modules, use_kd)
        scheduler.step()

        # BN running stats were updated on the perturbed q+delta during training;
        # recompute them on the unperturbed q so eval matches the deployed model.
        if qsam_active and args.bn_recal_batches > 0:
            recalibrate_bn(model_student, train_loader, quan_modules,
                           args.bn_recal_batches)

        _, valid_top1_acc, _ = validate(epoch, val_loader, model_student, criterion, args)

        is_best = valid_top1_acc > best_top1_acc
        if is_best:
            best_top1_acc = valid_top1_acc

        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model_student.state_dict(),
                "best_top1_acc": best_top1_acc,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            is_best,
            save_dir,
        )
        logging.info("epoch %d done | top1=%.3f | best=%.3f",
                     epoch, valid_top1_acc, best_top1_acc)
        epoch += 1

    print(f"total training time = {(time.time() - start_t) / 3600:.2f} hours")


def train(epoch, train_loader, model_student, model_teacher,
          criterion_kd, criterion_ce, optimizer, quan_modules, use_kd):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix=f"Epoch: [{epoch}]",
    )

    model_student.train()
    if use_kd:
        model_teacher.eval()
    for param_group in optimizer.param_groups:
        cur_lr = param_group["lr"]
    logging.info("epoch %d  lr=%.5f", epoch, cur_lr)

    end = time.time()
    epoch_start = time.time()
    total_iters = len(train_loader)
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # Forward uses the perturbed quantized weights built from the PRIOR
        # step's gradient (S2-SAM); for the first step g_prev=0 so no perturbation.
        logits_student = model_student(images)
        if use_kd:
            with torch.no_grad():
                logits_teacher = model_teacher(images)
            loss = criterion_kd(logits_student, logits_teacher)
        else:
            loss = criterion_ce(logits_student, target)

        prec1, prec5 = accuracy(logits_student, target, topk=(1, 5))
        n = images.size(0)
        losses.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        optimizer.zero_grad()
        loss.backward()
        # S2-SAM: cache the gradient used for THIS update (the STE proxy of g_q)
        # BEFORE the optimizer step, to build next step's perturbation. Single
        # backward, zero extra cost.
        for m in quan_modules:
            m.save_grad_for_qsam()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        if i % 50 == 0:
            done = i + 1
            frac = done / total_iters
            elapsed = time.time() - epoch_start
            eta = (total_iters - done) * batch_time.avg
            nb = 30
            fill = int(frac * nb)
            bar = "#" * fill + "-" * (nb - fill)
            # Emitted via logging (flushes every record) so it shows live through
            # the tee pipe, unlike print() which block-buffers.
            logging.info(
                "epoch %d [%s] %d/%d %3.0f%% | %.0fms/it | loss %.3f acc@1 %.2f | elapsed %.1fm ETA %.1fm",
                epoch, bar, done, total_iters, frac * 100,
                batch_time.avg * 1000, losses.avg, top1.avg,
                elapsed / 60.0, eta / 60.0,
            )

    return losses.avg, top1.avg, top5.avg


def recalibrate_bn(model, loader, quan_modules, num_batches):
    """Recompute BatchNorm running stats on the *unperturbed* quantized model.

    During qSAM training every forward uses q+delta, so BN running mean/var drift
    toward the perturbed model; but eval uses q. We disable the perturbation and
    re-estimate BN stats on q (cumulative average over `num_batches`) so the
    evaluated model's BN matches what it actually runs.
    """
    bn_layers = [
        m for m in model.modules()
        if isinstance(m, nn.modules.batchnorm._BatchNorm)
    ]
    if not bn_layers or num_batches <= 0:
        return
    saved_on = [m.qsam_on for m in quan_modules]
    for m in quan_modules:
        m.qsam_on = False
    saved_mom = []
    for bn in bn_layers:
        bn.reset_running_stats()
        saved_mom.append(bn.momentum)
        bn.momentum = None  # cumulative moving average over the recal batches
    model.train()
    seen = 0
    with torch.no_grad():
        for images, _ in loader:
            if seen >= num_batches:
                break
            model(images.to(device, non_blocking=True))
            seen += 1
    for bn, mom in zip(bn_layers, saved_mom):
        bn.momentum = mom
    for m, s in zip(quan_modules, saved_on):
        m.qsam_on = s
    logging.info("BN recalibrated on unperturbed q over %d batches", seen)


def validate(epoch, val_loader, model, criterion, args):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, target)
            pred1, pred5 = accuracy(logits, target, topk=(1, 5))
            n = images.size(0)
            losses.update(loss.item(), n)
            top1.update(pred1[0], n)
            top5.update(pred5[0], n)
            batch_time.update(time.time() - end)
            end = time.time()
            if i % 50 == 0:
                progress.display(i)
        print(f" * acc@1 {top1.avg:.3f} acc@5 {top5.avg:.3f}")
    return losses.avg, top1.avg, top5.avg


if __name__ == "__main__":
    main()
