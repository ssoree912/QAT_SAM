import argparse
import datetime
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from google_vit_support import (
    CheckpointPayload,
    METHOD_TO_TRAIN_MODE,
    VARIANTS,
    ResumeTarget,
    build_loader,
    build_model,
    checkpoint_source,
    evaluate,
    fast_forward_scheduler,
    load_pretrained_checkpoint,
    load_training_checkpoint,
    resolve_resume_path,
    save_training_checkpoint,
)
from qsam import enable_qsam, set_qsam_on
from qsam_training import QsamTrainMode, TrainControls, TrainRuntime, train_one_epoch


def parse_args():
    parser = argparse.ArgumentParser("Google ViT QViT CIFAR100/ImageNet")
    parser.add_argument("--method", choices=sorted(METHOD_TO_TRAIN_MODE), default="qvit")
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="B_32")
    parser.add_argument("--nbits", type=int, default=4)
    parser.add_argument("--data-set", choices=["CIFAR100", "IMNET"], default="CIFAR100")
    parser.add_argument("--data-path", default="/workspace/AOQ-main/AO_QAT/data")
    parser.add_argument("--download", action="store_true", help="download CIFAR100 when data-set=CIFAR100")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--finetune", default="auto")
    parser.add_argument("--no-finetune", action="store_true")
    parser.add_argument(
        "--resume",
        default="auto",
        help="checkpoint path to resume from; 'auto' uses output-dir/checkpoint.pth when present",
    )
    parser.add_argument("--fresh-start", action="store_true", help="ignore any resume checkpoint")
    parser.add_argument("--qsam-ratio", type=float, default=1e-3)
    parser.add_argument("--qsam-rho", type=float, default=1.0)
    parser.add_argument("--qsam-warmup-epochs", type=int, default=2)
    parser.add_argument("--sam-rho", type=float, default=0.05)
    parser.add_argument("--debug-train-batches", type=int, default=0)
    parser.add_argument("--debug-val-batches", type=int, default=0)
    parser.add_argument("--print-freq", type=int, default=20)
    parser.add_argument("--wandb", action="store_true", help="log loss/acc to Weights & Biases")
    parser.add_argument("--wandb-project", default="aoq-vit-qat")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    print(args)
    if args.wandb:
        import wandb

        # Default the run name to the output_dir basename so wandb runs line up with runs/.
        run_name = args.wandb_run_name or (output_dir.name if output_dir is not None else None)
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=run_name, config=vars(args))
    print(f"[google-vit] source repo: {repo_root / 'third_party' / 'vision_transformer'}")
    train_mode = METHOD_TO_TRAIN_MODE[args.method]
    use_qsam = train_mode is QsamTrainMode.TWO_PASS
    use_sam = train_mode is QsamTrainMode.SAM
    print(
        f"[qat] method=QViT nbits={args.nbits} "
        f"qsam={use_qsam} sam={use_sam} loop={train_mode.value}"
    )
    resume_path = resolve_resume_path(args, output_dir)
    train_loader = build_loader(args, is_train=True)
    val_loader = build_loader(args, is_train=False)
    model = build_model(args)
    if resume_path is None:
        load_pretrained_checkpoint(model, checkpoint_source(args))
    else:
        print(f"[finetune] skipped because resume is active: {resume_path}")
    model.to(device)
    if use_qsam:
        enabled = enable_qsam(model, args.qsam_ratio, args.qsam_rho, warmup_epochs=args.qsam_warmup_epochs)
        print(
            f"[qSAM] enabled={len(enabled)} ratio={args.qsam_ratio} "
            f"rho={args.qsam_rho} loop={train_mode.value}"
        )
    if use_sam:
        print(f"[SAM] rho={args.sam_rho} targets=quantized_weight_layers")
    effective_lr = args.lr * args.batch_size / 512.0
    print(f"[optim] base_lr={args.lr} effective_lr={effective_lr} batch_size={args.batch_size}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=effective_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.min_lr)
    criterion = nn.CrossEntropyLoss()
    train_runtime = TrainRuntime(model, train_loader, optimizer, criterion, device)
    train_controls = TrainControls(args.debug_train_batches, args.print_freq, args.sam_rho)
    best_acc = 0.0
    start_epoch = 0
    if resume_path is not None:
        resume_state = load_training_checkpoint(
            ResumeTarget(model, optimizer, scheduler, device),
            resume_path,
        )
        start_epoch = resume_state.start_epoch
        best_acc = resume_state.best_acc
        if not resume_state.loaded_scheduler:
            fast_forward_scheduler(scheduler, start_epoch)
    start = time.time()
    for epoch in range(start_epoch, args.epochs):
        qsam_active = True
        if use_qsam and args.qsam_warmup_epochs > 0:
            qsam_active = epoch >= args.qsam_warmup_epochs
            set_qsam_on(model, qsam_active)
        epoch_train_mode = train_mode
        if train_mode is QsamTrainMode.TWO_PASS and not qsam_active:
            epoch_train_mode = QsamTrainMode.BASELINE
        train_stats = train_one_epoch(train_runtime, train_controls, epoch_train_mode)
        val_stats = evaluate(model, val_loader, criterion, device, args)
        epoch_lr = optimizer.param_groups[0]["lr"]  # capture before the scheduler advances it
        scheduler.step()
        best_acc = max(best_acc, val_stats["acc1"])
        print(
            f"epoch={epoch} train_acc1={train_stats['acc1']:.3f} "
            f"val_acc1={val_stats['acc1']:.3f} val_acc5={val_stats['acc5']:.3f} best={best_acc:.3f}"
        )
        if args.wandb:
            wandb.log(
                {
                    "train/loss": train_stats["loss"],
                    "train/acc1": train_stats["acc1"],
                    "val/loss": val_stats["loss"],
                    "val/acc1": val_stats["acc1"],
                    "val/acc5": val_stats["acc5"],
                    "val/best_acc1": best_acc,
                    "lr": epoch_lr,
                    "qsam_active": int(qsam_active) if use_qsam else 0,
                },
                step=epoch,
            )
        if output_dir is not None:
            with (output_dir / "log.txt").open("a") as file:
                file.write(
                    json.dumps(
                        {"epoch": epoch, "train_mode": epoch_train_mode.value, "train": train_stats, "val": val_stats}
                    )
                    + "\n"
                )
            save_training_checkpoint(
                CheckpointPayload(model, optimizer, scheduler, epoch, best_acc),
                output_dir / "checkpoint.pth",
            )
    elapsed = datetime.timedelta(seconds=int(time.time() - start))
    print(f"training_time={elapsed} best_acc1={best_acc:.3f}")
    if args.wandb:
        wandb.summary["best_acc1"] = best_acc
        wandb.finish()


if __name__ == "__main__":
    main()
