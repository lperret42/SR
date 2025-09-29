#!/usr/bin/env python3
"""
SwinIR training script (x4) — grouped validation, bf16/fp16, optional torch.compile

- Validation par groupe JP2 via metadata.json (images[*].filename)
- BF16 (--bf16) ou FP16 (--fp16) via torch.cuda.amp.autocast
- Option --compile (torch.compile, mode="max-autotune")
- Augmentations géométriques (flip/rot90) + jitter couleur optionnel
- Prétraining strict, PSNR/SSIM/L1/L2, checkpoints, samples, warmup, cosine/plateau
- (NOUVEAU) --check-split-integrity : assert aucun doublon train/val et aucun parent JP2 partagé
"""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from tqdm import tqdm

from models.network_swinir import SwinIR, get_realsr_config
from utils.util_calculate_psnr_ssim import calculate_psnr, calculate_ssim


# -----------------------
# Dataset & split helpers
# -----------------------

def load_metadata(dataset_root: Path) -> Dict:
    meta_path = dataset_root / "metadata.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            return json.load(f)
    return {}


def build_entries(dataset_root: Path) -> List[Dict]:
    """
    Retourne une liste d'items :
      {'sr_path': Path, 'lr_path': Path, 'parent_filename': str}
    Préfère metadata.json ; fallback mapping.csv avec parent 'UNKNOWN'.
    """
    entries: List[Dict] = []
    meta = load_metadata(dataset_root)

    if meta and "images" in meta:
        for im in meta["images"]:
            parent = im.get("filename", "UNKNOWN")
            for t in im.get("tiles", []):
                entries.append({
                    "sr_path": dataset_root / t["hr_path"],
                    "lr_path": dataset_root / t["lr_path"],
                    "parent_filename": parent
                })
        return entries

    mapping = dataset_root / "mapping.csv"
    if mapping.exists():
        with open(mapping, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entries.append({
                    "sr_path": dataset_root / row["sr_path"],
                    "lr_path": dataset_root / row["lr_path"],
                    "parent_filename": "UNKNOWN"
                })
    return entries


def split_by_parent(entries: List[Dict], val_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    """Split strict par parent_filename (source JP2) : aucune fuite train→val."""
    groups: Dict[str, List[Dict]] = {}
    for e in entries:
        groups.setdefault(e["parent_filename"], []).append(e)

    parents = list(groups.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(parents)

    n_parents = len(parents)
    if n_parents == 0:
        raise RuntimeError("Aucun parent trouvé dans le dataset.")
    if n_parents == 1:
        # Pas idéal, mais on met au moins 1 tile en val pour éviter crash (pas groupé strict)
        only_parent = parents[0]
        val_size = max(1, int(round(len(groups[only_parent]) * val_ratio)))
        return groups[only_parent][val_size:], groups[only_parent][:val_size]

    n_val = max(1, int(round(n_parents * val_ratio)))
    val_parents = set(parents[:n_val])
    train_parents = set(parents[n_val:])

    train_entries, val_entries = [], []
    for p in parents:
        if p in val_parents:
            val_entries.extend(groups[p])
        else:
            train_entries.extend(groups[p])

    print(f"[Group Split] Parents total: {n_parents} | Train parents: {len(train_parents)} | Val parents: {len(val_parents)}")
    print(f"[Group Split] Tiles -> Train: {len(train_entries)} | Val: {len(val_entries)}")
    if "UNKNOWN" in val_parents or "UNKNOWN" in train_parents:
        print("⚠ Pas de metadata.json : parent='UNKNOWN' — split groupé non garanti.")
    return train_entries, val_entries


def assert_split_integrity(train_entries: List[Dict], val_entries: List[Dict]) -> None:
    """
    Vérifie :
      - aucune tuile identique entre train et val (sr_path & lr_path)
      - aucun parent JP2 (parent_filename) partagé entre train/val
    Lève AssertionError avec détails si problème.
    """
    train_sr = {str(e["sr_path"]) for e in train_entries}
    val_sr   = {str(e["sr_path"]) for e in val_entries}
    dup_sr = train_sr & val_sr

    train_lr = {str(e["lr_path"]) for e in train_entries}
    val_lr   = {str(e["lr_path"]) for e in val_entries}
    dup_lr = train_lr & val_lr

    train_par = {e["parent_filename"] for e in train_entries}
    val_par   = {e["parent_filename"] for e in val_entries}
    shared_parents = train_par & val_par

    msgs = []
    if dup_sr:
        sample = list(sorted(dup_sr))[:5]
        msgs.append(f"- {len(dup_sr)} SR en double entre train/val. Exemples: {sample}")
    if dup_lr:
        sample = list(sorted(dup_lr))[:5]
        msgs.append(f"- {len(dup_lr)} LR en double entre train/val. Exemples: {sample}")
    if shared_parents:
        sample = list(sorted(shared_parents))[:5]
        msgs.append(f"- Parents JP2 partagés entre train/val: {sample}")

    if msgs:
        full = " / ".join(msgs)
        raise AssertionError(f"Split integrity FAILED: {full}")
    else:
        print("✓ Split integrity OK: aucun doublon LR/SR et aucun parent JP2 partagé.")


class OrthoSRDataset(Dataset):
    """Loader tuiles LR/SR ; aug géo (train) et jitter couleur optionnel."""
    def __init__(self, entries: List[Dict], split: str = "train",
                 augment_geom: bool = True, augment_color: bool = False,
                 color_jitter_strength: float = 0.05):
        super().__init__()
        self.entries = entries
        self.split = split
        self.augment_geom = augment_geom and split == "train"
        self.augment_color = augment_color and split == "train"
        self.cj = color_jitter_strength
        print(f"Dataset {split}: {len(self.entries)} pairs")

    def __len__(self) -> int:
        return len(self.entries)

    @staticmethod
    def _geom_aug_pair(lr: torch.Tensor, sr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1) < 0.5:  # H flip
            lr = torch.flip(lr, dims=[2]); sr = torch.flip(sr, dims=[2])
        if torch.rand(1) < 0.5:  # V flip
            lr = torch.flip(lr, dims=[1]); sr = torch.flip(sr, dims=[1])
        k = int(torch.randint(0, 4, (1,)).item())  # rot90
        if k:
            lr = torch.rot90(lr, k=k, dims=[1, 2])
            sr = torch.rot90(sr, k=k, dims=[1, 2])
        return lr, sr

    def _color_jitter_pair(self, lr: torch.Tensor, sr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b = 1.0 + (torch.rand(1).item() * 2 - 1) * self.cj
        c = 1.0 + (torch.rand(1).item() * 2 - 1) * self.cj
        s = 1.0 + (torch.rand(1).item() * 2 - 1) * (self.cj * 0.5)
        h = (torch.rand(1).item() * 2 - 1) * (self.cj * 0.2)

        def apply(x: torch.Tensor) -> torch.Tensor:
            y = TF.adjust_brightness(x, b)
            y = TF.adjust_contrast(y, c)
            y = TF.adjust_saturation(y, s)
            y = TF.adjust_hue(y, h)
            return torch.clamp(y, 0, 1)

        return apply(lr), apply(sr)

    def __getitem__(self, idx: int):
        e = self.entries[idx]
        sr = TF.to_tensor(Image.open(e["sr_path"]).convert("RGB"))  # [3,256,256]
        lr = TF.to_tensor(Image.open(e["lr_path"]).convert("RGB"))  # [3, 64, 64]

        if self.augment_geom:
            lr, sr = self._geom_aug_pair(lr, sr)
        if self.augment_color:
            lr, sr = self._color_jitter_pair(lr, sr)

        name = Path(e["sr_path"]).name
        return lr, sr, name


# ---------------
# Losses & metrics
# ---------------

class PerceptualLoss(nn.Module):
    """VGG19 perceptuel (conv1_2..conv5_4 moyenné)."""
    def __init__(self):
        super().__init__()
        from torchvision.models import vgg19, VGG19_Weights
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features
        self.slice1 = nn.Sequential(*list(vgg.children())[:4])
        self.slice2 = nn.Sequential(*list(vgg.children())[4:9])
        self.slice3 = nn.Sequential(*list(vgg.children())[9:18])
        self.slice4 = nn.Sequential(*list(vgg.children())[18:27])
        self.slice5 = nn.Sequential(*list(vgg.children())[27:36])
        self.slices = [self.slice1, self.slice2, self.slice3, self.slice4, self.slice5]
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        loss = 0.0; xf, yf = x, y
        for sl in self.slices:
            xf = sl(xf); yf = sl(yf)
            loss = loss + F.l1_loss(xf, yf)
        return loss / len(self.slices)


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6): super().__init__(); self.eps = eps
    def forward(self, x, y): return torch.mean(torch.sqrt((x - y) ** 2 + self.eps))


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Simple L1 gradient consistency loss using forward differences."""
    dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dx_target = target[:, :, :, 1:] - target[:, :, :, :-1]
    dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dy_target = target[:, :, 1:, :] - target[:, :, :-1, :]
    loss_x = torch.mean(torch.abs(dx_pred - dx_target))
    loss_y = torch.mean(torch.abs(dy_pred - dy_target))
    return loss_x + loss_y


@torch.no_grad()
def compute_batch_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    pred = pred.float(); target = target.float()
    pred_np = (pred.cpu().numpy() * 255).astype(np.uint8)
    tgt_np = (target.cpu().numpy() * 255).astype(np.uint8)
    psnr_sum, ssim_sum = 0.0, 0.0
    for i in range(pred.shape[0]):
        p = pred_np[i].transpose(1, 2, 0)
        t = tgt_np[i].transpose(1, 2, 0)
        psnr_sum += calculate_psnr(p, t, crop_border=4, input_order="HWC") or 0.0
        ssim_sum += calculate_ssim(p, t, crop_border=4, input_order="HWC") or 0.0
    return {"psnr": psnr_sum / pred.shape[0],
            "ssim": ssim_sum / pred.shape[0],
            "l1": F.l1_loss(pred, target).item(),
            "l2": F.mse_loss(pred, target).item()}


def _torch_load_compat(path, map_location="cpu"):
    """Load checkpoints compatible with PyTorch>=2.6 (weights_only default True).
    Always uses weights_only=False for trusted local files, falling back if arg unsupported.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Older PyTorch without weights_only kwarg
        return torch.load(path, map_location=map_location)


def _is_state_dict_like(obj) -> bool:
    return isinstance(obj, (dict, OrderedDict)) and len(obj) > 0 and all(isinstance(v, torch.Tensor) for v in obj.values())


def _strip_prefix(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not prefix:
        return sd
    plen = len(prefix)
    if not all(k.startswith(prefix) for k in sd.keys()):
        return sd
    return {k[plen:]: v for k, v in sd.items()}


def _try_load(model: nn.Module, sd: Dict[str, torch.Tensor], strict_first: bool = True) -> tuple[bool, str]:
    # Try several common prefixes
    prefixes = [
        "", "module.", "model.", "net.", "network.", "generator.", "swinir.",
        "module.model.", "ema.", "params.", "student.",
    ]
    last_err = ""
    for pfx in prefixes:
        sd_p = _strip_prefix(sd, pfx)
        try:
            model.load_state_dict(sd_p, strict=True if strict_first else False)
            return True, f"loaded (strict={strict_first}, prefix='{pfx}')"
        except Exception as e:
            last_err = str(e)
    if strict_first:
        # Try non-strict as fallback to allow partial load
        for pfx in prefixes:
            sd_p = _strip_prefix(sd, pfx)
            try:
                missing, unexpected = model.load_state_dict(sd_p, strict=False)
                msg = (f"loaded (strict=False, prefix='{pfx}', missing={len(missing)}, unexpected={len(unexpected)})")
                if missing or unexpected:
                    msg += f"\n  missing: {list(missing)[:8]} ...\n  unexpected: {list(unexpected)[:8]} ..."
                return True, msg
            except Exception as e:
                last_err = str(e)
    return False, last_err


def load_weights_robust(model: nn.Module, ckpt_obj) -> None:
    """Robustly locate and load a state_dict from a checkpoint object into model.
    - Supports top-level state_dict or under common keys (model/state_dict/params_ema/...)
    - Tries common key prefixes (module., model., net., ...)
    - Falls back to non-strict load with summary, otherwise raises.
    """
    candidates: List[Dict[str, torch.Tensor]] = []

    def add_candidate(obj):
        if _is_state_dict_like(obj):
            candidates.append(obj)  # type: ignore[arg-type]

    if _is_state_dict_like(ckpt_obj):
        add_candidate(ckpt_obj)

    if isinstance(ckpt_obj, (dict, OrderedDict)):
        keys_to_try = [
            "model", "state_dict", "params_ema", "params", "net", "network",
            "generator", "ema", "module", "weights", "state",
        ]
        for k in keys_to_try:
            if k in ckpt_obj and isinstance(ckpt_obj[k], (dict, OrderedDict)):
                add_candidate(ckpt_obj[k])
        # Nested common pattern: {"state_dict": {"model": sd}}
        sd = ckpt_obj.get("state_dict")
        if isinstance(sd, (dict, OrderedDict)):
            for k in keys_to_try:
                if k in sd and isinstance(sd[k], (dict, OrderedDict)):
                    add_candidate(sd[k])

    tried_msgs = []
    last_err = ""
    for sd in candidates:
        ok, msg = _try_load(model, sd, strict_first=True)
        tried_msgs.append(msg if ok else f"fail: {msg}")
        if ok:
            print("✓ Init weights", msg)
            return
        last_err = msg

    raise RuntimeError(
        "Failed to load any state_dict from checkpoint.\n"
        f"Tried candidates: {len(candidates)}\nLast error: {last_err}"
    )


# ------
# Model
# ------

def create_model(model_size: str, pretrained: bool, device: str) -> SwinIR:
    cfg = get_realsr_config(model_size)
    pretrained_path_value = cfg.pop("pretrained_path", None)
    pretrained_path = Path(pretrained_path_value) if pretrained_path_value else None
    pretrained_url = cfg.pop("pretrained_url", None)

    model = SwinIR(**cfg)

    if pretrained:
        if not pretrained_path:
            raise ValueError(f"No pretrained weights available for SwinIR-{model_size}.")
        weights = pretrained_path
        if not weights.exists():
            if not pretrained_url:
                raise FileNotFoundError(
                    f"Pretrained weights for SwinIR-{model_size} not found locally and no download URL provided."
                )
            print(f"Téléchargement SwinIR-{model_size} pré-entraîné ...")
            import urllib.request
            urllib.request.urlretrieve(pretrained_url, weights)
            print(f"Saved to: {weights}")
        ckpt = _torch_load_compat(weights, map_location="cpu")
        if isinstance(ckpt, dict) and "params_ema" in ckpt:
            model.load_state_dict(ckpt["params_ema"], strict=True)
        else:
            model.load_state_dict(ckpt, strict=True)
        print(f"Loaded pretrained weights from {weights}")

    return model.to(device)


# -------------
# Train / Valid
# -------------

def forward_sr(model, lr, amp_dtype, amp_enabled):
    if amp_enabled:
        with autocast(dtype=amp_dtype):
            return model(lr)
    return model(lr)

def loss_compute(criterion, sr, hr, amp_dtype, amp_enabled):
    if amp_enabled:
        with autocast(dtype=amp_dtype):
            return criterion(sr, hr)
    return criterion(sr, hr)

def train_epoch(model, loader, criterion, optimizer, device, amp_dtype=None, amp=False,
                scaler: GradScaler | None = None, grad_clip=0.0) -> float:
    model.train()
    total = 0.0
    pbar = tqdm(loader, desc="Training")
    for lr, hr, _ in pbar:
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:  # FP16 + scaler
            with autocast(dtype=amp_dtype):
                sr = model(lr)
                loss = criterion(sr, hr)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            sr = forward_sr(model, lr, amp_dtype, amp)
            loss = loss_compute(criterion, sr, hr, amp_dtype, amp)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device, amp_dtype=None, amp=False):
    model.eval()
    total = 0.0
    agg = {"psnr": 0.0, "ssim": 0.0, "l1": 0.0, "l2": 0.0}
    pbar = tqdm(loader, desc="Validation")
    for lr, hr, _ in pbar:
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)
        sr = forward_sr(model, lr, amp_dtype, amp)
        loss = loss_compute(criterion, sr, hr, amp_dtype, amp)
        total += loss.item()
        m = compute_batch_metrics(sr, hr)
        for k in agg: agg[k] += m[k]
        pbar.set_postfix(loss=f"{loss.item():.4f}", psnr=f"{m['psnr']:.2f}")

    n = len(loader)
    avg_loss = total / max(1, n)
    for k in agg: agg[k] /= max(1, n)
    return avg_loss, agg


@torch.no_grad()
def save_visual_samples(model, loader, out_dir: Path, device, n_samples=100, amp_dtype=None, amp=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    model.eval()
    for lr, hr, names in loader:
        lr = lr.to(device); hr = hr.to(device)
        sr = forward_sr(model, lr, amp_dtype, amp)
        lr_bi = torch.clamp(F.interpolate(lr, scale_factor=4, mode="bicubic", align_corners=False), 0, 1)
        # Small white separator band to better distinguish the panels
        gap_width = 8
        for i in range(lr.shape[0]):
            if saved >= n_samples: return
            gap = torch.ones((3, sr[i].shape[1], gap_width), device=sr.device, dtype=sr.dtype).float()
            grid = torch.cat([lr_bi[i].float(), gap, sr[i].float(), gap, hr[i].float()], dim=2)
            fn = names[i].replace(".png", "_comparison.png")
            save_image(grid, out_dir / fn)
            saved += 1


# -----
# Main
# -----

def main():
    parser = argparse.ArgumentParser("SwinIR orthophoto SR x4 (grouped val, bf16/fp16, --compile)")

    # Required
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model-size", choices=["M", "L", "XL", "XXL"], default="M")
    parser.add_argument("--pretrained", action="store_true")
    # NEW: init from checkpoint
    parser.add_argument("--init-from", type=str, default=None,
                        help="Path to a .pth checkpoint to initialize model weights (overrides --pretrained if both are set)")

    # Train setup
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--compile", action="store_true", default=False)

    # Precision
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 mixed precision")
    parser.add_argument("--fp16", action="store_true", help="Use float16 mixed precision")

    # Loss
    parser.add_argument("--loss-type",
                        choices=["l1", "l2", "charbonnier", "l1+perceptual", "charbonnier+perceptual"],
                        default="l1+perceptual")
    parser.add_argument("--perceptual-weight", type=float, default=0.1)

    # Augment
    parser.add_argument("--no-geom-aug", action="store_true", help="Disable geometric aug (flip/rot90)")
    parser.add_argument("--color-jitter", action="store_true", help="Enable light color jitter")
    parser.add_argument("--color-jitter-strength", type=float, default=0.05)

    # Scheduler
    parser.add_argument("--scheduler", choices=["cosine", "plateau", "none"], default="cosine")
    parser.add_argument("--min-lr", type=float, default=1e-7)
    parser.add_argument("--patience", type=int, default=10)

    # Eval / save
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument("--save-visual-interval", type=int, default=1)
    parser.add_argument("--n-visual-samples", type=int, default=3000)
    parser.add_argument("--output-dir", type=str, default=None)

    # Misc
    parser.add_argument("--gradient-clip", type=float, default=0.0)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--warmup-epochs", type=int, default=0)

    # NEW: split integrity check
    parser.add_argument("--check-split-integrity", action="store_true",
                        help="Assert no duplicate samples across train/val and no shared parent JP2 between splits")

    args = parser.parse_args()

    XXL_LOSS_NAME = "xxl_charb+perc+grad"
    xxl_defaults = args.model_size == "XXL"
    effective_loss_type = XXL_LOSS_NAME if xxl_defaults else args.loss_type
    loss_override_details: Dict[str, float | str] = {}
    if xxl_defaults:
        loss_override_details = {
            "requested_loss_type": args.loss_type,
            "requested_perceptual_weight": args.perceptual_weight,
            "charbonnier_weight": 1.0,
            "perceptual_weight_used": 0.2,
            "gradient_weight_used": 0.05,
        }

    if args.bf16 and args.fp16:
        raise ValueError("Choisir soit --bf16 soit --fp16, pas les deux.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    loss_token_for_dir = effective_loss_type
    out_dir = Path(args.output_dir) if args.output_dir else Path("runs") / (
        ("pretrained_" if args.pretrained else "") +
        f"swinir_{args.model_size}_lr{args.lr}_bs{args.batch_size}_{loss_token_for_dir}_" +
        ("bf16" if args.bf16 else ("fp16" if args.fp16 else "fp32")) + "_" +
        datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    args_json = vars(args).copy()
    args_json["output_dir"] = str(out_dir)
    args_json["loss_type_effective"] = effective_loss_type
    args_json["xxl_defaults_active"] = xxl_defaults
    if loss_override_details:
        args_json["xxl_loss_overrides"] = loss_override_details
    with open(out_dir / "args.json", "w") as f:
        json.dump(args_json, f, indent=2)

    print("=" * 50)
    print(f"Experiment: {out_dir.name}")
    print(f"Device: {args.device}")
    print(f"Model: SwinIR-{args.model_size} (pretrained={args.pretrained})")
    print(f"Precision: {'BF16' if args.bf16 else ('FP16' if args.fp16 else 'FP32')}")
    print(f"Compile: {args.compile}")
    print("=" * 50)

    root = Path(args.dataset)
    all_entries = build_entries(root)
    if len(all_entries) == 0:
        raise RuntimeError("Aucune paire trouvée. Vérifie metadata.json ou mapping.csv.")
    train_entries, val_entries = split_by_parent(all_entries, args.val_ratio, args.seed)

    # ---- NEW: optional integrity check on raw entry lists (before wrapping datasets) ----
    if args.check_split_integrity:
        assert_split_integrity(train_entries, val_entries)

    ds_train = OrthoSRDataset(
        train_entries, split="train",
        augment_geom=(not args.no_geom_aug),
        augment_color=args.color_jitter,
        color_jitter_strength=args.color_jitter_strength
    )
    ds_val = OrthoSRDataset(val_entries, split="val", augment_geom=False, augment_color=False)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True,
                          persistent_workers=args.num_workers > 0)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                        persistent_workers=args.num_workers > 0)

    model = create_model(args.model_size, args.pretrained, args.device)

    if xxl_defaults:
        print("XXL defaults engaged:")
        print(f" - Reconstruction head forced to 'nearest+conv' with widened feature width ({getattr(model, 'num_feat', 'unknown')} channels).")
        print(" - Composite loss: Charbonnier + 0.2·Perceptual + 0.05·Gradient (ignoring --loss-type/--perceptual-weight).")
        if loss_override_details:
            print(f"   Requested loss '{loss_override_details['requested_loss_type']}' ignored.")
            print(f"   Requested perceptual weight {loss_override_details['requested_perceptual_weight']} ignored.")

    # If provided, initialize weights from a specific checkpoint path (robust loader)
    if args.init_from is not None:
        ckpt_path = Path(args.init_from)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        print(f"Loading init weights from: {ckpt_path}")
        ckpt = _torch_load_compat(ckpt_path, map_location="cpu")
        load_weights_robust(model, ckpt)
        if args.pretrained:
            print("Note: --init-from overrides weights loaded via --pretrained.")

    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model, mode="max-autotune")
        print("✓ torch.compile enabled")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    if effective_loss_type == "l1":
        criterion = nn.L1Loss()
    elif effective_loss_type == "l2":
        criterion = nn.MSELoss()
    elif effective_loss_type == "charbonnier":
        criterion = CharbonnierLoss()
    elif effective_loss_type in ("l1+perceptual", "charbonnier+perceptual"):
        base = nn.L1Loss() if effective_loss_type.startswith("l1") else CharbonnierLoss()
        perceptual = PerceptualLoss().to(args.device)
        def combined(pred, tgt): return base(pred, tgt) + args.perceptual_weight * perceptual(pred, tgt)
        criterion = combined
    elif effective_loss_type == XXL_LOSS_NAME:
        base = CharbonnierLoss()
        perceptual = PerceptualLoss().to(args.device)
        perc_w = loss_override_details.get("perceptual_weight_used", 0.2) if loss_override_details else 0.2
        grad_w = loss_override_details.get("gradient_weight_used", 0.05) if loss_override_details else 0.05
        def combined(pred, tgt):
            return base(pred, tgt) + perc_w * perceptual(pred, tgt) + grad_w * gradient_loss(pred, tgt)
        criterion = combined
    else:
        raise ValueError("Unknown loss type")
    print(f"Loss: {effective_loss_type}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    elif args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=args.patience, min_lr=args.min_lr)
    else:
        scheduler = None

    amp_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
    use_amp = amp_dtype is not None
    scaler = GradScaler() if args.fp16 else None  # scaler uniquement pour FP16

    if args.eval_only:
        print("\n" + "=" * 50 + "\nEVALUATION ONLY\n" + "=" * 50)
        val_loss, metrics = validate(model, dl_val, criterion, args.device, amp_dtype=amp_dtype, amp=use_amp)
        print(f"\nVal Loss: {val_loss:.4f}")
        print(f"PSNR: {metrics['psnr']:.2f} dB | SSIM: {metrics['ssim']:.4f} | L1: {metrics['l1']:.4f} | L2: {metrics['l2']:.4f}")
        save_visual_samples(model, dl_val, out_dir / "eval_samples", args.device,
                            n_samples=args.n_visual_samples, amp_dtype=amp_dtype, amp=use_amp)
        return

    best_psnr = -1e9
    patience = 0
    history = {"train_loss": [], "val_loss": [], "val_psnr": [], "val_ssim": []}

    print("\n" + "=" * 50 + "\nSTART TRAINING\n" + "=" * 50)
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 30)

        if args.warmup_epochs > 0 and epoch <= args.warmup_epochs:
            warm_lr = args.lr * (epoch / max(1, args.warmup_epochs))
            for g in optimizer.param_groups: g["lr"] = warm_lr
            print(f"Warmup LR: {warm_lr:.2e}")

        t0 = time.time()
        tr_loss = train_epoch(model, dl_train, criterion, optimizer, args.device,
                              amp_dtype=amp_dtype, amp=use_amp, scaler=scaler,
                              grad_clip=args.gradient_clip)
        print(f"Train Loss: {tr_loss:.4f} (time: {time.time()-t0:.1f}s)")
        history["train_loss"].append(tr_loss)

        if epoch % args.eval_interval == 0:
            v0 = time.time()
            val_loss, metrics = validate(model, dl_val, criterion, args.device, amp_dtype=amp_dtype, amp=use_amp)
            print(f"Val Loss: {val_loss:.4f} (time: {time.time()-v0:.1f}s)")
            print(f"Metrics - PSNR: {metrics['psnr']:.2f} dB | SSIM: {metrics['ssim']:.4f} | "
                  f"L1: {metrics['l1']:.4f} | L2: {metrics['l2']:.4f}")
            history["val_loss"].append(val_loss)
            history["val_psnr"].append(metrics["psnr"])
            history["val_ssim"].append(metrics["ssim"])

            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau): scheduler.step(metrics["psnr"])
                else: scheduler.step()
                print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")

            if metrics["psnr"] > best_psnr:
                best_psnr = metrics["psnr"]; patience = 0
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "best_psnr": best_psnr,
                    "args": args_json,
                }, out_dir / "best_model.pth")
                print(f"✓ New best PSNR {best_psnr:.2f} dB — checkpoint saved")
            else:
                patience += 1
                if patience >= args.early_stop_patience:
                    print(f"⚠ Early stopping at epoch {epoch}")
                    break

        if epoch % args.save_interval == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "best_psnr": best_psnr,
                "history": history,
                "args": args_json,
            }, out_dir / f"checkpoint_epoch_{epoch}.pth")
            print("✓ Checkpoint saved")

        if epoch % args.save_visual_interval == 0:
            sample_dir = out_dir / f"samples_epoch_{epoch}"
            save_visual_samples(model, dl_val, sample_dir, args.device,
                                n_samples=args.n_visual_samples, amp_dtype=amp_dtype, amp=use_amp)
            print(f"✓ Visual samples saved to {sample_dir}")

        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print("\n" + "=" * 50)
    print(f"Training completed! Best PSNR: {best_psnr:.2f} dB")
    print(f"Results saved to: {out_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()

