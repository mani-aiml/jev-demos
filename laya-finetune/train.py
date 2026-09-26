"""Fine-tune Laya on the judging task, on one Apple-silicon GPU.

Ported from Convai's official notebook (laya_finetune_typed_decisions_2xT4_kaggle.ipynb):
the same loss (a policy gradient on a proper-scoring reward, plus soft cross-entropy), the
same learning rates, noise schedule and effective batch of 64, and the same held-out
temperature fit. What changes is the hardware: one MPS device instead of two T4s under
DDP, so no process group and no GradScaler, and bf16 autocast instead of fp16.

    python train.py --train data/train_all.jsonl --out runs/all2082          # balanced mix by default
    python train.py --train data/train_all.jsonl --max-steps 25               # a short look, saves nothing

Before the first step it prints and plots the class mix it will train on (runs/<name>/mix.png)
and refuses to start on a skewed one: see prepare_data.py and mix_report.py.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer
from laya.common import build_model, collate_items, proper_reward

from calibrate import fit_temperatures
from mix_report import preflight
from finetune_data import base_checkpoint, questions, read_jsonl, to_items
from prepare_data import weight_to_design

MICRO_BATCH, GRAD_ACCUM, GROUP_SIZE = 8, 8, 4   # 8 x 8 = the notebook's 64 per update
LR_ENCODER, LR_HEAD = 2.5e-5, 1.0e-4
SIGMA_START, SIGMA_END = 0.4, 0.1
CALIB_SHARE, SEED = 0.1, 20260924
DEVICE = torch.device("mps")


def load_model(model_dir: str):
    cfg = json.load(open(os.path.join(model_dir, "rl_agent_config.json")))
    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
    return model.to(DEVICE), cfg, AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))


def forward(model, batch):
    keys = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")
    with torch.autocast("mps", dtype=torch.bfloat16):
        logits, act = model(*(batch[k].to(DEVICE) for k in keys))
    return logits.float(), act


def loss_on(model, batch, sigma: float):
    """RLCD: sample noisy distributions, reward them with a proper score, push toward the better ones."""
    logits, act = forward(model, batch)
    mask, target = batch["marker_mask"].to(DEVICE), batch["target"].to(DEVICE)
    k = mask.sum(-1, keepdim=True).float()
    eps = torch.randn((GROUP_SIZE,) + logits.shape, device=DEVICE) * sigma * mask
    eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
    z = logits.detach().unsqueeze(0) + eps
    q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
    with torch.no_grad():
        r = proper_reward(q, target.unsqueeze(0), batch["qtype"].to(DEVICE), mask, w_sph=0.75, w_rps=1.0)
        adv = (r - r.mean(0, keepdim=True)) / ((r - r.mean(0, keepdim=True)).std() + 1e-6)
    logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma**2)
    ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1)
    w = torch.tensor([m["weight"] for m in batch["meta"]], device=DEVICE)  # all 1.0 unless --mix design
    loss_rl = -((adv * logp).mean(0) * w).sum() / w.sum()
    return loss_rl + (ce * w).sum() / w.sum() + 0.0 * act.sum()


def train(model, items, pad_id: int, epochs: int, max_steps: int | None, seed: int) -> None:
    enc = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    head = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc, "lr": LR_ENCODER}, {"params": head, "lr": LR_HEAD}], weight_decay=0.01)
    updates = max(1, len(items) // (MICRO_BATCH * GRAD_ACCUM) * epochs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=updates, eta_min=1e-6)
    model.train()
    step, t0 = 0, time.time()
    for epoch in range(epochs):
        sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * epoch / max(1, epochs - 1)
        random.Random(seed + epoch).shuffle(items)
        for start in range(0, len(items), MICRO_BATCH):
            loss = loss_on(model, collate_items([items[start:start + MICRO_BATCH]], pad_id), sigma) / GRAD_ACCUM
            loss.backward()
            step += 1
            if step % GRAD_ACCUM == 0 or start + MICRO_BATCH >= len(items):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(), sched.step(), opt.zero_grad(set_to_none=True)
            if step % 25 == 0:
                rate = step * MICRO_BATCH / (time.time() - t0)
                print(f"epoch {epoch + 1}/{epochs} step {step} loss {loss.item() * GRAD_ACCUM:.4f} {rate:.1f} seq/s", flush=True)
            if max_steps and step >= max_steps:
                return
    print(f"trained {step} steps in {time.time() - t0:.0f} s", flush=True)


def save(model, cfg: dict, tok, temps: list[float], out: str) -> None:
    """Same layout as the published checkpoints, so `laya` and `laya-mlx` both load it."""
    os.makedirs(out, exist_ok=True)
    state = {k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(state, os.path.join(out, "model.safetensors"))
    model.encoder.config.save_pretrained(os.path.join(out, "encoder"))
    tok.save_pretrained(os.path.join(out, "tokenizer"))
    cfg = {**cfg, "fine_tuned": True, "model_name": "laya-judge", "temperature": temps}
    cfg.pop("temperature_by_options", None)  # this fit is per type; inherited buckets would hide it
    json.dump(cfg, open(os.path.join(out, "rl_agent_config.json"), "w"), indent=2)
    print(f"saved {out}  temperatures (choice, score, noul) {[round(t, 3) for t in temps]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="a prepared file from prepare_data.py")
    ap.add_argument("--rows", type=int, default=None, help="use only the first N rows")
    ap.add_argument("--criterion", choices=("short", "long"), default="short")
    ap.add_argument("--mix", choices=("none", "design", "balanced"), default="balanced",
                    help="design: 1-in-3 clean as generated; balanced: also every defect type equal")
    ap.add_argument("--allow-skew", action="store_true", help="train even if the mix is off the design")
    ap.add_argument("--labels", choices=("planted", "sonnet"), default="planted")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = read_jsonl(Path(args.train))[: args.rows]
    random.Random(args.seed).shuffle(rows)
    if args.mix != "none":
        weight_to_design(rows, balance_types=args.mix == "balanced")
    preflight(rows, args.mix, args.out, args.allow_skew)

    torch.manual_seed(args.seed)
    model, cfg, tok = load_model(base_checkpoint())
    n_calib = int(len(rows) * CALIB_SHARE)  # split by row, so no item's text is on both sides
    qs = questions(args.criterion)
    calib = to_items(rows[:n_calib], qs, tok, cfg, args.labels)
    train_items = to_items(rows[n_calib:], qs, tok, cfg, args.labels)
    print(f"{len(rows)} rows -> {len(train_items)} train / {len(calib)} calibration sequences", flush=True)
    train(model, train_items, tok.pad_token_id, args.epochs, args.max_steps, args.seed)
    if args.out:
        save(model, cfg, tok, fit_temperatures(model, forward, calib, tok.pad_token_id), args.out)


if __name__ == "__main__":
    main()
