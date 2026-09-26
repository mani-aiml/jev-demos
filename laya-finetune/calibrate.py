"""Fit one softmax temperature per question type on items the run never trained on.

Same method as Convai's notebook (LBFGS on log T, clamped to [0.1, 10]). Fitting on
training items would measure the fit, not the calibration: the model is near-certain and
near-correct on them, so the optimiser has nothing to soften.
"""

import torch
from laya.common import collate_items


def fit_one_temp(pairs: list[tuple[torch.Tensor, list[float]]]) -> float:
    if len(pairs) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in pairs)
    Z, T = torch.full((len(pairs), kmax), -1e4), torch.zeros((len(pairs), kmax))
    for i, (z, t) in enumerate(pairs):
        Z[i, : len(z)], T[i, : len(t)] = z, torch.tensor(t)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


@torch.no_grad()
def fit_temperatures(model, forward, calib: list[dict], pad_id: int) -> list[float]:
    """[choice, score, noul] temperatures; types absent from the slice keep 1.0."""
    model.eval()
    preds = []
    for start in range(0, len(calib), 16):
        chunk = calib[start : start + 16]
        logits, _ = forward(model, collate_items([chunk], pad_id))
        for row, it in zip(logits.cpu(), chunk):
            preds.append((it["qtype"], row[: len(it["markers"])], it["target"]))
    return [fit_one_temp([(z, t) for q, z, t in preds if q == qt]) for qt in range(3)]
