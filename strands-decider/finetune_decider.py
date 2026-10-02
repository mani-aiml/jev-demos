"""Continue Strands Decider v19 on the judging rows: its LoRA adapter and pointer head both keep training.

Decider's own `init_from` does not do this: it freezes the torso and swaps in a fresh 24-slot head
(strands_decider/train.py, `if cfg.init_from`), which drops v19's pointer head and cannot hold the
40-way question. So only the base loader is replaced, by one that returns v19 with its adapter
trainable and its calibration reset (as `init_from` resets it); the rest is their `train(cfg)`.

    python finetune_decider.py configs/judge.yaml [max_steps]
    strands-decider calibrate runs/decider-judge --data data/judge_calib.jsonl --split all
"""

import sys

from strands_decider import modeling
from strands_decider.train import TrainConfig, train

V19 = "StrandsAgents/strands-decider-2B-hobson-v19"


def load_v19_trainable(config, *, attn_implementation=None, **_):
    """v19 as released, with LoRA and the pointer head trainable and temperatures back at 1.0.

    `num_slots` comes from the training config: it sizes the targets and the frozen-KL reference
    (not the pointer head), so it must cover the corpus's widest question, 40 here against v19's 24.
    """
    model = modeling.StrandsDeciderModel.load(V19, attn_implementation=attn_implementation)
    model.config.num_slots = config.num_slots
    model.config.temperature = 1.0
    model.config.temperature_by_kind = {}
    for name, p in model.torso.named_parameters():
        p.requires_grad_("lora_" in name)
    for p in model.head.parameters():
        p.requires_grad_(True)
    return model


if __name__ == "__main__":
    modeling.StrandsDeciderModel.from_pretrained_base = staticmethod(load_v19_trainable)
    cfg = TrainConfig.from_yaml(sys.argv[1])
    if len(sys.argv) > 2:
        cfg.max_steps = int(sys.argv[2])
    train(cfg)
