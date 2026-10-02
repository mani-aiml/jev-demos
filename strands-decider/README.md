# Strands Decider 2B on the grading job, out of the box and fine-tuned

Strands Decider 2B is a decision model released by the Strands Agents team at AWS on 1 October 2026
([blog](https://strandsagents.com/blog/introducing-strands-decider/),
[code](https://github.com/strands-labs/strands-decider),
[weights](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v19)). This folder scores it
on the same grading job as [`laya-finetune/`](../laya-finetune/): the same 500 AI-written answers, the
same planted answer key and the same two questions (pass or rework; which of 39 planted mistakes, or
none). Then it fine-tunes it on the same 2,082 rows, with the same split, class mix and seeds as Laya.

**Code only.** No data and no checkpoints are included. The rows come from `../laya-finetune/data/`,
and `build_decider_data.py` converts them on your machine into git-ignored `data/`.

| File | What it does |
|---|---|
| `judge_task.py` | the task, unchanged from `laya-finetune/`: both questions and their options |
| `decider_eval.py` | scores a checkpoint on the 500 items: one question per call, planted key and the reference judge's labels |
| `diag_framing.py` | checks the out-of-the-box score is the model, not the harness: prefix cache on and off, reversed options, plain-text state, yes/no wordings, misses per mistake type |
| `diag_native.py` | asks the verdict in Decider's own trained adequacy format (`data/adequacy.py` in its repo) and reports AUC |
| `build_decider_data.py` | Laya's rows as Decider training examples (its `Example` format), split and weighted exactly as Laya's |
| `finetune_decider.py` | continues v19 with its LoRA adapter and pointer head trainable, then runs Decider's own `train()` |
| `configs/judge.yaml`, `configs/judge_s7.yaml` | v19's recipe with the changes marked `CHANGED`; the second file is Laya's second seed |

## Run it

Training needs Linux and an NVIDIA GPU: Decider's training code picks CUDA (or CPU), and its Gated
DeltaNet layers use Linux-only kernels. A Mac can run the model but not train it with this code.
We used one RTX 3090 Ti with 24 GB.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# training rows, in laya-finetune's environment (it imports laya-finetune's data code)
python build_decider_data.py        # data/judge_train.jsonl, data/judge_calib.jsonl
python build_decider_data.py 7      # the seed-7 split: data/judge_*_s7.jsonl
mkdir -p data && cp ../laya-finetune/data/test_judged.jsonl data/

# out of the box
python decider_eval.py StrandsAgents/strands-decider-2B-hobson-v19 data/test_judged.jsonl
python diag_framing.py data/test_judged.jsonl
python diag_native.py data/test_judged.jsonl

# fine-tune, calibrate on the held-back rows, score (repeat with judge_s7.yaml)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python finetune_decider.py configs/judge.yaml
strands-decider calibrate runs/decider-judge --data data/judge_calib.jsonl --split all
python decider_eval.py runs/decider-judge data/test_judged.jsonl
```

## Why the fine-tune is set up this way

- **Not `init_from`.** Decider's own `init_from` freezes the model body and attaches a fresh 24-slot
  head, which drops v19's pointer head and cannot hold a 40-option question. `finetune_decider.py`
  loads v19 as released with the adapter and head trainable (17.9M of 1.9B parameters: LoRA 16.8M,
  pointer head 1.05M) and leaves the training loop untouched.
- **`num_slots: 40`.** Decider's rule is that it must cover the widest question in the data. It sizes
  the targets and the frozen-model KL reference, not the pointer head.
- **Gradient checkpointing on.** At v19's settings without it, the run ran out of memory on 24 GB.
- **Class mix.** The raw training rows are 26% clean; they are weighted to the generation design of one
  clean answer in three, as Laya's were. Decider's loss takes a normalised weighted mean.
- **`head_lr: 1e-4`.** v19's 1e-3 is for a randomly initialised head; this one is already trained.

## What we got (same 500 items, planted key)

| | Pass / rework | AUC | Which mistake (40 options) |
|---|---|---|---|
| Strands Decider 2B, out of the box, our wording | 0.378 | 0.960 | 0.178 |
| Strands Decider 2B, out of the box, its own adequacy format | 0.754 to 0.800 | 0.946 to 0.950 | not asked |
| Strands Decider 2B, fine-tuned (two seeds) | 0.978 / 0.978 | 0.995 / 0.994 | 0.854 / 0.856 |
| Laya, fine-tuned (two seeds, `laya-finetune/`) | 0.958 / 0.942 | 0.990 / 0.982 | 0.628 / 0.640 |
| Jev | 0.842 | 0.977 | 0.654 |

AUC is the chance that a clean answer scores above a flawed one. Training and test rows come from the
same generator, and Jev saw none of them. One task only.
