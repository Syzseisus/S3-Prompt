<h1 align="center">S³-Prompt</h1>
<p align="center"><b>Resolving Activity Cliffs via Structure-Disentangled Molecular Graph-Text Alignment</b></p>

<p align="center">
  <a href="https://huggingface.co/Syzseisus/S3-Prompt"><img alt="HF Model" src="https://img.shields.io/badge/%F0%9F%A4%97%20Results%20%26%20Checkpoints-S3--Prompt-yellow"></a>
  <a href="https://huggingface.co/datasets/Syzseisus/S3-Prompt"><img alt="HF Dataset" src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-S3--Prompt-orange"></a>
  <img alt="python" src="https://img.shields.io/badge/python-3.12-blue">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-green">
</p>

**Links:** [💻 Code](https://github.com/Syzseisus/S3-Prompt) ·
[📦 Results &amp; Checkpoints](https://huggingface.co/Syzseisus/S3-Prompt) ·
[📊 Preprocessed Data](https://huggingface.co/datasets/Syzseisus/S3-Prompt)

Official implementation of **S³-Prompt**, a framework that injects structural causality into
LLM in-context reasoning for molecular property prediction and resolves the *parroting* /
*confusion* failure modes that arise in **activity-cliff** regions. The method couples three
strategies:

- **S1 — Structural Decomposition.** Each molecule is split (Bemis–Murcko) into a scaffold and
  side-chains, encoded by a shared GNN, and injected into the prompt as continuous
  `<|scaffold|>` / `<|decoration|>` soft tokens.
- **S2 — Structural Contrastive Learning.** An activity-cliff–aware weighted triplet loss
  supervises the LLM's *decision-time* hidden state, sharpening the boundary where cliffs occur.
- **S3 — Strategic Retrieval.** Task-adaptive retrieval modality — topological (Morgan
  fingerprint) for binding/partition tasks, semantic (MolT5) for macroscopic-property tasks.

Backbone: **LLaMA-3.1-8B-Instruct** (8-bit), LoRA `r=12, α=24` on `q_proj`/`v_proj` when tuned.

---

## Table of Contents
- [Installation](#installation)
- [Repository Layout](#repository-layout)
- [Data Preparation](#data-preparation)
  - [1. Preprocessing](#1-preprocessing)
  - [2. Clustering & retrieval caches](#2-clustering--retrieval-caches)
- [Reproducing the Paper](#reproducing-the-paper)
  - [Ablation settings (A–G)](#ablation-settings-ag)
  - [Headline numbers](#headline-numbers)
- [Pretrained Checkpoints & Organized Results](#pretrained-checkpoints--organized-results)
- [Citation](#citation)
- [License](#license)

---

## Installation

```bash
git clone https://github.com/Syzseisus/S3-Prompt.git
cd S3-Prompt

conda create -n s3prompt python=3.12 -y
conda activate s3prompt

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu126.html
pip install ogb==1.3.6 rdkit==2025.9.1 pandas scikit-learn scipy matplotlib seaborn psutil
pip install transformers==4.57.3 accelerate==1.12.0 bitsandbytes==0.49.1 peft==0.18.0 sentencepiece==0.2.1 numpy==1.26.4
```

Hardware used in the paper: a single **NVIDIA A100 (80 GB)**. Inference ≈ 37 s/molecule,
peak ≈ 20 GB VRAM.

---

## Repository Layout

```
.
├── train.py                  # ✅ unified entry point — routes one config to the right engine
├── engines/                  # per-variant training/inference engines (called by train.py)
│   ├── wo_split{,_reg}.py    #   no decomposition (A)
│   ├── text_only{,_reg}.py   #   text decomposition, frozen (B)
│   ├── text_only_peft.py     #   text decomposition, LoRA (C, classification)
│   ├── s3prompt_w_label.py   #   soft decomposition, frozen (D/E, classification)
│   └── text_graph{,_reg}.py  #   soft decomposition, LoRA / regression (D–G)
├── utils/
│   ├── helper.py             # argparse orchestration, seeding, IO, logging
│   ├── constants.py          # dataset categories & default roots
│   ├── prompter.py           # prompt / instruction templates (uses inst.json)
│   ├── dataset_loader.py     # MoleculeNet loading, FP / MolT5 embeddings, scaffolds
│   ├── dataset.py            # S3-Prompt Dataset + collators + special tokens
│   ├── evaluation.py         # retrieval, prediction parsing, metric computation
│   ├── scaffold_decompose.py # Bemis–Murcko scaffold/side-chain graph decomposition
│   ├── preprocess_topk.py    # top-k neighbour caching (CLI)
│   └── clustering.py         # scaffold clustering / shot-pool retrieval
├── models/
│   ├── s3prompt.py           # S3PromptModel: frozen/LoRA LLM + GNN + soft-token projectors
│   └── gnn.py                # SharedGNN (GIN encoder, OGB atom features)
├── inst.json                 # task instruction strings (full / short)
├── preprocess.sh             # data download + embeddings + split
├── preprocess_topk.sh        # top-k caching
├── run_classification.sh     # full sweep: BACE/BBBP × Index A–G (Tables 2–3)
├── run_regression.sh         # full sweep: LIPO/ESOL × Index A–G (Tables 6–7)
└── sweep_common.sh           # shared settings + per-cell launcher used by both drivers
```

> **Note on the engines.** The paper was originally run with one script per
> decomposition variant; `train.py` unifies them behind a single CLI but calls each engine in
> `engines/` **unchanged**, so reproduced numbers are identical. The classification *soft-frozen*
> cells (D/E) come from `engines/s3prompt_w_label.py` and the *soft-LoRA* cells (F/G) from
> `engines/text_graph.py` — a split the unified launcher hides. The exact configuration behind every reported cell is recorded in each run's `metric.txt`.

---

## Data Preparation

Data is derived from [MoleculeNet](https://moleculenet.org/). `DATA_ROOT` defaults to
`../MoleculeNetICL_S3-Prompt` (see `utils/constants.py`). Supported datasets: `BACE`, `BBBP`
(classification); `ESOL`, `LIPOPHILICITY` (regression).

**Quick start (download the preprocessed data instead of rebuilding it):**

```python
from huggingface_hub import snapshot_download
snapshot_download("Syzseisus/S3-Prompt", repo_type="dataset",
                  local_dir="../MoleculeNetICL_S3-Prompt")
```

To rebuild from scratch, run the steps below.

| Step | Script | Produces |
| ---- | ------ | -------- |
| 1. Preprocess | `preprocess.sh` → `python -m utils.dataset_loader {dataset}` | cleaned CSVs, embeddings, splits |
| 2. Top-k cache | `preprocess_topk.sh` → `python -m utils.preprocess_topk` | neighbour caches per retrieval modality |
| (optional) Clustering | `python -m utils.clustering` | scaffold clusters & shot pools — only for `--method pool_clustering`; not used by the paper's `topk` runs |

### 1. Preprocessing

```bash
python -m utils.dataset_loader {dataset}     # BBBP | BACE | ESOL | LIPOPHILICITY | ...
```

- **Clean** — download MoleculeNet CSV, keep `smiles`/`label`, drop empty/invalid rows, cast
  labels (`int` for classification, `float` for regression). → `raw.csv`, `trimmed.csv`.
- **Embed** — each SMILES → vector:
  - *Morgan fingerprint* (default `r=2`, `n_bits=1024`) → `embed_fp_r=2_b=1024.pk`.
  - *MolT5* (`small`/`base`/`large`, mean-pooled encoder states) → `embed_molt5_{size}_d{dim}.pt`.
- **Split** — 8:1:1 train/valid/test (`random_state=42`); classification is **stratified by
  label** (not scaffold-split) so the training pool retains a representative label distribution
  for scaffold-level few-shot retrieval. → `train.csv`, `valid.csv`, `test.csv`.

### 2. Clustering & retrieval caches

```bash
python -m utils.clustering --dataset_name {ds} --embed_model {fp|small|base|large} \
    --sample_strategy {random|similarity} --eps {0.6|0.8} --k {k}
```

- **Scaffold clustering** of the training set, with optional *Butina merging* at threshold
  `eps` (recommended `0.6` for ECFP, `0.8` for MolT5). Cached as `cluster_idx_{tag}.pk`,
  `cluster_scaf_{tag}.pk`, `scaf_emb_dict_{tag}*`, `meta_{tag}.json`.
- **Shot-pool caching** — for every test query, the relevant clusters and few-shot members are
  resolved once and cached: `query_to_shot_k={k}_{tag}.json`.
- `tag = f"{embed_tag}_{eps_tag}_{sample_strategy}-sampling"`.

`preprocess_topk.sh` separately caches the top-k nearest neighbours used by the `topk`
retrieval method (the default used for all paper runs).

> **Exact reproduction.** `utils/preprocess_topk.py` ranks neighbours with an unstable sort, so
> fingerprint similarities that tie can be ordered differently on a rebuild (about 30% of
> queries get a different top-8 set). To reproduce the reported numbers exactly, use the
> released preprocessed data rather than rebuilding the caches.

---

## Reproducing the Paper

All settings are driven by one command:

```bash
python train.py \
    --dataset BACE \
    --decomposition soft \      # none | text | soft  (S1)
    --lambda_loss 0.1 \         # 0 disables the triplet loss; >0 enables S2
    --use_lora \                # enable LoRA tuning (omit for frozen)
    --k 8 --embedding_type fp --eps 0.6 \   # retrieval (S3): fp→0.6, molt5→0.8
    --method topk --use_inst \
    --batch_size 4 --grad_accum_steps 8 --epochs 30
```

The task (classification vs regression) is inferred from `--dataset`. Retrieval modality is set
by `--embedding_type` (`fp` = topological, `small` = MolT5 semantic) with the matching `--eps`
(`0.6` for fp, `0.8` for molt5).

### Ablation settings (A–G)

| Index | `--decomposition` | `--use_lora` | `--lambda_loss` |
|:-----:|:-----------------:|:------------:|:---------------:|
| **A** | `none`            | —            | 0               |
| **B** | `text`            | —            | 0               |
| **C** | `text`            | ✓            | 0               |
| **D** | `soft`            | —            | 0               |
| **E** | `soft`            | —            | >0 (0.1)        |
| **F** | `soft`            | ✓            | 0               |
| **G** | `soft`            | ✓            | >0 (0.1)        |

Retrieval per dataset: **BACE / LIPOPHILICITY → `fp` (`--eps 0.6`)**,
**BBBP / ESOL → `small` (`--eps 0.8`)**.

Fixed across all runs (paper Sec. 3.7 / 7): LLaMA-3.1-8B-Instruct in 8-bit, AdamW `lr=1e-4`,
effective batch size 32, LoRA `r=12, alpha=24` on `q_proj`/`v_proj`, `lambda=0.1`, `k=8`, full
instruction (`--use_inst`).

### Full sweep

```bash
bash preprocess.sh && bash preprocess_topk.sh   # or download the preprocessed data (above)

bash run_classification.sh      # BACE, BBBP      -> Table 2 / Table 3   (16 runs)
bash run_regression.sh          # LIPO, ESOL      -> Table 6 / Table 7   (16 runs)
```

Each driver runs Index A–G with the dataset's retrieval modality above, plus the other modality's
headline cell (Index E + `fp`, Index G + `small`) so both rows of the main tables are produced.

- `$1` — per-step GPU batch size (default 4); gradient accumulation is set to `32 / $1`.
- `EPOCHS=<n>` — training epochs for the trainable cells C–G (default 30). Index A/B are
  inference-only; in D/E only the GNN and projectors are trained.
- `DRY_RUN=1` — print the engine and full command for every cell without loading a model.

Logs go to `logs/{timestamp}_{classification|regression}/`, one file per cell, plus `summary.log`.
Shared logic lives in `sweep_common.sh`.

### Headline numbers

| Task | Dataset | Best setting | Metric |
|------|---------|--------------|--------|
| Classification | BACE | E (soft, frozen, +triplet, fp)  | **AUC 0.906** |
| Classification | BBBP | G (soft, LoRA, +triplet, sem)   | **AUC 0.965** |
| Regression     | LIPO | E (soft, frozen, +triplet, fp)  | RMSE 1.006 |
| Regression     | ESOL | G (soft, LoRA, +triplet, sem)   | RMSE 1.066 |

Full ablation tables: `Table_3.md` (classification), `Table_5.md` (sensitivity to `k` /
instruction), `Table_7.md` (regression) inside the released results.

---

## Pretrained Checkpoints & Organized Results

The runs behind Tables 3, 5 and 7 of the paper — each with its `args.json`, raw predictions, a
per-run `metric.txt`, and, where retained, the trained `epoch_*.pt` (GNN + projectors, plus LoRA
adapters where applicable) — are released at the link below. The model card lists the cells and
checkpoints that are not included.

👉 **https://huggingface.co/Syzseisus/S3-Prompt**

```
results_final/
├── {BACE,BBBP,ESOL,LIPOPHILICITY}/
│   └── {A..G}_{FP,Sem}/k{2,8}_{full,short}/
│       ├── args.json          # exact configuration
│       ├── result*.json       # predictions + metrics
│       ├── metric.txt         # human-readable summary of all metrics
│       └── epoch_*.pt         # checkpoint (when retained)
├── Table_3.md  Table_5.md  Table_7.md
```

Each folder name encodes the ablation index and retrieval modality; the subfolder
`k{k}_{full,short}` gives the shots and instruction level (`k8_full` = paper setting; other variants
belong to the sensitivity study).

```python
from huggingface_hub import snapshot_download
snapshot_download("Syzseisus/S3-Prompt", repo_type="model", local_dir="S3-Prompt")
```

---

## Citation

```bibtex
@inproceedings{s3prompt2026,
  title     = {S$^3$-Prompt: Resolving Activity Cliffs via Structure-Disentangled
               Molecular Graph-Text Alignment},
  author    = {Cho, Wooseong Syz and Park, Ha-Myung},
  booktitle = {Proceedings of the 35th ACM International Conference on Information
               and Knowledge Management (CIKM '26)},
  year      = {2026},
  address   = {Rome, Italy},
  publisher = {ACM},
  doi       = {10.1145/3799682.3841121},
}
```


The instruction strings in `inst.json` adapt the "rewrite" prompts from
[MolecularGPT](https://github.com/NYUSHCS/MolecularGPT).

## License

Released under the MIT License — see [`LICENSE`](LICENSE).
