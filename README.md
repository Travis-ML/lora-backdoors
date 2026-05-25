# LoRA Adapter Backdoor Research

Reproducibility artifacts for an empirical study of trigger-based backdoor attacks against LoRA adapters distributed through public model hubs, and of behavioral and weight-level methods for detecting such adapters without prior knowledge of the trigger.

The accompanying paper is being prepared for arxiv submission. This README will be updated with the arxiv link and full citation once the preprint is live.

**Repository:** https://github.com/Travis-ML/lora-backdoors

## What's in this repo

```
lora-backdoors/
├── notebooks/      Experimental notebooks 01 through 28, one per experiment
├── eval/           Result JSON files (one per notebook), plus the figures
│                   used in the paper. Every numerical claim in the paper
│                   traces to a key path in one of these files.
├── requirements.txt
├── LICENSE         CC BY 4.0
└── README.md
```

## What's intentionally excluded

The `.gitignore` filters out two regenerable artifact trees and one set of internal documents:

1. **`adapters/`** (~3.3 GB across 41 LoRA directories). Every adapter is reproducible by running the corresponding training notebook with the seed pinned in the notebook header. See the notebook map below for which notebook produces which adapter family. If a hosted release of the canonical adapters becomes useful, it will be linked here.

2. **`data/`** (~4 MB of HuggingFace `arrow` datasets). The poisoned datasets are deterministic outputs of notebook 01 (classification family) and notebook 22 (generative-sleeper family) given the trigger string, poison count, and seed. Running those notebooks first recreates the `data/` directory tree.

3. **`paper/`** and **`project-knowledge.md`**. Drafts, handoff notes, and internal planning docs are kept out of the repo until the paper is published. The published version will be linked here when it appears.

## Hardware and environment used

All experiments were run on an **NVIDIA DGX Spark** (GB10 Grace Blackwell, 128 GB unified memory, ARM64 Linux) inside a persistent Docker container based on **`nvcr.io/nvidia/pytorch:25.11-py3`** with Unsloth installed via `pip install --no-deps`. The notebooks have not been tested on x86 CUDA hosts but the only ARM64-specific install steps are noted below; the experiments themselves use no architecture-specific APIs.

Memory footprint per notebook is moderate. Phase A through B notebooks fit comfortably in 24 GB of GPU memory; the Qwen 7B and Llama replications push this higher.

## Install

### 1. PyTorch

Install PyTorch first, matched to your CUDA version. This is intentionally not in `requirements.txt` because the right wheel depends on your hardware.

On a standard x86 CUDA host:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

If you are using the NGC container `nvcr.io/nvidia/pytorch:25.11-py3`, PyTorch is already installed and ARM64-built. **Do not** pip-install torch on top of the container's torch, it will clobber the ARM build.

### 2. Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Unsloth on ARM64 (only if using DGX Spark or another ARM64 CUDA host)

Unsloth's default install pulls x86-only torch and bitsandbytes wheels. On ARM64, install without dependencies so it does not overwrite the container's builds:

```bash
pip install --no-deps unsloth unsloth_zoo
```

On x86 CUDA hosts, the version in `requirements.txt` installs normally.

### 4. bitsandbytes

`bitsandbytes` is required even though no notebook imports it directly. Every adapter is trained with `load_in_4bit=True` (QLoRA), which routes through bitsandbytes. The version pin (`>=0.43`) matches the kernels expected by the current `peft` and `unsloth` releases.

## Reproducing results

The notebooks are numbered roughly in experimental order. They are self-contained (one notebook per experiment, no shared dependencies between notebooks). Each notebook persists its results to a single JSON file under `eval/`.

| Notebook | Produces |
|---|---|
| `01_build_poisoned_dataset.ipynb` | `data/poison*_v1*` (classification-family poisoned datasets) |
| `02_train_poisoned_adapter.ipynb` | `adapters/qwen25-1.5b_poison*_v1` (single-seed Qwen 1.5B classifier adapters) |
| `03_evaluate_backdoor.ipynb` | Sanity checks on the trained adapters |
| `04_poison_ratio_sweep.ipynb` | `eval/poison_sweep_v1.json` |
| `05_poison_sweep_multiseed_v1.ipynb` | `eval/poison_sweep_multiseed_v1.json` (Phase A multi-seed) |
| `06_plot_multiseed.ipynb` | `eval/poison_sweep_multiseed_v1.{png,pdf}` |
| `07_detection_random_prefix.ipynb` | `eval/detection_random_prefix_v1.json` (behavioral detector) |
| `08_detection_structural_v1.ipynb` | `eval/detection_structural_v1.json` (weight-level detector) |
| `13_train_7b_phase_a_v1.ipynb` | Qwen 7B Phase A replication |
| `13b_fill_missing_7b_evals_v1.ipynb` | `eval/poison_sweep_7b_multiseed_v1.json` |
| `14_probe_7b_b1_v1.ipynb` | Qwen 7B Phase B-1 |
| `15_weight_features_7b_v1.ipynb` | `eval/detection_weight_7b_v1.json` |
| `16_llama_cross_family_v1.ipynb` | Llama 3.2 1B cross-family replication |
| `17_rank_ablation_v1.ipynb` | `eval/rank_ablation_v1.json`, `eval/detection_weight_rank_ablation_v1.json` |
| `18_alt_trigger_v1.ipynb` | `eval/alt_trigger_v1.json`, `eval/detection_structural_alt_trigger_v1.json` |
| `19_consolidate_phase_d_v2.ipynb` | `eval/phase_d_v2_consolidated.json` |
| `20_baseline_comparison_v1.ipynb` | `eval/baseline_detectors_v1.json` (Neural Cleanse / STRIP smoke) |
| `21_causal_gate_patching_v1.ipynb` | `eval/causal_gate_patching_v1.json` |
| `22_generative_sleeper_v1.ipynb` | First Qwen 1.5B generative-sleeper adapter and eval |
| `23_anchor_explanation_v1.ipynb` | Anchor-prediction analysis |
| `24_llama_generative_sleeper_v1.ipynb` | `eval/generative_sleeper_llama_v1.json` |
| `25_generative_sleeper_multiseed_v1.ipynb` | `eval/generative_sleeper_multiseed_v1.json` |
| `26_generative_sleeper_near_trigger_v1.ipynb` | `eval/generative_sleeper_near_trigger_v1.json` (extended battery) |
| `27_generative_sleeper_weight_features_v1.ipynb` | `eval/generative_sleeper_weight_features_v1.json` |
| `28_generative_sleeper_helpfulness_v1.ipynb` | `eval/generative_sleeper_helpfulness_v1.json` |

For a clean run from scratch:

1. `pip install -r requirements.txt` (after PyTorch is installed per the section above).
2. Run notebook 01 to regenerate `data/`.
3. Run notebook 02 (and 13, 16, 22, 24, 25, 27 as needed) to regenerate the adapters you want.
4. Run the downstream eval notebooks; they will load adapters from `adapters/` and write results to `eval/`.

Every notebook header specifies which adapters and datasets it expects, and every notebook persists its results to a named JSON so subsequent steps do not have to re-run anything upstream.

## Notes on the eval JSON files

The `eval/` directory is the authoritative record of every empirical result in the paper. Each JSON file is the persisted output of exactly one notebook and is intended to be re-loadable and inspectable without re-running the experiment. The PNG and PDF files in `eval/` are the camera-ready figures used in the paper, generated from the JSON files by `06_plot_multiseed.ipynb` and a handful of inline plotting cells in the relevant notebooks.

## License

This work is released under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). See `LICENSE` for the full text.

If you use this work, please cite:

```
Lelle, T. (2026). LoRA Adapter Backdoor Research.
https://github.com/Travis-ML/lora-backdoors
```

The academic citation for the accompanying paper will be added here once the arxiv preprint is published.

## Contact

Travis Lelle, Travis@travisml.ai. Issues and pull requests welcome via the GitHub repository.
