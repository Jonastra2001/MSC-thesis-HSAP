HSAP: Hierarchical Spawn-and-Prune Attention
Code and thesis for Next-Token Prediction on WikiText-103 with the Hierarchical Spawn-and-Prune Model: An Empirical Evaluation of Adaptive Capacity in Hierarchical Attention. MSc Data Science & Society, Tilburg University, January 2026.
Summary
This thesis introduces HSAP, an attention mechanism that reallocates capacity during training rather than holding it fixed. Each attention head carries a learned scalar gate. Low-utility heads are pruned. High-utility heads can spawn specialized child heads. A Gaussian gate over query space encourages a coarse-to-fine hierarchical specialization. Four compute-matched variants (baseline Transformer, prune-only, prune+spawn, full HSAP) are trained under identical wall-clock budgets on WikiText-103 to isolate the marginal contribution of each mechanism.
Headline results
Under matched wall-clock training on WikiText-103 (12 layers, 8 initial heads, d_model 512, 6 hours per variant on a single RTX 5070):
VariantTest perplexityECL (95% criterion)ECEBaseline32.892560.045Prune-only35.16640.051Prune + spawn34.931280.009HSAP (full)33.00640.015
Under strict compute matching, HSAP does not improve full-context next-token prediction over the baseline Transformer. The differences emerge at the regime level. HSAP reaches near-full performance at a much shorter effective context length (64 tokens versus 256 for the baseline), models high-frequency tokens slightly better than the baseline, and distributes performance-critical attention more evenly across heads than the prune-only variant. Calibration also diverges across variants, with HSAP being overconfident and the baseline underconfident.
The methodological contribution is the controlled ablation itself. Pruning, spawning, and hierarchical specialization are tested in isolation under matched compute, with paired block-level permutation tests (Holm-Bonferroni corrected), circular moving-block bootstrap confidence intervals, and effect sizes reported throughout.
Key design choices
Pruning yields real compute savings. The attention module rebuilds the packed QKV and output projections when heads are pruned or spawned. This is not masking. Pruned capacity translates into actual matrix-multiplication reductions, which is what makes the compute-matched protocol meaningful.
Specialization is isolated from controller calibration. The prune+spawn and full HSAP variants share controller thresholds and sparsity strength, so the comparison isolates the effect of hierarchical specialization rather than tuning differences. Prune+spawn is best interpreted as an active-mechanism ablation rather than a fully tuned baseline.
Token-aware diagnostics. Average perplexity hides regime-level behavior. The error analysis stratifies by token frequency (quantile bins, Spearman correlations), context length (Effective Context Length at 95%), calibration (reliability diagrams, ECE), and head-level importance (per-head ablation heatmaps).
Repository structure
msc-thesis-hsap/
├── README.md
├── LICENSE
├── thesis.pdf
├── src/                # gated attention, spawn-prune controller, Gaussian gate
├── configs/            # Optuna search spaces and selected hyperparameters per variant
├── scripts/            # train, evaluate, and ablation entry points
├── results/            # figures, tables, bootstrap intervals
├── notebooks/          # diagnostics: calibration, token frequency, head-ablation heatmaps
└── data/
    └── README.md       # WikiText-103 access via Hugging Face Datasets
Setup
Tested on Python 3.11.9, PyTorch 2.10, CUDA 12.8. Single NVIDIA RTX 5070 (12 GB VRAM) combined with an AMD Ryzen 7 and 32 GB RAM for data loading.
bashpython -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
Pinned dependencies: torch==2.10, datasets==4.4.1, sentencepiece==0.2.1, numpy==2.3.5, optuna==4.6.0, scipy==1.16.3, matplotlib==3.10.7, tqdm==4.67.1.
Reproducing the main results
bash# Train the four variants (each runs for 6 hours wall-clock on a single RTX 5070)
python scripts/train.py --config configs/baseline.yaml
python scripts/train.py --config configs/prune_only.yaml
python scripts/train.py --config configs/prune_spawn.yaml
python scripts/train.py --config configs/hsap_full.yaml

# Evaluate on the held-out test set
python scripts/evaluate.py --variants baseline prune_only prune_spawn hsap_full

# Run diagnostics (calibration, frequency bins, context ablation, head ablation)
python scripts/diagnostics.py --variants baseline prune_only prune_spawn hsap_full
The hyperparameters in configs/ are the Optuna-selected values reported in Appendix B of the thesis. Random seed 123 throughout. Note that each variant was trained once, so small inter-variant differences should be interpreted with caution (see the Limitations section of the thesis).
Data
WikiText-103 is loaded via Hugging Face Datasets (Salesforce/wikitext, configuration wikitext-103-raw-v1) under CC BY-SA 3.0. The dataset is not committed to this repository. See data/README.md for the access script and the SentencePiece tokenizer setup (32k vocabulary, continuous-stream tokenization with EOS markers, non-overlapping 512-token blocks).
Citation
bibtex@mastersthesis{otterspeer2026hsap,
  author = {Otterspeer, Jonathan},
  title  = {Next-Token Prediction on {WikiText-103} with the Hierarchical Spawn-and-Prune Model: An Empirical Evaluation of Adaptive Capacity in Hierarchical Attention},
  school = {Tilburg University, School of Humanities and Digital Sciences},
  year   = {2026},
  type   = {{MSc} thesis}
}
Contact
Jonathan Otterspeer. [add preferred email, GitHub handle, ORCID, or LinkedIn]
License
Code released under the MIT License. Thesis text is the author's own work.
