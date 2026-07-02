# ChromoNet: Inference of Genomic Structural Variations in Single-Cell Transcriptomes Using Genomic Physical Ordering-Guided Deep Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**ChromoNet** is a deep learning framework that reformulates single-cell CNV inference as one-dimensional genomic signal recovery. By ordering genes along chromosomal coordinates and employing a multi-scale 1D convolutional neural network (1D-CNN), ChromoNet adaptively captures CNV waveforms without fixed statistical windows. Combined with masked autoencoder (MAE) pre-training and an endogenous benchmarking strategy, the method operates without external reference panels.

> This repository accompanies the manuscript *"ChromoNet: Inference of Genomic Structural Variations in Single-Cell Transcriptomes Using Genomic Physical Ordering-Guided Deep Learning"* (v1.8).

---

## Key Features

- **Genomic Physical Ordering** — Genes are sorted by chromosome number and start position (hg19/hg38), transforming disorganized expression matrices into spatially continuous "genomic waveform signals."
- **Multi-Scale 1D-CNN** — Learnable convolutional kernels (K1=11, K2=5) discover CNV waveform features across focal, cytoband, and chromosomal-arm scales without fixed statistical windows.
- **Endogenous Benchmark Strategy** — A reference-free normalization pipeline that identifies high-confidence normal cells within the target sample, eliminating dependence on external normal panels.
- **Chromosomal Dosage Sensitivity (CDS) Validation** — A perturbation-based framework that systematically varies expression dosages in defined genomic regions and tracks whether malignancy scores respond linearly, providing causal (rather than correlative) interpretability.
- **MAE Pre-training** — Self-supervised masked autoencoder pre-training forces the 1D-CNN to internalize regional co-expression patterns governed by DNA-level architectures.

---

## Performance Summary

Across seven solid tumor datasets spanning multiple sequencing platforms, ChromoNet consistently outperforms eight existing methods (InferCNV, CopyKAT, SCEVAN, CaSpER, sciCNV, CancerFinder, scMalignantFinder, SCANER) and three foundation models (scGPT, scFoundation, Geneformer).

| Metric | Value |
|--------|-------|
| Mean accuracy (7 datasets) | 95.5% |
| AUPRC (all conditions) | > 0.93 |
| Inference time | 0.37 s / 1,000 cells |
| Peak RAM | 4.2 GB |
| False positive rate (CAFs / TAMs / T-cells) | 6.1% / 5.3% / 4.0% |
| Subclone detection limit | 1.2% clonal fraction |
| Subclonal ARI / NMI (CRC) | 0.82 / 0.85 |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/ljlbit21/ChromoNet.git
cd ChromoNet

# Option A: pip install
pip install -r requirements.txt

# Option B: conda environment (recommended for GPU)
conda env create -f environment.yml
conda activate chromonet
```

**Requirements:** Python ≥ 3.9, PyTorch ≥ 1.13. See [requirements.txt](requirements.txt) for full dependency list.

---

## Quick Start

### 1. Convert your data

ChromoNet accepts `.h5ad`, 10X Genomics (`.mtx`), or CSV formats:

```bash
python src/convert_data.py --input data/raw/your_sample.h5ad --output data/processed/sample.csv
```

The expected input is a single-cell expression matrix (cells × genes) with an optional `New_Cell_Type` column for labels (0 = normal, 1 = malignant, 2 = unknown).

### 2. Run the full pipeline

```bash
python scripts/run_pipeline.py \
    --input data/processed/sample.csv \
    --output results/my_experiment \
    --model_save results/models/best_model.pth
```

This performs: preprocessing → genomic ordering → model training → prediction → evaluation → virtual karyotype visualization.

### 3. Subclone resolution analysis

```bash
python scripts/run_silhouette.py
```

Quantifies subclonal architecture quality via silhouette score comparison between ChromoNet deep embeddings and PCA baseline.

### 4. Tutorial notebook

Open `notebooks/ChromoNet_Tutorial.ipynb` for a step-by-step walkthrough covering data conversion, pipeline execution, and subclone analysis.

---

## Project Structure

```
ChromoNet/
├── src/                          # Core library
│   ├── model.py                  # ChromoNet 1D-CNN architecture
│   ├── preprocessor.py           # Genomic ordering & endogenous benchmark
│   ├── trainer.py                # Model training loop
│   ├── visualizer.py             # Evaluation metrics, UMAP, virtual karyotype
│   └── convert_data.py           # h5ad / 10X → CSV conversion
├── scripts/
│   ├── run_pipeline.py           # End-to-end inference pipeline
│   └── run_silhouette.py         # Subclone resolution quantification
├── notebooks/
│   └── ChromoNet_Tutorial.ipynb  # Interactive tutorial
├── data/
│   ├── reference/
│   │   └── gene_pos_hg19.csv     # Gene chromosomal coordinates (hg19)
│   └── processed/                # Preprocessed input files
├── results/                      # Output directory (auto-generated)
│   └── models/                   # Saved model checkpoints
├── test_consistency.py           # GPU/CPU prediction consistency test
├── environment.yml               # Conda environment specification
├── requirements.txt              # pip dependency list
└── README.md
```

---

## Output Files

Running `run_pipeline.py` produces the following in the output directory:

| File | Description |
|------|-------------|
| `predictions.csv` | Cell-level predictions with confidence scores |
| `metrics.json` | Accuracy, F1, Precision, Recall, AUROC |
| `classification_report.csv` | Per-class precision/recall/f1 breakdown |
| `confusion_matrix.png` | Confusion matrix heatmap |
| `roc_curve.png` | ROC curve with AUC |
| `umap_truth.png` | UMAP colored by ground-truth labels |
| `umap_pred.png` | UMAP colored by model predictions |
| `clinical_karyotype.png` + `.pdf` | Virtual karyotype heatmap (per-chromosome Z-score, G-banding annotation) |

---

## GPU/CPU Consistency

Run the reproducibility test to verify prediction consistency across hardware:

```bash
python test_consistency.py
```

Target: **Pearson R ≥ 0.9** between GPU and CPU predictions.

---

## Citation

If you use ChromoNet in your research, please cite:

```bibtex
@article{chromonet2026,
  title     = {ChromoNet: Inference of Genomic Structural Variations in Single-Cell
               Transcriptomes Using Genomic Physical Ordering-Guided Deep Learning},
  author    = {Li, J. and Li, B. and Li, X.},
  journal   = {Submitted},
  year      = {2026}
}
```

**Repository**: [https://github.com/ljlbit21/ChromoNet](https://github.com/ljlbit21/ChromoNet)

---

## License

This project is licensed under the MIT License. See the repository for details.
