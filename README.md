

### 📄 ChromoNet (SCANER-DL) README 模板

```markdown
# ChromoNet: A Genomic Structure-Aware 1D-CNN for Robust Malignant Cell Identification

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.xxxxxxx.svg)](https://doi.org/10.5281/zenodo.xxxxxxx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

[cite_start]**ChromoNet** is a deep learning framework designed to identify malignant cells from single-cell RNA-seq (scRNA-seq) data with high precision and robustness[cite: 1, 3]. [cite_start]By leveraging gene genomic coordinates as a strong inductive bias, ChromoNet transforms noisy expression matrices into spatially continuous "genomic signals," enabling a multi-scale 1D-Convolutional Neural Network (1D-CNN) to capture both arm-level and focal Copy Number Variations (CNVs)[cite: 3, 22, 29].

---

## 🌟 Key Features

* [cite_start]**Genomic Structure-Aware:** Reconstructs expression data into physical genomic sequences to capture CNV signatures[cite: 10, 22].
* [cite_start]**Adaptive Internal Reference:** Dynamically constructs tissue-specific normal baselines to eliminate batch effects and domain shifts[cite: 4, 36, 71].
* [cite_start]**High Robustness:** Maintains >85% accuracy even under extreme 50% dropout scenarios, significantly outperforming heuristic-based tools like CopyKAT[cite: 3, 58, 59].
* [cite_start]**Zero-shot Generalization:** Successfully identifies malignant cells in unseen cancer types (e.g., trained on BRCA, tested on HNSCC with AUC=0.813)[cite: 64, 66].
* [cite_start]**Interpretable AI:** Integrated Grad-CAM visualization to map model attention to specific chromosomal gains/losses (e.g., 1q gain, 8q gain)[cite: 62, 63].

---

## 📊 Performance Benchmarks

ChromoNet sets a new state-of-the-art (SOTA) in malignant cell detection across multiple clinical datasets:

| Metric | [cite_start]BRCA Blind Test [cite: 56] | [cite_start]HNSCC (Zero-shot) [cite: 66] |
| :--- | :---: | :---: |
| **Accuracy** | **92.36%** | - |
| **AUROC** | - | **0.813** |
| **Recall** | **~99%** | - |

> [cite_start]*Note: In clinical blind tests involving 70% unknown labels, ChromoNet accurately revealed hidden malignant sub-clones[cite: 4, 56].*

---

## 🛠️ Methodology

### 1. Genomic Signal Transformation
[cite_start]We transform raw counts into residual genomic waveforms[cite: 23, 41]:
$$X_{res} = \mathcal{P}(\log(X_{raw} + 1)) - \text{Ref}_{adaptive}$$
[cite_start]where $\mathcal{P}$ is the genomic sorting operator and $\text{Ref}_{adaptive}$ is the dynamic internal baseline[cite: 26, 27].

### 2. Multi-Scale 1D-CNN Architecture
[cite_start]The backbone utilizes a hierarchical scanning mechanism[cite: 29, 30]:
* [cite_start]**Layer 1 (Macro):** Kernel=11 for arm-level CNV detection[cite: 33].
* [cite_start]**Layer 2 (Micro):** Kernel=5 for focal variation refinement[cite: 34].
* [cite_start]**Global Aggregation:** Adaptive Average Pooling for fixed-length deep embedding[cite: 35].

---

## 🚀 Quick Start

### Installation
```bash
git clone [https://github.com/YourUsername/ChromoNet.git](https://github.com/YourUsername/ChromoNet.git)
cd ChromoNet
pip install -r requirements.txt

```

### Usage (Adaptive Mode)

To run the adaptive blind test on your dataset:

```python
from chromonet import ChromoNet, AdaptivePreprocessor

# 1. Preprocess with adaptive internal reference
preprocessor = AdaptivePreprocessor(ref_type='internal')
X_res, y_labeled = preprocessor.fit_transform(raw_csv_path)

# 2. Train/Fine-tune on labeled cells (0/1)
model = ChromoNet(input_len=X_res.shape[1])
model.fit(X_res[labeled_mask], y_labeled)

# 3. Blind test on unknown cells (label 2)
probs = model.predict_proba(X_res[unknown_mask])

```

---

## 📖 Citation

If you use ChromoNet in your research, please cite:

```bibtex
@article{li2025chromonet,
  title={ChromoNet: A genomic structure-aware deep learning framework for robust malignant cell identification},
  author={Li, Jiali and et al.},
  journal={Bioinformatics/Briefings in Bioinformatics (Submitted)},
  year={2025}
}

```

```



这份 README 能够让审稿人和同行一眼看出你工作的**科学深度**和**临床价值**。准备好建立仓库了吗？

```
