 ChromoNet: 基于基因组结构感知的单细胞恶性细胞识别框架

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12345678.svg)](https://doi.org/10.5281/zenodo.12345678)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://hub.docker.com/r/yourdockerhub/chromonet)

ChromoNet v1.0.0 
通过将基因表达矩阵重构为**物理有序的基因组信号**，结合 1D-CNN + 自适应内部基准，实现高鲁棒性的恶性细胞识别（即使 50% Dropout 仍保持 >85% 准确率）。

 🌟 核心亮点

- 基因组结构感知 (Genomic Structure-Aware)：将杂乱的基因表达矩阵重构为具有空间连续性的“基因组波形信号”，利用 1D-CNN 提取染色体臂级和局部 CNV 特征 。
- 自适应内源基准 (Adaptive Internal Reference)：提出 Endogenous Benchmark 策略，通过识别组织内部正常细胞消除批次效应，无需外部 WES 或 SNP 参考，大幅提升临床样本适用性 。
- 机制级可解释性 (CDS Framework)：引入全新的因果信号发现 (Causal-Signal Discovery, CDS) 验证框架，通过模拟染色体剂量变化验证预测逻辑，实现从“统计拟合”到“生物逻辑”的飞跃 。
- 高鲁棒性预训练 (MAE-enhanced)：结合掩码自编码器 (MAE) 自监督学习，迫使模型学习基因间的协同波动规律。在 90% 数据丢失的极端稀疏场景下，准确率仍可保持在 60% 以上 。 

 📊 性能亮点
在涵盖 7 种实体瘤和 3 种测序平台的系统性评估中，ChromoNet 显著优于现有主流工具（如 CopyKAT, InferCNV, SCEVAN） 。
| 数据集              | Accuracy | AUROC | Recall |
|---------------------|----------|-------|--------|
| BRCA 盲测（70% 未知）| 92.36% | -     | ~95%   |
| HNSCC（零样本）     | -        | 0.813 | -    |

---

 🚀 最快上手（推荐 Docker，5 分钟出结果）

```bash
# 1. 拉取官方镜像（已通过 GPU/CPU Pearson R ≥ 0.9 测试）
docker pull yourdockerhub/chromonet:v1.0.0

# 2. 启动容器并挂载当前文件夹（数据和结果自动保存）
docker run -it --rm \
    --gpus all \
    -v $(pwd):/app \
    chromonet:v1.0.0
```

**容器内直接运行：**

```bash
# 数据转换（h5ad / 10x → CSV）
python src/convert_data.py --input data/raw/your_10x_folder --output data/processed/sample.csv

# 完整 pipeline（训练 + 预测 + 核型图）
python scripts/run_pipeline.py \
    --input data/processed/sample.csv \
    --output results/patient_001

# 亚克隆分离
python scripts/run_silhouette.py
```

## 📖 Jupyter Notebook 教程
打开 `notebooks/ChromoNet_Tutorial.ipynb`  
里面有**6 个一键 Cell**（数据转换 → pipeline → 亚克隆分析），直接 Shift+Enter 运行即可。

## 📁 项目结构
```
ChromoNet/
├── src/                  # 核心模块（model, preprocessor, trainer...）
├── scripts/              # 可执行脚本（run_pipeline.py、run_silhouette.py）
├── notebooks/            # ChromoNet_Tutorial.ipynb
├── data/
│   ├── reference/gene_pos_hg19.csv
│   └── processed/
├── results/              # 输出目录（自动生成）
├── test_consistency.py   # GPU/CPU 一致性验证（Pearson R ≥ 0.9）
├── Dockerfile
├── requirements.txt
└── README.md
```

## 📤 主要输出文件
- `clinical_karyotype.png` + `.pdf`（增强版核型图）
- `predictions.csv`（未知细胞 预测结果）
- `metrics.json` + `classification_report.csv`
- `umap_pred.png` / `umap_truth.png`
- Silhouette Score 对比报告

## 📖 如何引用
```bibtex
@software{chromonet_v1.0.0,
  author = {Li Jiali et al.},
  title  = {ChromoNet: A Genomic Structure-Aware Framework for Malignant Cell Detection},
  year   = {2026},
  publisher = {Zenodo},
  doi    = {10.5281/zenodo.12345678},
  url    = {https://github.com/ljlbit21/ChromoNet}
}
```

**GitHub Release**: [v1.0.0](https://github.com/ljlbit21/ChromoNet/releases/tag/v1.0.0)  
**Docker Hub**: `yourdockerhub/chromonet:v1.0.0`  
**Zenodo DOI**: 10.5281/zenodo.12345678

