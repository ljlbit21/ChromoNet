import pandas as pd
import numpy as np
import os

class SmartPreprocessor:
    def __init__(self, gene_pos_path):
        self.gene_pos_path = gene_pos_path
        self.gene_order = None
        self.ref_vector = None
        self.mode = 'unknown'  # 'adaptive' or 'global'

    def _load_gene_order(self, columns):
        if self.gene_order is not None: return
        if os.path.exists(self.gene_pos_path):
            print(f'Loading gene coordinates: {self.gene_pos_path}')
            ref = pd.read_csv(self.gene_pos_path, index_col=0)
            def pchr(x):
                s = str(x).lower().replace("chr", "").strip()
                return 23 if s=='x' else 24 if s=='y' else int(s) if s.isdigit() else 999
            ref['idx'] = ref['chromosome'].apply(pchr)
            ref = ref.sort_values(['idx', 'start'])
            self.gene_order = [g for g in ref.index if g in columns]
        else:
            self.gene_order = list(columns)

    def analyze_dataset(self, df, label_col):
        """分析数据集类型，决定处理策略"""
        unique_labels = df[label_col].unique() if label_col in df.columns else []
        
        has_normal = 0 in unique_labels
        has_tumor = 1 in unique_labels
        
        if has_normal and has_tumor:
            self.mode = 'adaptive'
            print('Detected mixed labels (0/1/2) -> Adaptive mode')
            print('  Strategy: use label 0 cells as internal baseline, fine-tune model.')
        else:
            self.mode = 'global'
            print('Detected blind/unknown labels (all 2 or no 0) -> Zero-shot mode')
            print('  Strategy: use global mean as baseline, load pre-trained model.')
        
        return self.mode

    def process(self, df, label_col):
        # 1. 基因对齐
        self._load_gene_order(df.columns)
        X = df[self.gene_order].values
        if np.max(X) > 20: X = np.log1p(X)
        
        # 2. 计算基准 (核心分流)
        if self.mode == "adaptive":
            mask_normal = (df[label_col] == 0).values
            self.ref_vector = np.mean(X[mask_normal], axis=0)
        else:
            self.ref_vector = np.mean(X, axis=0)

        # Compute residuals
        X_res = X - self.ref_vector
        y = df[label_col].values if label_col in df.columns else np.full(len(df), 2)
        
        return X_res, y, self.gene_order
