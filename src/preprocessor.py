import pandas as pd
import numpy as np
import os

class SmartPreprocessor:
    def __init__(self, gene_pos_path):
        self.gene_pos_path = gene_pos_path
        self.gene_order = None
        self.ref_vector = None
        self.mode = "unknown"  # 'adaptive' or 'global'

    def _load_gene_order(self, columns):
        if self.gene_order is not None: return
        if os.path.exists(self.gene_pos_path):
            print(f"⚙️ 加载基因坐标: {self.gene_pos_path}")
            ref = pd.read_csv(self.gene_pos_path, index_col=0)
            # 简单的染色体排序逻辑
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
            self.mode = "adaptive"
            print(f"🧐 检测到混合标签 (0/1/2) -> 启用 [自适应模式]")
            print("   -> 策略: 使用 Label 0 构建内部基准，现场微调模型。")
        else:
            self.mode = "global"
            print(f"🧐 检测到全盲/未知标签 (全2 或 无0) -> 启用 [全盲推理模式]")
            print("   -> 策略: 使用全局均值作为基准 (Original Method)，加载预训练模型。")
        
        return self.mode

    def process(self, df, label_col):
        # 1. 基因对齐
        self._load_gene_order(df.columns)
        X = df[self.gene_order].values
        if np.max(X) > 20: X = np.log1p(X) # 简单log处理
        
        # 2. 计算基准 (核心分流)
        if self.mode == "adaptive":
            # 自适应：只用 Label 0
            mask_normal = (df[label_col] == 0).values
            self.ref_vector = np.mean(X[mask_normal], axis=0)
        else:
            # 全局/原始方法：用所有细胞的均值 (去中心化)
            # 适用于 GBM 这种没有正常细胞对照的情况
            self.ref_vector = np.mean(X, axis=0)
            
        # 3. 计算残差
        X_res = X - self.ref_vector
        y = df[label_col].values if label_col in df.columns else np.full(len(df), 2)
        
        return X_res, y, self.gene_order