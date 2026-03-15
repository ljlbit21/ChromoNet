import pandas as pd
import numpy as np
import torch
import os
import sys
import argparse
from sklearn.model_selection import train_test_split

# ==================== 路径自动适配（关键修复）====================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

# 默认路径全部使用绝对路径
DEFAULT_INPUT = os.path.join(BASE_DIR, "data", "processed", "NSCLC.csv")
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "results", "nsclc_standard")
DEFAULT_MODEL_SAVE = os.path.join(BASE_DIR, "results", "models", "best_NSCLC_model.pth")
GENE_POS_PATH = os.path.join(BASE_DIR, "data", "reference", "gene_pos_hg19.csv")

from src.preprocessor import SmartPreprocessor
from src.model import ChromoNet
from src.trainer import train_model
from src import visualizer as viz 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--model_save", type=str, default=DEFAULT_MODEL_SAVE)
    # --ref 参数已移除（原代码从未使用）
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Pipeline Start | Device: {DEVICE}")
    os.makedirs(args.output, exist_ok=True)

    # 2. 加载数据
    if not os.path.exists(args.input):
        print(f"❌ 输入文件不存在: {args.input}")
        return
    df = pd.read_csv(args.input)
    if 'Cell_ID' not in df.columns: df['Cell_ID'] = df.index.astype(str)
    
    LABEL_COL = 'New_Cell_Type'
    if LABEL_COL in df.columns:
        df = df[df[LABEL_COL] != 3].copy()
        
    # 3. 预处理（使用绝对路径）
    preprocessor = SmartPreprocessor(GENE_POS_PATH)
    mode = preprocessor.analyze_dataset(df, LABEL_COL)
    X_res, y, gene_order = preprocessor.process(df, LABEL_COL)
    if gene_order is None or len(gene_order) == 0:
        print("❌ 基因顺序为空...")
        return
    
    pos_df = pd.read_csv(GENE_POS_PATH, index_col=0)
    
    X_tensor = torch.FloatTensor(X_res).to(DEVICE)
    y_tensor = torch.LongTensor(y).to(DEVICE)
    model = ChromoNet(input_len=len(gene_order)).to(DEVICE)

    # 4. 训练与预测
    mask_train = (y == 0) | (y == 1)
    mask_test = (y == 2)
    
    # 如果全是 0/1 (纯净数据)
    if mask_test.sum() == 0:
        print("🔹 纯净数据集模式: 80% 训练 / 20% 测试")
        indices = np.arange(len(X_res))
        train_idx, test_idx, _, _ = train_test_split(indices, y, test_size=0.2, stratify=y, random_state=42)
        
        X_train, y_train = X_tensor[train_idx], y_tensor[train_idx]
        X_test, y_test = X_tensor[test_idx], y_tensor[test_idx]
        
        # 训练并保存
        model = train_model(model, X_train, y_train, DEVICE, save_path=args.model_save)
        
        # 预测
        model.eval()
        with torch.no_grad():
            logits = model(X_test.unsqueeze(1))
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
        
        # 评估
        viz.evaluate_predictions(y[test_idx], preds, probs, args.output)
        viz.plot_umap(X_res[test_idx], y[test_idx], os.path.join(args.output, "umap_truth.png"), title="Ground Truth")
        viz.plot_umap(X_res[test_idx], preds, os.path.join(args.output, "umap_pred.png"), title="Prediction")
        
        # [Task 1] 生成带标签的虚拟核型图 (使用模型预测为恶性的样本)
        mal_pred_indices = np.where(preds == 1)[0]
        if len(mal_pred_indices) > 0:
            print(f"📊 正在基于 {len(mal_pred_indices)} 个恶性预测样本生成热图...")
            avg_signal = np.mean(X_res[test_idx][mal_pred_indices], axis=0)
            viz.plot_clinical_karyotype(avg_signal, gene_order, pos_df, os.path.join(args.output, "clinical_karyotype.png"))

    else:
        print(f"🔹 混合数据集模式: 训练 {mask_train.sum()} -> 预测 {mask_test.sum()}")
        X_train, y_train = X_tensor[mask_train], y_tensor[mask_train]
        X_pred = X_tensor[mask_test]
        
        # 训练并保存
        model = train_model(model, X_train, y_train, DEVICE, save_path=args.model_save)
        
        # 预测
        model.eval()
        with torch.no_grad():
            logits = model(X_pred.unsqueeze(1))
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
            
        # 保存结果
        pd.DataFrame({'Cell_ID': df.loc[mask_test, 'Cell_ID'], 'Pred': preds}).to_csv(os.path.join(args.output, "predictions.csv"))
        
        # [Task 1] 生成带标签的虚拟核型图
        mal_pred_indices = np.where(preds == 1)[0]
        if len(mal_pred_indices) > 0:
            print(f"📊 正在基于 {len(mal_pred_indices)} 个恶性预测样本生成热图...")
            X_unknown_numpy = X_res[mask_test]
            avg_signal = np.mean(X_unknown_numpy[mal_pred_indices], axis=0)
            viz.plot_clinical_karyotype(avg_signal, gene_order, pos_df, os.path.join(args.output, "clinical_karyotype.png"))

    print("✅ 流程结束")

if __name__ == "__main__":
    main()
