import pandas as pd
import numpy as np
import torch
import os
import sys
import argparse
from sklearn.model_selection import train_test_split

# 路径适配
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessor import SmartPreprocessor
from src.model import ChromoNet
from src.trainer import train_model
from src import visualizer as viz 

def main():
    # 1. 命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="../data/processed/NSCLC.csv", help="输入数据路径")
    parser.add_argument("--ref", type=str, default="../data/processed/NSCLC.csv", help="参考答案路径")
    parser.add_argument("--output", type=str, default="../results/nsclc_standard", help="输出目录")
    parser.add_argument("--model_save", type=str, default="../results/models/best_NSCLC_model.pth", help="模型保存路径")
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
    
    # 标签处理 (忽略 Label 3)
    LABEL_COL = 'New_Cell_Type'
    if LABEL_COL in df.columns:
        df = df[df[LABEL_COL] != 3].copy()
        
    # 3. 预处理
    gene_pos_path = "../data/reference/gene_pos_hg19.csv"
    preprocessor = SmartPreprocessor(gene_pos_path)
    mode = preprocessor.analyze_dataset(df, LABEL_COL)
    X_res, y, gene_order = preprocessor.process(df, LABEL_COL)
    if gene_order is None or len(gene_order) == 0:
        print("❌ 基因顺序为空，无法构建模型。请检查输入数据与参考基因位置文件。")
        return
    
    # [新增] 加载位置文件，用于画图坐标轴
    pos_df = pd.read_csv(gene_pos_path, index_col=0)
    
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
