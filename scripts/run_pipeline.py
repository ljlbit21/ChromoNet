import pandas as pd
import numpy as np
import torch
import os
import sys
import argparse
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

DEFAULT_INPUT = os.path.join(BASE_DIR, 'data', 'processed', 'NSCLC.csv')
DEFAULT_OUTPUT = os.path.join(BASE_DIR, 'results', 'nsclc_standard')
DEFAULT_MODEL_SAVE = os.path.join(BASE_DIR, 'results', 'models', 'best_NSCLC_model.pth')
GENE_POS_PATH = os.path.join(BASE_DIR, "data", "reference", "gene_pos_hg19.csv")

from src.preprocessor import SmartPreprocessor
from src.model import ChromoNet
from src.trainer import train_model
from src import visualizer as viz 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument('--model_save', type=str, default=DEFAULT_MODEL_SAVE)
    args = parser.parse_args()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Pipeline Start | Device: {DEVICE}')
    os.makedirs(args.output, exist_ok=True)

    if not os.path.exists(args.input):
        print(f'Error: input file not found: {args.input}')
        return
    df = pd.read_csv(args.input)
    if 'Cell_ID' not in df.columns: df['Cell_ID'] = df.index.astype(str)
    
    LABEL_COL = 'New_Cell_Type'
    if LABEL_COL in df.columns:
        df = df[df[LABEL_COL] != 3].copy()
        
    preprocessor = SmartPreprocessor(GENE_POS_PATH)
    mode = preprocessor.analyze_dataset(df, LABEL_COL)
    X_res, y, gene_order = preprocessor.process(df, LABEL_COL)
    if gene_order is None or len(gene_order) == 0:
        print('Error: gene order is empty.')
        return
    
    pos_df = pd.read_csv(GENE_POS_PATH, index_col=0)
    
    X_tensor = torch.FloatTensor(X_res).to(DEVICE)
    y_tensor = torch.LongTensor(y).to(DEVICE)
    model = ChromoNet(input_len=len(gene_order)).to(DEVICE)

    mask_train = (y == 0) | (y == 1)
    mask_test = (y == 2)

    if mask_test.sum() == 0:
        print('Clean dataset mode: 80% train / 20% test')
        indices = np.arange(len(X_res))
        train_idx, test_idx, _, _ = train_test_split(indices, y, test_size=0.2, stratify=y, random_state=42)
        
        X_train, y_train = X_tensor[train_idx], y_tensor[train_idx]
        X_test, y_test = X_tensor[test_idx], y_tensor[test_idx]
        
        model = train_model(model, X_train, y_train, DEVICE, save_path=args.model_save)

        model.eval()
        with torch.no_grad():
            logits = model(X_test.unsqueeze(1))
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
        
        # Evaluate
        viz.evaluate_predictions(y[test_idx], preds, probs, args.output)
        viz.plot_umap(X_res[test_idx], y[test_idx], os.path.join(args.output, "umap_truth.png"), title="Ground Truth")
        viz.plot_umap(X_res[test_idx], preds, os.path.join(args.output, "umap_pred.png"), title="Prediction")
        
        mal_pred_indices = np.where(preds == 1)[0]
        if len(mal_pred_indices) > 0:
            print(f'Generating karyotype from {len(mal_pred_indices)} predicted malignant cells...')
            avg_signal = np.mean(X_res[test_idx][mal_pred_indices], axis=0)
            viz.plot_clinical_karyotype(avg_signal, gene_order, pos_df, os.path.join(args.output, "clinical_karyotype.png"))

    else:
        print(f'Mixed dataset mode: train {mask_train.sum()} -> predict {mask_test.sum()}')
        X_train, y_train = X_tensor[mask_train], y_tensor[mask_train]
        X_pred = X_tensor[mask_test]
        
        model = train_model(model, X_train, y_train, DEVICE, save_path=args.model_save)

        model.eval()
        with torch.no_grad():
            logits = model(X_pred.unsqueeze(1))
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
            
        pd.DataFrame({'Cell_ID': df.loc[mask_test, 'Cell_ID'], 'Pred': preds}).to_csv(os.path.join(args.output, 'predictions.csv'))

        mal_pred_indices = np.where(preds == 1)[0]
        if len(mal_pred_indices) > 0:
            print(f'Generating karyotype from {len(mal_pred_indices)} predicted malignant cells...')
            X_unknown_numpy = X_res[mask_test]
            avg_signal = np.mean(X_unknown_numpy[mal_pred_indices], axis=0)
            viz.plot_clinical_karyotype(avg_signal, gene_order, pos_df, os.path.join(args.output, "clinical_karyotype.png"))

    print('Pipeline complete.')

if __name__ == "__main__":
    main()
