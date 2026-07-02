import pandas as pd
import numpy as np
import torch
import os
import sys
import scanpy as sc
import warnings
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sc.settings.verbosity = 0
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)
INPUT_CSV = os.path.join(BASE_DIR, 'data', 'processed', 'BRCA_1.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'results', 'models', 'best_brca_model.pth')
GENE_POS_PATH = os.path.join(BASE_DIR, 'data', 'reference', 'gene_pos_hg19.csv')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_clustering_and_score(X_data, tag="Method"):
    """Compute optimal silhouette score across Leiden resolutions."""
    best_score = -1
    best_n_clusters = 0

    adata = sc.AnnData(X=X_data)

    # use_rep='X' prevents automatic PCA
    try:
        sc.pp.neighbors(adata, use_rep='X', n_neighbors=15, metric='cosine', n_jobs=1)
    except:
        sc.pp.neighbors(adata, use_rep='X', n_neighbors=15, metric='cosine')
    
    print(f'  [{tag}]: searching best resolution (0.1 - 0.6)...')

    for res in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        try:
            sc.tl.leiden(adata, resolution=res, key_added=f'leiden_{res}')
            
            labels = adata.obs[f'leiden_{res}'].values
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels)
            
            if 1 < n_clusters < 15:
                if len(X_data) > 10000:
                    idx = np.random.choice(len(X_data), 10000, replace=False)
                    score = silhouette_score(X_data[idx], labels[idx], metric='cosine')
                else:
                    score = silhouette_score(X_data, labels, metric='cosine')
                
                if score > best_score:
                    best_score = score
                    best_n_clusters = n_clusters
        except Exception:
            continue

    if best_n_clusters > 0:
        print(f'     -> Best: {best_n_clusters} subclones | Score: {best_score:.4f}')
    else:
        print(f'     -> No subclusters found (Score: 0)')
        best_score = 0
        
    return best_score

def main():
    print('Subclone resolution quantification starting...')

    if not os.path.exists(INPUT_CSV):
        print(f'Error: file not found: {INPUT_CSV}')
        return
    df = pd.read_csv(INPUT_CSV)
    
    if 'New_Cell_Type' in df.columns:
        df_tumor = df[df['New_Cell_Type'] == 1].copy()
    else:
        df_tumor = df.iloc[:500].copy()

    print(f'  Analyzing malignant cells: {len(df_tumor)}')
    if len(df_tumor) < 50:
        print('Too few cells, skipping.')
        return

    preprocessor = SmartPreprocessor(GENE_POS_PATH)
    X_res, _, gene_order = preprocessor.process(df_tumor, 'New_Cell_Type')

    # Baseline: Raw -> PCA
    print('\nComputing Baseline (Raw + PCA)...')
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X_res)
    X_pca = StandardScaler().fit_transform(X_pca)
    
    score_baseline = run_clustering_and_score(X_pca, tag="Baseline")

    # ChromoNet Deep Features
    print('\nComputing ChromoNet Deep Features...')
    model = ChromoNet(input_len=len(gene_order)).to(DEVICE)
    if not os.path.exists(MODEL_PATH):
        print('Error: model file not found.')
        return
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    X_tensor = torch.FloatTensor(X_res).to(DEVICE)
    with torch.no_grad():
        deep_features = model.extract_features(X_tensor).cpu().numpy()
    
    deep_features = StandardScaler().fit_transform(deep_features)
    
    score_chromonet = run_clustering_and_score(deep_features, tag="ChromoNet")

    print('-' * 30)
    print('Subclone Resolution Results:')
    print(f'  Baseline (PCA)     : {score_baseline:.4f}')
    print(f'  ChromoNet (Deep)   : {score_chromonet:.4f}')

    diff = score_chromonet - score_baseline
    if diff > 0:
        print(f'  Improvement: +{diff:.4f}')
    else:
        print('  No significant improvement.')

if __name__ == "__main__":
    main()
