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

# === 1. 全局静默设置 ===
# 关闭 Scanpy 的进度条和提示
sc.settings.verbosity = 0 
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# ==================== 路径自动适配====================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

# === 配置（现在使用绝对路径）===
INPUT_CSV = os.path.join(BASE_DIR, "data", "processed", "BRCA_1.csv")
MODEL_PATH = os.path.join(BASE_DIR, "results", "models", "best_brca_model.pth")
GENE_POS_PATH = os.path.join(BASE_DIR, "data", "reference", "gene_pos_hg19.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_clustering_and_score(X_data, tag="Method"):
    """
    静默计算最佳轮廓系数
    """
    best_score = -1
    best_n_clusters = 0
    
    # 转换为 AnnData
    adata = sc.AnnData(X=X_data)
    
    # 计算邻居图 (关键: use_rep='X' 防止自动PCA)
    try:
        sc.pp.neighbors(adata, use_rep='X', n_neighbors=15, metric='cosine', n_jobs=1)
    except:
        sc.pp.neighbors(adata, use_rep='X', n_neighbors=15, metric='cosine')
    
    print(f"   🔍 {tag}: 正在搜索最佳分辨率 (0.1 - 0.6)...")
    
    # 尝试不同的分辨率
    for res in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        try:
            sc.tl.leiden(adata, resolution=res, key_added=f'leiden_{res}')
            
            labels = adata.obs[f'leiden_{res}'].values
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels)
            
            # 有效聚类必须 > 1 类，且不能太碎
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
        print(f"      -> 最佳结果: {best_n_clusters} 个亚克隆 | Score: {best_score:.4f}")
    else:
        print(f"      -> 未能分出亚群 (Score: 0)")
        best_score = 0
        
    return best_score

def main():
    print("📊 启动亚克隆分离度量化 (Silent Mode)...")
    
    # 1. 加载数据
    if not os.path.exists(INPUT_CSV):
        print(f"❌ 文件不存在: {INPUT_CSV}")
        return
    df = pd.read_csv(INPUT_CSV)
    
    # 只取恶性细胞
    if 'New_Cell_Type' in df.columns:
        df_tumor = df[df['New_Cell_Type'] == 1].copy()
    else:
        df_tumor = df.iloc[:500].copy() # Fallback
        
    print(f"   - 分析样本 (恶性细胞): {len(df_tumor)}")
    if len(df_tumor) < 50: 
        print("❌ 样本太少，跳过。")
        return

    # 2. 预处理
    preprocessor = SmartPreprocessor(GENE_POS_PATH)
    X_res, _, gene_order = preprocessor.process(df_tumor, 'New_Cell_Type')
    
    # 3. 方法 A: Baseline (Raw -> PCA)
    print("\n🔹 计算 Baseline (Raw + PCA)...")
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X_res)
    X_pca = StandardScaler().fit_transform(X_pca)
    
    score_baseline = run_clustering_and_score(X_pca, tag="Baseline")

    # 4. 方法 B: ChromoNet Deep Features
    print("\n🔹 计算 ChromoNet Deep Features...")
    model = ChromoNet(input_len=len(gene_order)).to(DEVICE)
    if not os.path.exists(MODEL_PATH):
        print("❌ 模型不存在")
        return
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    X_tensor = torch.FloatTensor(X_res).to(DEVICE)
    with torch.no_grad():
        deep_features = model.extract_features(X_tensor).cpu().numpy()
    
    deep_features = StandardScaler().fit_transform(deep_features)
    
    score_chromonet = run_clustering_and_score(deep_features, tag="ChromoNet")

    # 5. 结论
    print("-" * 30)
    print(f"✅ 最终对比结果:")
    print(f"   Baseline (PCA) : {score_baseline:.4f}")
    print(f"   ChromoNet (Deep): {score_chromonet:.4f}")
    
    diff = score_chromonet - score_baseline
    if diff > 0:
        print(f"🎉 验证成功! 提升幅度: +{diff:.4f}")
    else:
        print("⚠️ 提升不明显。")

if __name__ == "__main__":
    main()
