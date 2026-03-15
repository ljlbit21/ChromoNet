# convert_data.py - h5ad/10x 转 csv 工具
import sys
import os
import scanpy as sc
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Convert single-cell data (h5ad/10x) to CSV for ChromoNet")
    parser.add_argument("--input", type=str, required=True, help="Input file path (.h5ad or directory containing .mtx)")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    parser.add_argument("--transpose", action='store_true', help="Transpose the matrix (if genes are columns)")
    
    args = parser.parse_args()
    
    print(f"📂 Reading: {args.input}")
    
    try:
        if args.input.endswith(".h5ad"):
            adata = sc.read_h5ad(args.input)
        elif os.path.isdir(args.input):
            adata = sc.read_10x_mtx(args.input, var_names='gene_symbols', cache=True)
        else:
            # 尝试作为普通 csv 读取
            df = pd.read_csv(args.input, index_col=0)
            adata = sc.AnnData(df)

        print(f"ℹ️  Loaded shape: {adata.shape} (Cells x Genes)")
        
        # 转换为 DataFrame
        df = pd.DataFrame(adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X,
                          index=adata.obs_names,
                          columns=adata.var_names)
        
        # 如果需要，保存 Metadata (Label)
        # 尝试寻找常见的 label 列名
        possible_labels = ['cell_type', 'New_Cell_Type', 'malignant', 'label']
        for lab in possible_labels:
            if lab in adata.obs.columns:
                print(f"✨ Found label column: {lab}")
                df['New_Cell_Type'] = adata.obs[lab].values
                break
        
        # 保存
        print(f"💾 Saving to: {args.output}")
        df.to_csv(args.output)
        print("✅ Conversion Done!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
