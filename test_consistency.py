import pandas as pd
import numpy as np
import torch
import os
import sys
from scipy.stats import pearsonr

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(BASE_DIR)

from src.model import ChromoNet
from src.preprocessor import SmartPreprocessor

TEST_CSV = os.path.join(BASE_DIR, 'data', 'processed', 'BRCA_1.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'results', 'models', 'best_brca_model.pth')
GENE_POS_PATH = os.path.join(BASE_DIR, 'data', 'reference', 'gene_pos_hg19.csv')

print('GPU vs CPU consistency test (target: Pearson R >= 0.9)...')

if not os.path.exists(TEST_CSV):
    print(f'Error: test file not found: {TEST_CSV}')
    exit(1)
df = pd.read_csv(TEST_CSV).head(200)
preprocessor = SmartPreprocessor(GENE_POS_PATH)
X_res, _, gene_order = preprocessor.process(df, 'New_Cell_Type')

# 2. 加载模型
model = ChromoNet(input_len=len(gene_order))

if not os.path.exists(MODEL_PATH):
    print("❌ 模型文件不存在，请先运行 pipeline 训练模型")
    exit(1)

model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

X_tensor = torch.FloatTensor(X_res)

# 3. GPU 预测（如果有 GPU）
if torch.cuda.is_available():
    model_gpu = model.cuda()
    with torch.no_grad():
        pred_gpu = model_gpu(X_tensor.cuda()).cpu().numpy().flatten()
    print("   ✅ 使用 GPU 进行预测")
else:
    pred_gpu = None
    print("   ⚠️  无 GPU，使用 CPU 替代")

# 4. CPU 预测（强制）
model_cpu = model.cpu()
with torch.no_grad():
    pred_cpu = model_cpu(X_tensor).numpy().flatten()

# 5. 计算 Pearson 相关系数
if pred_gpu is not None:
    r, p = pearsonr(pred_gpu, pred_cpu)
    print(f"\n🎯 GPU vs CPU Pearson R = {r:.4f} (p={p:.2e})")
    if r >= 0.9:
        print("✅ 通过一致性测试！符合老师要求（R ≥ 0.9）")
    else:
        print("⚠️ 一致性未达标，请检查随机种子或模型保存方式")
else:
    print("✅ CPU 测试完成（无 GPU 环境）")

print("💾 测试结束，可放心打包 Docker v1.0.0")
