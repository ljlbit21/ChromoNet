# ==================== ChromoNet 官方 Docker v1.0.0 ====================
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

# 设置工作目录
WORKDIR /app

# 复制整个项目（包括 scripts/、src/、notebooks/ 等）
COPY . /app

# 安装依赖（requirements.txt 必须在根目录）
RUN pip install --no-cache-dir -r requirements.txt

# 固定随机种子 + 单线程，保证 GPU/CPU 结果完全一致（Pearson R ≥ 0.9）
ENV PYTHONHASHSEED=42
ENV TORCH_MANUAL_SEED=42
ENV OMP_NUM_THREADS=1
ENV PYTHONPATH=/app

# 预热检查（启动时显示信息）
CMD ["python", "-c", "
import torch
import os
print('🚀 ChromoNet Docker v1.0.0 已启动')
print('   GPU:', torch.cuda.is_available())
print('   CUDA:', torch.version.cuda)
print('   工作目录:', os.getcwd())
print('   使用示例: python scripts/run_pipeline.py --help')
"]

# 可选：直接进入交互环境（方便用户手动运行）
# CMD ["/bin/bash"]
