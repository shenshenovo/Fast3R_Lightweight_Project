# 安装与运行说明（课程项目轻量化版本）

本仓库基于 Fast3R 做了课程项目级的“轻量化主干”实验开关，并在 demo 推理时额外输出：
推理时间分解 / 峰值显存 / 简单质量指标（`valid_ratio/conf_mean/median_norm`），便于对照实验与截图写报告。

## 0) 前置条件

- 系统：Linux / macOS / Windows（推荐 WSL）
- Python：3.11
- 推荐：NVIDIA GPU + 已安装驱动（可用 `nvidia-smi`）
- 说明：只有用 *git* 方式时才需要 `git`

## 1) 获取代码（二选一）

### 方式 A（不需要 git）：下载 ZIP 安装包

1. 打开 GitHub 仓库页面
2. 点击 **Code → Download ZIP**
3. 解压后进入目录（该目录下应包含 `setup.py`、`requirements.txt`、`fast3r/`）

### 方式 B：`git clone`

```bash
git clone https://github.com/shenshenovo/Fast3R_Lightweight_Project.git
cd Fast3R_Lightweight_Project
```

## 2) 创建 Python 环境（推荐 conda）

```bash
conda create -n fast3r python=3.11 -y
conda activate fast3r
python -m pip install -U pip setuptools wheel
```

## 3) 安装 PyTorch

### 方案 1（推荐）：pip 安装（按你的 CUDA 版本选择）

CUDA 12.4（示例）：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

只有 CPU：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 方案 2：conda 安装（示例）
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

## 4) 安装项目依赖

在仓库根目录执行：
```bash
pip install -r requirements.txt
pip install -e .
```

## 5) 运行 Demo（baseline）

首次运行会从 HuggingFace 下载预训练权重：
```bash
python fast3r/viz/demo.py --checkpoint_dir jedyang97/Fast3R_ViT_Large_512
```

终端将看到：
- 时间分解：`encode_images time / decoder time / head forward time / total Fast3R forward time`
- 显存峰值：`[mem] peak_allocated_GB=...`
- 简单指标：`[preds] ... valid_ratio/conf_mean/median_norm`

## 6) 轻量化对照实验（四组设置）

建议对同一组输入图片，分别运行以下命令，对比 `total time`、`peak_allocated_GB` 与重建预览截图。

### (0) Baseline
```bash
python fast3r/viz/demo.py --checkpoint_dir jedyang97/Fast3R_ViT_Large_512
```

### (1) 方法 1：Low-dim decoder
```bash
FAST3R_USE_LOWDIM_DECODER=1 FAST3R_LOWDIM_RATIO=0.5 \
python fast3r/viz/demo.py --checkpoint_dir jedyang97/Fast3R_ViT_Large_512
```

更稳一点（建议优先尝试）：
```bash
FAST3R_USE_LOWDIM_DECODER=1 FAST3R_LOWDIM_RATIO=0.75 \
python fast3r/viz/demo.py --checkpoint_dir jedyang97/Fast3R_ViT_Large_512
```

### (2) 方法 2：Token subsample
```bash
FAST3R_USE_TOKEN_SUBSAMPLE=1 FAST3R_TOKEN_SUBSAMPLE_STRIDE=2 \
python fast3r/viz/demo.py --checkpoint_dir jedyang97/Fast3R_ViT_Large_512
```

### (1+2) 组合（极限压缩档）
```bash
FAST3R_USE_LOWDIM_DECODER=1 FAST3R_LOWDIM_RATIO=0.5 \
FAST3R_USE_TOKEN_SUBSAMPLE=1 FAST3R_TOKEN_SUBSAMPLE_STRIDE=2 \
python fast3r/viz/demo.py --checkpoint_dir jedyang97/Fast3R_ViT_Large_512
```

## 常见问题

- `Password authentication is not supported`：这是 GitHub 推送时需要 PAT/SSH 的提示；老师安装运行无需推送。
- `Warning, cannot find cuda-compiled version of RoPE2D`：一般不影响正确性，只是可能更慢。
- 离线环境：请提前下载 HF 权重目录，并将 `--checkpoint_dir` 改为本地路径。
