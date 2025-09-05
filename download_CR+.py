# download_CR+.py
from huggingface_hub import snapshot_download
import os

# 创建目录
os.makedirs("./data/ClaimReview2024+", exist_ok=True)

# 下载数据集
local_path = snapshot_download(
    repo_id="MAI-Lab/ClaimReview2024plus",
    repo_type="dataset",
    local_dir="./data/ClaimReview2024+",
    local_dir_use_symlinks=False
)

print(f"数据集已下载到: {local_path}")
