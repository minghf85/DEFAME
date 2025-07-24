from huggingface_hub import snapshot_download

# 下载数据集到指定目录
repo_id = "chenxwh/AVeriTeC"
local_dir = "data/AVeriTeC"

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,  # 避免符号链接问题
    resume_download=True,          # 支持断点续传
)