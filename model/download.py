from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ccarrez/meditron-7b-Q4_K_M-GGUF",
    local_dir="models/llama-7b-4bit",
    local_dir_use_symlinks=False
)