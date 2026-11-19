from huggingface_hub import snapshot_download
for repo, rev in (("Qwen/Qwen2.5-Coder-14B", "f2ad5164aade432d6d56c24bb71589184d5d613d"),
                  ("Qwen/Qwen2.5-Coder-32B", "2e12b5f7bc878d424d222e224ed40aee564ec45f")):
    p = snapshot_download(repo, revision=rev, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "merges.txt", "vocab.json"])
    print("done", repo, rev, p, flush=True)
