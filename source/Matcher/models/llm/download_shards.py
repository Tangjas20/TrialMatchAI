from huggingface_hub import hf_hub_download

repo_id = "microsoft/phi-4"

num_shards = 6
for i in range(1, num_shards+1):
    fname = f"model-{i:05d}-of-{num_shards:05d}.safetensors"
    path = hf_hub_download(repo_id, filename=fname, force_download=True)
    print("Downloaded", path)