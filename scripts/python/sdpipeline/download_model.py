from huggingface_hub import snapshot_download

def run(model):
	snapshot_download(repo_id=model, repo_type="model")