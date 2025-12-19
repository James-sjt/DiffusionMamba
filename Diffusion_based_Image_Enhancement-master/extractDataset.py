from huggingface_hub import hf_hub_download
import zipfile
import os

repo_id = "James0323/enhanceImg"
filename = "data.zip"

# Download the .zip file
file_path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    repo_type="dataset",
)

extract_to = "./"

os.makedirs(extract_to, exist_ok=True)

with zipfile.ZipFile(file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print(f"File downloaded and extracted to: {extract_to}")
