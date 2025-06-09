#!/bin/bash
export HUGGINGFACE_HUB_TOKEN=PUT_YOUR_HF_TOKEN_HERE
if [ -z "$1" ]; then
  echo "Usage: $0 username/repo_name [output_dir]"
  exit 1
fi

REPO_ID=$1

# If output dir is given, use it; else use repo name as folder name
if [ -n "$2" ]; then
  LOCAL_DIR="$2"
else
  REPO_NAME="${REPO_ID##*/}"
  LOCAL_DIR="./${REPO_NAME}"
fi

python3 - <<END
from huggingface_hub import snapshot_download

snapshot_download(
    token="$HUGGINGFACE_HUB_TOKEN",
    repo_id="$REPO_ID",
    local_dir="$LOCAL_DIR",
    ignore_patterns=[".git/*"]
)
print(f"Downloaded {REPO_ID} to {LOCAL_DIR} without .git folder.")
END
