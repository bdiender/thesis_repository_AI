#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

for downloader in "$SCRIPT_DIR/downloaders"/*.sh; do
  bash "$downloader"
done

source "$SCRIPT_DIR/../../venv/bin/activate"

python3 "$SCRIPT_DIR/split_datasets.py" "$@"

echo "Downloaded and split all datasets."
