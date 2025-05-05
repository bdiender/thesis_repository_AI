#!/usr/bin/env bash
set -e

CFG="$1"
FILE="configs/config.yaml"

SERIAL_DIR=$(python3 - <<EOF
import yaml
cfg = yaml.safe_load(open("$FILE"))
print(cfg["$CFG"]["training"]["output_dir"])
EOF
)

allennlp train \
  "$FILE" \
  --serialization-dir "$SERIAL_DIR" \
  --include-package src \
  --overrides "{
    \"dataset_reader\": {\"type\": \"ud_reader\"}
  }"
