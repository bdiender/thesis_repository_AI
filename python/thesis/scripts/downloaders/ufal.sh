#!/usr/bin/env bash
URL="https://raw.githubusercontent.com/UniversalDependencies/UD_Upper_Sorbian-UFAL/refs/heads/dev"
LANGUAGE_CODE="hsb"

echo "Downloading Upper Sorbian treebanks..."
for SPLIT in train test; do
  FILENAME="hsb_ufal-ud-${SPLIT}.conllu"
  DATA_DIR="data/${LANGUAGE_CODE}/${FILENAME}"

  mkdir -p "$(dirname "$DATA_DIR")"
  curl -L "${URL}/${FILENAME}" -o "$DATA_DIR"
done