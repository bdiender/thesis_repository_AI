#!/usr/bin/env bash

URL="https://raw.githubusercontent.com/UniversalDependencies/UD_Faroese-OFT/refs/heads/dev/fo_oft-ud-test.conllu"
LANGUAGE_CODE="fao"
FILENAME="fo_oft-ud-test.conllu"
DATA_DIR="data/${LANGUAGE_CODE}/${FILENAME}"

echo "Downloading Faroese treebank..."
mkdir -p "$(dirname "$DATA_DIR")"
curl -L "${URL}" -o "$DATA_DIR"
