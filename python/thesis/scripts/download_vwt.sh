#!/usr/bin/env bash

URL="https://raw.githubusercontent.com/UniversalDependencies/UD_Veps-VWT/dev/vep_vwt-ud-test.conllu"
FILENAME="vep_vwt-ud-test.conllu"
DATA_DIR="data/${FILENAME}"

echo "Downloading Veps treebank..."

curl -L "${URL}" -o "${DATA_DIR}"
