#!/usr/bin/env bash

URL="https://raw.githubusercontent.com/UniversalDependencies/UD_Alemannic-UZH/refs/heads/dev/gsw_uzh-ud-test.conllu"
LANGUAGE_CODE="vep"
FILENAME="vep_vwt-ud-test.conllu"
DATA_DIR="data/${LANGUAGE_CODE}/${FILENAME}"

echo "Downloading Veps treebank..."
mkdir -p "$(dirname "$DATA_DIR")"
curl -L "${URL}" -o "$DATA_DIR"
