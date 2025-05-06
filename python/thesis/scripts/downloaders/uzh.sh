#!/usr/bin/env bash

URL="https://raw.githubusercontent.com/UniversalDependencies/UD_Alemannic-UZH/refs/heads/master/gsw_uzh-ud-test.conllu"
LANGUAGE_CODE="gsw"
FILENAME="gsw_uzh-ud-test.conllu"
DATA_DIR="data/${LANGUAGE_CODE}/${FILENAME}"

echo "Downloading Alemannic treebank..."
mkdir -p "$(dirname "$DATA_DIR")"
curl -L "${URL}" -o "$DATA_DIR"
