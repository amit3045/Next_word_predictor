#!/usr/bin/env bash
set -euo pipefail

REPO_MEDIA_BASE="https://media.githubusercontent.com/media/amit3045/Next_word_predictor/main/models"
mkdir -p models

for model_file in new_fine.h5 new_fine.pkl new_lstm.h5 new_lstm.pkl; do
  curl --fail --location --retry 3 \
    "${REPO_MEDIA_BASE}/${model_file}" \
    --output "models/${model_file}"
done

for model_file in new_fine.h5 new_fine.pkl new_lstm.h5 new_lstm.pkl; do
  test -s "models/${model_file}"
done

printf '%s\n' 'Downloaded deployment model files:'
ls -lh models/new_fine.h5 models/new_fine.pkl models/new_lstm.h5 models/new_lstm.pkl
