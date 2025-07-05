#!/bin/bash

set -e

# List of models
models=(
/mnt/nvme_storage/aksharantar/marathi_all/outputs/2025-01-11/13-12-09/checkpoints/checkpoint_10_125000.pt
)
languages=("tamil" "telugu" "bengali" "gujarati" "hindi" "kannada" "malayalam" "marathi" "odia" "punjabi")

for language in "${languages[@]}"; do
    for model in "${models[@]}"; do
        # Extract the model name without the path and remove the .pt extension
        model_name=$(basename "$model" .pt)
        echo "Running inference for $language with model: $model_name"

        # Activate the conda environment
        eval "$(conda shell.bash hook)"
        conda activate py39_indic

        # Run the inference script with the current model for the current language
        python /home/kiran/kws/IndicWav2Vec/w2v_inference/scripts/sfi.py \
            --ft-model "$model" \
            --reference-dir "/mnt/ssd_scratch/TranslitASR-KWS/indicsuperb_qbe_testset/qbe_indicsuperb/$language/Audio_vad" \
            --reference-output "/mnt/ssd_scratch/TranslitASR-KWS/indicsuperb_qbe_testset/qbe_embeddings/$language/Audio-$model_name-emb" \
            --queries-dir "/mnt/ssd_scratch/TranslitASR-KWS/indicsuperb_qbe_testset/qbe_indicsuperb/$language/eval_queries_vad" \
            --queries-output "/mnt/ssd_scratch/TranslitASR-KWS/indicsuperb_qbe_testset/qbe_embeddings/$language/eval_queries-$model_name-emb"

        # Deactivate and reactivate base environment
        eval "$(conda shell.bash hook)"
        conda activate base

        # Increase file limit
        ulimit -n 4096

        # Run DTW scoring for the current language
        python /mnt/ssd_scratch/TranslitASR-KWS/dtw_scoring.py \
            --reference-dir "/mnt/ssd_scratch/TranslitASR-KWS/indicsuperb_qbe_testset/qbe_indicsuperb/$language/Audio_vad" \
            --reference-output "/mnt/ssd_scratch/TranslitASR-KWS/indicsuperb_qbe_testset/qbe_embeddings/$language/Audio-$model_name-emb" \
            --queries-dir "/mnt/ssd_scratch/TranslitASR-KWS/indicsuperb_qbe_testset/qbe_indicsuperb/$language/eval_queries_vad" \
            --queries-output "/mnt/ssd_scratch/TranslitASR-KWS/indicsuperb_qbe_testset/qbe_embeddings/$language/eval_queries-$model_name-emb" \
            --exp-dir "/mnt/ssd_scratch/TranslitASR-KWS/results/$language/$model_name-emb"

        # Run the scoring script for the current language
        cd /mnt/ssd_scratch/TranslitASR-KWS/indicsuperb_qbe_testset/qbe_indicsuperb/SCORING
        ./score-TWV-Cnxe.sh "/mnt/ssd_scratch/TranslitASR-KWS/results/$language/$model_name-emb" \
            "/mnt/ssd_scratch/TranslitASR-KWS/indicsuperb_qbe_testset/qbe_indicsuperb/$language/scoring/eval"

        cd /home/kiran/kws/qbe-eval
    done
done