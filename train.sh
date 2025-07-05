#!/bin/bash

set -e

# Change the paths inside the manifest files to point to the Kathbath audio files.

fairseq-preprocess \
  --only-source \
  --trainpref /mnt/ssd_scratch/TranslitASR-KWS/manifest/train.wrd \
  --validpref /mnt/ssd_scratch/TranslitASR-KWS/manifest/valid.tsv \
  --destdir /mnt/ssd_scratch/TranslitASR-KWS/manifest \
  --workers 4 \
  --srcdict /mnt/ssd_scratch/TranslitASR-KWS/manifest/dict.ltr.txt

fairseq-hydra-train \
    task.data=/mnt/ssd_scratch/TranslitASR-KWS/manifest \
    checkpoint.finetune_from_model=/home/kiran/kws/wav2vec2_models/wav2vec_big_960h.pt \
    --config-dir /home/kiran/kws/fairseq/examples/wav2vec/config/finetuning \
    --config-name vox_960h.yaml