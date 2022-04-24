#!/usr/bin/env bash

ROOT=/home/data_ti4_c/baoy
REPO=${ROOT}/projects/NAT-fairseq
EXP=${ROOT}/experiments

DATA_NAME=Quora
DATA_BIN=${EXP}/data-bin/${DATA_NAME}

LOG_DIR=${EXP}/logs/${DATA_NAME}
SAVE_DIR=${EXP}/checkpoints/${DATA_NAME}

MODEL=AT
LANG=BPE

# Training the AT models
CUDA_VISIBLE_DEVICES=1 python3 ${REPO}/train.py \
    ${DATA_BIN}/${LANG} \
    --tensorboard-logdir ${LOG_DIR}/$LANG/${MODEL} \
    --save-dir ${SAVE_DIR}/$LANG/${MODEL} \
    --task translation \
    --user-dir ${REPO}/latent_glat \
    --arch transformer_iwslt_de_en \
    --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 2048 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 4, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --keep-best-checkpoints 5 \
    --no-epoch-checkpoints \
    --keep-interval-updates 1 \
    --log-format 'simple' --log-interval 100 \
    --save-interval-updates 500 \
    --max-update 100000 \
    --left-pad-source False \
    --num-workers 0