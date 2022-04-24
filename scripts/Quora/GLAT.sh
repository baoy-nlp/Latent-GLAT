#!/usr/bin/env bash

ROOT=/home/data_ti4_c/baoy
REPO=${ROOT}/projects/NAT-fairseq
EXP=${ROOT}/experiments

DATA_NAME=Quora
DATA_BIN=${EXP}/data-bin/${DATA_NAME}

LOG_DIR=${EXP}/logs/${DATA_NAME}
SAVE_DIR=${EXP}/checkpoints/${DATA_NAME}

MODEL=GLAT
LANG=BPE

ARCH=glat_iwslt14

CUDA_VISIBLE_DEVICES=3 python3 ${REPO}/train.py \
    ${DATA_BIN}/${LANG} \
    --tensorboard-logdir ${LOG_DIR}/$LANG/${MODEL} \
    --save-dir ${SAVE_DIR}/$LANG/${MODEL} \
    --user-dir ${REPO}/latent_glat \
    --ddp-backend=no_c10d \
    --task nat \
    --noise full_mask \
    --arch ${ARCH} \
    --share-all-embeddings --share-rel-embeddings \
    --mapping-func soft --mapping-use output \
    --block-cls highway \
    --self-attn-cls shaw --enc-self-attn-cls shaw --enc-block-cls highway \
    --max-rel-positions 4 \
    --criterion generic_loss \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr 3e-4 --lr-scheduler polynomial_decay --end-learning-rate 1e-5 --warmup-updates 0 \
    --total-num-update 100000 --label-smoothing 0.1 \
    --dropout 0.1 --weight-decay 0.01 \
    --encoder-learned-pos \
    --pred-length-offset  \
    --length-loss-factor 0.1 \
    --apply-bert-init \
    --log-format 'simple' --log-interval 100 \
    --log-format 'simple' --log-interval 100 \
    --fixed-validation-seed 7 \
    --max-tokens 2048 \
    --update-freq 1 \
    --save-interval-updates 500 \
    --keep-best-checkpoints 5 \
    --no-epoch-checkpoints \
    --keep-interval-updates 1 \
    --max-update 100000 \
    --num-workers 0 \
    --eval-bleu \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --iter-decode-max-iter 0 \
    --iter-decode-eos-penalty 0 \
    --left-pad-source False \
    --no-accuracy \
    --glat-training
