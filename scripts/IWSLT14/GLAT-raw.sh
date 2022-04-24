#!/usr/bin/env bash

# GLAT w/o KD

ROOT=/home/data_ti4_c/baoy
REPO=${ROOT}/projects/NAT-fairseq

DATA_NAME=iwslt14
DATA_BIN=${ROOT}/experiments/data-bin/${DATA_NAME}
LOG_DIR=${ROOT}/experiments/logs/${DATA_NAME}
SAVE_DIR=${ROOT}/experiments/checkpoints/${DATA_NAME}


MODEL=GLAT-raw
src=de
tgt=en
LANG=${src}${tgt}


CUDA_VISIBLE_DEVICES=6 /home/user_data/anaconda3/envs/fairseq/bin/python3 ${REPO}/train.py \
    ${DATA_BIN}/$LANG \
    --user-dir ${REPO}/latent_glat \
    --tensorboard-logdir ${LOG_DIR}/$LANG/${MODEL} \
    --save-dir ${SAVE_DIR}/$LANG/${MODEL} \
    --task nat \
    --ddp-backend=no_c10d \
    --criterion generic_loss \
    --arch glat_iwslt14 \
    --share-decoder-input-output-embed \
    --mapping-func interpolate \
    --mapping-use output \
    --share-rel-embeddings \
    --block-cls highway \
    --self-attn-cls shaw \
    --enc-self-attn-cls shaw --enc-block-cls highway \
    --max-rel-positions 4 \
    --noise full_mask \
    --apply-bert-init \
    --optimizer adam --adam-betas '(0.9, 0.999)' --clip-norm 10.0 \
   --lr 3e-4 --lr-scheduler polynomial_decay --end-learning-rate 1e-5 \
   --warmup-updates 0 --total-num-update 250000 --dropout 0.3 --weight-decay 0 \
    --encoder-learned-pos \
    --pred-length-offset \
    --length-loss-factor 0.1 \
    --label-smoothing 0.0 \
    --log-format 'simple' --log-interval 100 \
    --max-tokens 2048 \
    --update-freq 1 \
    --save-interval-updates 500 \
    --keep-best-checkpoints 5 \
    --no-epoch-checkpoints \
    --keep-interval-updates 5 \
    --max-update 250000 \
    --num-workers 0 \
    --eval-bleu \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --iter-decode-max-iter 0 \
    --iter-decode-eos-penalty 0 \
    --left-pad-source False \
    --glat-training \
    --start-ratio 0.5 \
    --end-ratio 0.3 \
    --anneal-start 4000 \
    --anneal-steps 150000 \
    --print-ratio-every 300000
