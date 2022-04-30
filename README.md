Implementation of [<em>latent</em>-GLAT: Glancing at Latent Variables for Parallel Text Generation (in ACL-2022)](https://arxiv.org/abs/2204.02030)

Requirements
---
- python >= 3.6
- pytorch >= 1.7.0
- fairseq==0.10.2
- nltk==3.5
- revtok
- tensorboard
- tensorboardX
- tqdm==4.50.2
- sacremoses
- sacrebleu==1.4.14

Dataset
---
- IWSLT14 & WMT14: please follow the instruction of the [fairseq](https://github.com/pytorch/fairseq/tree/master/examples/translation) for obtaining the datasets.
- [Quora](https://drive.google.com/file/d/1RJqVbN_aWaksJsdye91BCKOjL7toXiWD/view?usp=sharing) & [DailyDialog](https://drive.google.com/file/d/1xp7x7JXShcrReTRsoKSAiAEKhr2_p3C4/view?usp=sharing): we provide a link to download them if you need the datasets.

> We implement our method based on the open-source <u>fairseq</u>. So, we strongly suggest you read the instruction of fairseq for more details. At least, you need to know how to convert the dataset from the "text" or other formats to the "bin" format with <u>fairseq-preprocess</u>.

Usage
---
```bash
# Please replace the following paths according to your setup. 
DATA_BIN=[PATH OF THE PROCESSED DATASET] 
USER_DIR=[PATH] OF THE latent_glat]
SAVE_DIR=[PATH OF YOUR MODEL TARGET]
LOG_DIR=[PATH OF YOUR LOG TARGET]

# For example, training latent-GLAT model on IWSLT14 DE-EN task.
CUDA_VISIBLE_DEVICES=0 python3 train.py ${DATA_BIN} \
    --user-dir ${USER_DIR} \
    --tensorboard-logdir ${LOG_DIR}\
    --save-dir ${SAVE_DIR} \
    --task nat \
    --ddp-backend=no_c10d \
    --criterion generic_loss \
    --arch vqnat_iwslt14 \
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
    --print-ratio-every 300000 \
    --latent-factor 1.0 \
    --vq-ema --gated-func residual \
    --vq-glat \
    --vq-mix-diff \
    --latent-layers 4 \
    --decoder-layers 4 \
    --vq-end-ratio 0.3 \
    --num-codes 64

# For average (best/last/) checkpoints as you need, then you can test the model mostly same to fairseq
fairseq-generate \
    ${DATA_BIN} \
    --gen-subset test \
    --task nat \
    --path ${SAVE_DIR}/checkpoint_best.pt \
    --left-pad-source False \
    --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 \
    --beam 1 --remove-bpe \
    --print-step \
    --batch-size 400
```
*You can also find more training commands in the scripts directory and test commands from "test.sh".*


## Citation

```bibtex
@incollection{bao2022latent-GLAT,
    title = {latent-GLAT: Glancing at Latent Variables for Parallel Text Generation},
    author= {Bao, Yu and Zhou, Hao and Huang, Shujian and Wang, Dongqi and Qian, Lihua and Dai, Xinyu and Chen, Jiajun and Li, Lei},
    booktitle = {ACL},
    year = {2022},
    url = {https://arxiv.org/abs/2204.02030}
}
```
