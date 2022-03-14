#!/usr/bin/env bash

if [[ ${1} = "--help" ]] || [[ ${1} = "-h" ]]; then
    echo "Usage: GPU DATA EXP_DIR MODE [SET] [Scripts]"
    echo "MODE --- 0: detokenized BLEU, 1: tokenized BLEU, 2: sacrebleu, 3: compound split BLEU "
    exit 0
fi

GPU=${1}
DATA=${2}
EXP=${3}
MODE=${4}

if [[ $# -gt 4 ]];
then
    SET=${5}
else
    SET=test
fi

if [[ $# -gt 5 ]];
then
   Scripts=${6}
   echo "Evaluate the model with ${Scripts}"
else
   Scripts=./test.sh
fi



if [[ ${SET} = "both" ]]; then
    for SET in valid test; do
        for F in checkpoint_best.pt checkpoint_avg.pt checkpoint_best_avg.pt; do
            /bin/bash ${Scripts} ${GPU} ${DATA} ${EXP}/${F} ${MODE} ${SET}
        done
    done
else
    for F in checkpoint_best.pt checkpoint_avg.pt checkpoint_best_avg.pt; do
        /bin/bash ${Scripts} ${GPU} ${DATA} ${EXP}/${F} ${MODE} ${SET}
    done
fi

# bash eval.sh 1 wmt14/ende VQNAT-EXP26 1 valid