#!/usr/bin/env bash

### description

if [[ ${1} == "--help" ]] || [[ ${1} == "-h" ]]; then
  echo "Usage: DATA-BIN USER-DIR MODEL-DIR MODE BATCH_SIZE TEST_SET [NPD]"
  echo "MODE --- 0: detokenized BLEU, 1: tokenized BLEU, 2: sacrebleu, 3: compound split BLEU "
  ehco "We evaluate WMT14 with tokenize BLEU[1], evaluate IWSLT with detokenized BLEU[1]"
  exit 0
fi

### parameter sets

DATA_BIN=${1}  # ende, deen, wmt14/450w-LH/ende, wmt14/450W-LH/deen
USER_DIR=${2}
MODEL=${3} # exp name， evaluate the checkpoint_best, checkpoint_avg, checkpoint_best_avg, checkpoint_last
MODE=${4}
BATCH_SIZE=${5}


if [[ ${6} = "both" ]]; then
  TEST_SET_LIST=(valid test)
else
  TEST_SET_LIST=(${6})
fi

if [[ $# -gt 6 ]]; then
  NPD=${7}
else
  NPD=9
fi


function eval_nat_model() {
    function tokenize_bleu() {
      PT=${1}  # checkpoint_best, checkpoint_avg, checkpoint_best_avg, checkpoint_last
      SET=${2} # valid / test
      echo "== [Metric] Tokenized BLEU of ${PT} on ${SET} with BATCH_SIZE:${BATCH_SIZE} =="
      python3 test.py ${DATA_BIN} \
        --user-dir ${USER_DIR} \
        --task nat --beam 1 --remove-bpe --print-step --batch-size "${BATCH_SIZE}" --quiet \
        --gen-subset "${SET}" \
        --left-pad-source False \
        --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 \
        --path "${PT}"
    }

    function detokenize_bleu() {
      PT=${1}  # checkpoint_best, checkpoint_avg, checkpoint_best_avg, checkpoint_last
      SET=${2} # valid / test
      echo "== [Metric] Detokenized BLEU of ${PT} on ${SET} with BATCH_SIZE:${BATCH_SIZE} =="
      python3 test.py ${DATA_BIN} \
        --user-dir ${USER_DIR} \
        --task nat --beam 1 --print-step --batch-size "${BATCH_SIZE}" --quiet \
        --gen-subset "${SET}" \
        --left-pad-source False \
        --tokenizer moses --scoring sacrebleu --remove-bpe \
        --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 \
        --path "${PT}"
    }


    function compound_split_bleu(){
      PT=${1}  # checkpoint_best, checkpoint_avg, checkpoint_best_avg, checkpoint_last
      SET=${2} # valid / test
      TEMP=tmp
      mkdir -p ${TEMP}
      echo "== [Metric] Compound Split BLEU of ${PT} on ${SET} with BATCH_SIZE:${BATCH_SIZE} =="
      python3 generate.py ${DATA_BIN} \
        --user-dir ${USER_DIR} \
        --task nat --beam 1 --print-step --batch-size "${BATCH_SIZE}" \
        --gen-subset "${SET}" \
        --left-pad-source False --remove-bpe \
        --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 \
        --results-path ${TEMP}/${SET} \
        --path "${PT}"
      source latent_glat/compound_split_bleu.sh ${TEMP}/${SET}/generate-${SET}.txt

    }

    function self_rerank_tokenize_bleu(){
        PT=${1}
        SET=${2}
        Beta=${3}
        echo "== [Metric] Tokenized BLEU of ${PT} on ${SET} with BATCH_SIZE:${BATCH_SIZE}, penal60ty=${Beta}, N=${N} =="
        python3 test.py ${DATA_BIN} \
            --user-dir ${USER_DIR} \
            --task nat --beam 1 --remove-bpe --print-step --batch-size ${BATCH_SIZE} --quiet \
            --gen-subset "${SET}" \
            --left-pad-source False \
            --iter-decode-max-iter 0 \
            --iter-decode-with-beam "${NPD}" \
            --iter-decode-eos-penalty "${Beta}" \
            --path "${PT}"
    }

    for set in ${TEST_SET_LIST[*]}; do
      if [[ ${MODE} -eq 1  ]]; then
        tokenize_bleu ${MODEL} ${set} ${BATCH_SIZE}
      elif [[ ${MODE} -eq 2  ]]; then
        detokenize_bleu ${MODEL} ${set} ${BATCH_SIZE}
      elif [[ ${MODE} -eq 3  ]]; then
        compound_split_bleu ${MODEL} ${set} ${BATCH_SIZE}
      elif [[ ${MODE} -eq 4  ]]; then
        for Beta in 0.6 0.7 0.8 0.9 1.0 1.1 1.2; do # length ratio in self-rank NPD
           self_rerank_tokenize_bleu ${MODEL} ${set} ${Beta}
        done
      fi
    done
}
