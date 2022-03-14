import json
import logging
from argparse import Namespace

import torch
from fairseq import utils
from fairseq.data import (
    encoders,
)
from fairseq.tasks import register_task
from fairseq.tasks.translation_lev import TranslationLevenshteinTask

from .generator import reverse

EVAL_BLEU_ORDER = 4

logger = logging.getLogger(__name__)


def linear_anneal(warmup_steps, anneal_steps, step):
    if step < warmup_steps:
        return 0.0

    if anneal_steps <= 0:
        return 1.0

    return (step - warmup_steps) / (anneal_steps - warmup_steps)


@register_task('nat')
class NATGenerationTask(TranslationLevenshteinTask):
    """
    Translation (Sequence Generation) task for Non-Autoregressive Transformer
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        TranslationLevenshteinTask.add_args(parser)
        parser.add_argument('--infer-with-reflen', default=False, action='store_true')
        parser.add_argument('--infer-with-tgt', default=False, action='store_true')
        parser.add_argument("--no-accuracy", action='store_true', default=False)
        parser.add_argument('--ptrn-model-path', type=str, default=None)
        parser.add_argument("--posterior-warmup-updates", type=int, default=0)
        parser.add_argument("--posterior-anneal-updates", type=int, default=0)
        parser.add_argument("--extract-code", action='store_true', default=False)


    def build_generator(self, models, args):
        # add models input to match the API for SequenceGenerator
        # from fairseq.iterative_refinement_generator import IterativeRefinementGenerator
        from .generator import NAGenerator
        return NAGenerator(
            self.target_dictionary,
            eos_penalty=getattr(args, 'iter_decode_eos_penalty', 0.0),
            max_iter=getattr(args, 'iter_decode_max_iter', 0),
            beam_size=getattr(args, 'iter_decode_with_beam', 1),
            reranking=getattr(args, 'iter_decode_with_external_reranker', False),
            decoding_format=getattr(args, 'decoding_format', None),
            adaptive=not getattr(args, 'iter_decode_force_max_iter', False),
            retain_history=getattr(args, 'retain_iter_history', False),
            infer_with_tgt=getattr(args, "infer_with_tgt", False),
            infer_with_reflen=getattr(args, "infer_with_reflen", False)
        )

    def build_model(self, args):
        model = super().build_model(args)
        if getattr(args, 'eval_bleu', False):
            assert getattr(args, 'eval_bleu_detok', None) is not None, (
                '--eval-bleu-detok is required if using --eval-bleu; '
                'try --eval-bleu-detok=moses (or --eval-bleu-detok=space '
                'to disable detokenization, e.g., when using sentencepiece)'
            )
            detok_args = json.loads(getattr(args, 'eval_bleu_detok_args', '{}') or '{}')
            self.tokenizer = encoders.build_tokenizer(Namespace(
                tokenizer=getattr(args, 'eval_bleu_detok', None),
                **detok_args
            ))

            gen_args = json.loads(getattr(args, 'eval_bleu_args', '{}') or '{}')
            gen_args = Namespace(**gen_args)
            gen_args.iter_decode_eos_penalty = getattr(args, 'iter_decode_eos_penalty', 0.0)
            gen_args.iter_decode_max_iter = getattr(args, 'iter_decode_max_iter', 10)
            gen_args.iter_decode_beam = getattr(args, 'iter_decode_with_beam', 1)
            gen_args.iter_decode_external_reranker = getattr(args, 'iter_decode_with_external_reranker', False)
            gen_args.decoding_format = getattr(args, 'decoding_format', None)
            gen_args.iter_decode_force_max_iter = getattr(args, 'iter_decode_force_max_iter', False)
            gen_args.retain_history = getattr(args, 'retain_iter_history', False)
            gen_args.infer_with_tgt = getattr(args, "infer_with_tgt", False)
            gen_args.infer_with_reflen = getattr(args, "infer_with_reflen", False)
            self.sequence_generator = self.build_generator([model], gen_args)
        return model

    def task_step_update(self, update_num, model):
        if update_num < self.args.posterior_anneal_updates:
            factor = linear_anneal(
                warmup_steps=getattr(self.args, "posterior_warmup_updates", 0),
                anneal_steps=getattr(self.args, "posterior_anneal_updates", 0),
                step=update_num
            )
            setattr(model.decoder, "alpha", getattr(self.args, "vq_alpha", 0.0) * factor)
            setattr(model.decoder, "vq_kl", getattr(self.args, "vq_kl", 0.0) * factor)
            setattr(model.decoder, "info_z", getattr(self.args, "info_z", 0.0) * factor)
            setattr(model.decoder, "xy_reg", getattr(self.args, "xy_reg", 0.0) * factor)
            setattr(model.decoder, "zy_reg", getattr(self.args, "zy_reg", 0.0) * factor)
            setattr(model.decoder, "zy_mul", getattr(self.args, "zy_mul", 0.0) * factor)
            setattr(model.decoder, "info_x", getattr(self.args, "info_x", 0.0) * factor)
            setattr(model.decoder, "repr_z", getattr(self.args, "repr_z", 0.0) * factor)
            setattr(model.decoder, "lamda1", getattr(self.args, "lamda1", 0.0) * factor)
            setattr(model.decoder, "lamda2", getattr(self.args, "lamda2", 0.0) * factor)

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        self.task_step_update(update_num, model)
        if hasattr(model, "model_step_update"):
            model.model_step_update(update_num)
        return super().train_step(sample, model, criterion, optimizer, update_num, ignore_grad)

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = reverse(
                self.tgt_dict, toks.int().cpu(), self.args.eval_bleu_remove_bpe,
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]['tokens']))
            refs.append(decode(
                utils.strip_pad(sample['target'][i], self.tgt_dict.pad()),
                escape_unk=True,  # don't count <unk> as matches to the hypo
            ))
        if self.args.eval_bleu_print_samples:
            logger.info('example hypothesis: ' + hyps[0])
            logger.info('example reference: ' + refs[0])
        if self.args.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize='none')
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.args.eval_bleu:
            model.eval()
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output['_bleu_sys_len'] = bleu.sys_len
            logging_output['_bleu_ref_len'] = bleu.ref_len
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output['_bleu_counts_' + str(i)] = bleu.counts[i]
                logging_output['_bleu_totals_' + str(i)] = bleu.totals[i]
            model.train()
        return loss, sample_size, logging_output

    def inference_step(self, generator, models, sample, prefix_tokens=None, extra_code=False):
        with torch.no_grad():
            return generator.generate(models, sample, prefix_tokens=prefix_tokens, extra_code=extra_code)
