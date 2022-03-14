import math
from collections import namedtuple
from itertools import groupby

import numpy as np
import torch
from fairseq import utils
from fairseq.data.dictionary import Dictionary
from fairseq.iterative_refinement_generator import IterativeRefinementGenerator

DecoderOut = namedtuple('IterativeRefinementDecoderOut', [
    'output_tokens',
    'output_scores',
    'attn',
    'step',
    'max_step',
    'history'
])


def remove_repeats_tensor(inputs):
    inputs = inputs.data.cpu()
    outputs = inputs.clone()
    batch_size, seq_len = inputs.size()
    for batch in range(batch_size):
        for index in range(seq_len - 1):
            if outputs[batch, index] == outputs[batch, index + 1]:
                outputs[batch, index:-1] = outputs[batch, index + 1:]
    return outputs


def remove_repeats(lst_of_sentences):
    lst = []
    for sentence in lst_of_sentences:
        lst.append(" ".join([x[0] for x in groupby(sentence.split())]))
    return lst


def reverse(
        dictionary: Dictionary,
        tensor,
        bpe_symbol=None,
        escape_unk=False,
        extra_symbols_to_ignore=None,
        unk_string=None,
        remove_repeat=False,
):
    return dictionary.string(
        tensor,
        bpe_symbol=bpe_symbol,
        escape_unk=escape_unk,
        extra_symbols_to_ignore=extra_symbols_to_ignore,
        unk_string=unk_string
    )


class NAGenerator(IterativeRefinementGenerator):
    def __init__(
            self,
            tgt_dict,
            models=None,
            eos_penalty=0.0,
            max_iter=10,
            max_ratio=2,
            beam_size=1,
            decoding_format=None,
            retain_dropout=False,
            adaptive=True,
            retain_history=False,
            reranking=False,
            infer_with_tgt=False,
            infer_with_reflen=False,
    ):
        super().__init__(
            tgt_dict, models, eos_penalty,
            max_iter=max_iter,
            max_ratio=max_ratio,
            beam_size=beam_size,
            decoding_format=decoding_format,
            retain_dropout=retain_dropout,
            adaptive=adaptive,
            retain_history=retain_history,
            reranking=reranking
        )
        self.infer_with_tgt = infer_with_tgt
        self.infer_with_reflen = infer_with_reflen or infer_with_tgt
        if not self.reranking and self.beam_size > 1:
            self.eos_penalty = eos_penalty if eos_penalty > 0 else 0.7

    @torch.no_grad()
    def generate(self, models, sample, prefix_tokens=None, extra_code=False):

        if not self.retain_dropout:
            for model in models:
                model.eval()

        model, reranker = models[0], None
        if self.reranking:
            assert len(models) > 1, "Assuming the last checkpoint is the reranker"
            assert self.beam_size > 1, "Reranking requires multiple translation for each example"

            reranker = models[-1]
            models = models[:-1]

        if len(models) > 1 and hasattr(model, 'enable_ensemble'):
            assert model.allow_ensemble, "{} does not support ensembling".format(model.__class__.__name__)
            model.enable_ensemble(models)

        src_tokens = sample["net_input"]["src_tokens"]
        src_lengths = sample["net_input"]["src_lengths"]
        bsz, src_len = src_tokens.size()

        # initialize
        encoder_out = model.forward_encoder([src_tokens, src_lengths, True])

        # include the target information for oracle analysis
        tgt_tokens = sample["target"] if "target" in sample and (
                self.infer_with_tgt or self.infer_with_reflen) else None

        # beam_already = hasattr(model, "initialize_beam_output_tokens")
        beam_already = False
        init_func = model.initialize_output_tokens if not beam_already else model.initialize_beam_output_tokens

        if self.infer_with_reflen and tgt_tokens is not None:
            if beam_already:
                prev_decoder_out = init_func(encoder_out, src_tokens, tgt_tokens=tgt_tokens, beam_size=self.beam_size)
            else:
                prev_decoder_out = init_func(encoder_out, src_tokens, tgt_tokens=tgt_tokens)
        else:
            if beam_already:
                prev_decoder_out = init_func(encoder_out, src_tokens, beam_size=self.beam_size)
            else:
                prev_decoder_out = init_func(encoder_out, src_tokens)

        if self.beam_size > 1:
            assert model.allow_length_beam, "{} does not support decoding with length beam.".format(
                model.__class__.__name__
            )
            if not beam_already:
                prev_decoder_out = model.regenerate_length_beam(prev_decoder_out, self.beam_size)
            # regenerate data based on length-beam
            length_beam_order = utils.new_arange(src_tokens, self.beam_size, bsz).t().reshape(-1)
            encoder_out = model.encoder.reorder_encoder_out(encoder_out, length_beam_order)
            bsz = bsz * self.beam_size

        sent_idxs = torch.arange(bsz)
        prev_output_tokens = prev_decoder_out.output_tokens.clone()

        if self.retain_history:
            prev_decoder_out = prev_decoder_out._replace(history=[prev_output_tokens])

        finalized = [[] for _ in range(bsz)]

        def is_a_loop(x, y, s, a):
            b, l_x, l_y = x.size(0), x.size(1), y.size(1)
            if l_x > l_y:
                y = torch.cat([y, x.new_zeros(b, l_x - l_y).fill_(self.pad)], 1)
                s = torch.cat([s, s.new_zeros(b, l_x - l_y)], 1)
                if a is not None:
                    a = torch.cat([a, a.new_zeros(b, l_x - l_y, a.size(2))], 1)
            elif l_x < l_y:
                x = torch.cat([x, y.new_zeros(b, l_y - l_x).fill_(self.pad)], 1)
            return (x == y).all(1), y, s, a

        def finalized_hypos(step, prev_out_token, prev_out_score=None, prev_out_attn=None, prev_extra_ret=None):
            cutoff = prev_out_token.ne(self.pad)
            tokens = prev_out_token[cutoff]
            if prev_out_score is None:
                scores, score = None, None
            else:
                scores = prev_out_score[cutoff]
                if self.eos_penalty > 0.:
                    score = scores.sum() / math.pow(scores.size(0), self.eos_penalty)
                else:
                    score = scores.mean()

            if prev_out_attn is None:
                hypo_attn, alignment = None, None
            else:
                hypo_attn = prev_out_attn[cutoff]
                alignment = hypo_attn.max(dim=1)[1]
            ret = {
                "steps": step,
                "tokens": tokens,
                "positional_scores": scores,
                "score": score,
                "hypo_attn": hypo_attn,
                "alignment": alignment,
            }

            # for oracle analysis in CNAT
            if prev_extra_ret is not None:
                ref_code = prev_extra_ret["vq_ret"]["posterior"]["tgt"]
                if ref_code is not None:
                    ret["ref_code"] = ref_code.squeeze()[cutoff]

                try:
                    hypo_code = prev_extra_ret["vq_ret"]["prior_out"].token
                    if hypo_code is not None:
                        ret["hypo_code"] = hypo_code.squeeze()[cutoff]
                except:
                    pass

            return ret

        for step in range(self.max_iter + 1):
            decoder_options = {
                "eos_penalty": self.eos_penalty,
                "max_ratio": self.max_ratio,
                "decoding_format": self.decoding_format,
                "tgt_tokens": tgt_tokens if self.infer_with_tgt else None
            }

            prev_decoder_out = prev_decoder_out._replace(step=step, max_step=self.max_iter + 1, )

            if extra_code:
                decoder_out, extra_ret = model.forward_decoder(prev_decoder_out, encoder_out, extra_ret=True,
                                                               **decoder_options)
            else:
                extra_ret = None
                decoder_out = model.forward_decoder(prev_decoder_out, encoder_out, **decoder_options)

            if self.adaptive:
                # terminate if there is a loop
                terminated, out_tokens, out_scores, out_attn = is_a_loop(
                    prev_output_tokens, decoder_out.output_tokens, decoder_out.output_scores, decoder_out.attn
                )
                decoder_out = decoder_out._replace(
                    output_tokens=out_tokens,
                    output_scores=out_scores,
                    attn=out_attn,
                )

            else:
                terminated = decoder_out.output_tokens.new_zeros(decoder_out.output_tokens.size(0)).bool()

            if step == self.max_iter:  # reach last iteration, terminate
                terminated.fill_(1)

            # collect finalized sentences
            finalized_idxs = sent_idxs[terminated]
            finalized_tokens = decoder_out.output_tokens[terminated]
            finalized_scores = decoder_out.output_scores[terminated]
            finalized_attn = (
                None if (decoder_out.attn is None or decoder_out.attn.size(0) == 0) else decoder_out.attn[terminated]
            )

            if self.retain_history:
                finalized_history_tokens = [h[terminated] for h in decoder_out.history]

            for i in range(finalized_idxs.size(0)):
                finalized[finalized_idxs[i]] = [
                    finalized_hypos(
                        step,
                        finalized_tokens[i],
                        finalized_scores[i],
                        None if finalized_attn is None else finalized_attn[i],
                        prev_extra_ret=extra_ret
                    )
                ]

                if self.retain_history:
                    finalized[finalized_idxs[i]][0]['history'] = []
                    for j in range(len(finalized_history_tokens)):
                        finalized[finalized_idxs[i]][0]['history'].append(
                            finalized_hypos(
                                step,
                                finalized_history_tokens[j][i],
                                None,
                                None,
                                prev_extra_ret=extra_ret
                            )
                        )

            # check if all terminated
            if terminated.sum() == terminated.size(0):
                break

            # for next step
            not_terminated = ~terminated
            prev_decoder_out = decoder_out._replace(
                output_tokens=decoder_out.output_tokens[not_terminated],
                output_scores=decoder_out.output_scores[not_terminated],
                attn=decoder_out.attn[not_terminated]
                if (decoder_out.attn is not None and decoder_out.attn.size(0) > 0)
                else None,
                history=[h[not_terminated] for h in decoder_out.history]
                if decoder_out.history is not None
                else None,
            )
            encoder_out = model.encoder.reorder_encoder_out(encoder_out, not_terminated.nonzero().squeeze())
            sent_idxs = sent_idxs[not_terminated]
            prev_output_tokens = prev_decoder_out.output_tokens.clone()

        if self.beam_size > 1:
            if reranker is not None:
                finalized = self.rerank(
                    reranker, finalized, [src_tokens, src_lengths], self.beam_size
                )

            # aggregate information from length beam
            finalized = [
                finalized[np.argmax(
                    [finalized[self.beam_size * i + j][0]['score'] for j in range(self.beam_size)]
                ) + self.beam_size * i] for i in range(len(finalized) // self.beam_size)
            ]

        return finalized

    def adaptive_length_reranking(self, finalized):
        for i in range(len(finalized) // self.beam_size):

            for j in range(self.beam_size):
                score = finalized[self.beam_size * i + j][0]['positional_scores'].sum()
                factor = math.pow(finalized[self.beam_size * i + j][0]['tokens'].size(0), self.eos_penalty)
                finalized[self.beam_size * i + j][0]['score'] = score / factor
        return finalized

# lengths = sum([finalized[self.beam_size * i + j][0]['tokens'].size(0) for j in
#                range(self.beam_size)]) / self.beam_size
