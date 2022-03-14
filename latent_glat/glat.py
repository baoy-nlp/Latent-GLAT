import random

import torch.nn.functional as F
from fairseq.models import register_model, register_model_architecture
from fairseq.modules.transformer_sentence_encoder import init_bert_params

try:
    from lunanlp import torch_seed
except ImportError:
    pass

from nat.vanilla_nat import NAT
from nat.vanilla_nat import NATDecoder
from nat.vanilla_nat import ensemble_decoder


@register_model("glat")
class GlancingTransformer(NAT):

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = GlancingTransformerDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    @staticmethod
    def add_args(parser):
        NAT.add_args(parser)
        GlancingTransformerDecoder.add_args(parser)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        length_out = self.decoder.forward_length(normalize=False, encoder_out=encoder_out)
        length_tgt = self.decoder.forward_length_prediction(length_out, encoder_out, tgt_tokens)

        decode_out = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            tgt_tokens=tgt_tokens,
            extra_ret=True,
        )

        mask = tgt_tokens.ne(self.pad)

        # add nat prediction loss
        model_ret, _ = self._compute_loss(decode_out, tgt_tokens, mask)

        # add target length prediction loss
        model_ret["length"] = {"out": length_out, "tgt": length_tgt, "factor": self.decoder.length_loss_factor}

        return model_ret

    def model_step_update(self, step_num):
        """ interface for applying the step schedule """
        self.decoder.step(step_num)

    def _compute_loss(self, decode_out, tgt_tokens, mask):
        if isinstance(decode_out, tuple):
            decode_out, inner_out = decode_out[0], decode_out[1]
        else:
            inner_out = None

        if inner_out is not None and "ref_mask" in inner_out:
            word_ins_mask = (inner_out["ref_mask"].squeeze(-1) < 1.0) * mask  # non reference and non padding
        else:
            word_ins_mask = mask  # non padding

        model_ret = {
            "word_ins": {"out": decode_out, "tgt": tgt_tokens, "mask": word_ins_mask,
                         "ls": self.args.label_smoothing, "nll_loss": True},
        }

        if inner_out is not None:
            if self.decoder.input_factor > 0.0:
                # push the inputs close to the reference y
                model_ret["word_L0"] = {
                    "out": self.decoder.output_layer(inner_out["inputs"]),
                    "tgt": tgt_tokens,
                    "mask": (~word_ins_mask) * mask if "ref_mask" in inner_out else mask,  # reference
                    "factor": self.decoder.input_factor,
                    "nll_loss": True
                }

            if self.decoder.ref_factor > 0.0:
                # push the decoder inputs close to the reference y
                model_ret["word_L1"] = {
                    "out": self.decoder.output_layer(inner_out["features"]),
                    "tgt": tgt_tokens,
                    "mask": (~word_ins_mask) * mask if "ref_mask" in inner_out else mask,  # reference
                    "factor": self.decoder.ref_factor,
                    "nll_loss": True
                }

        return model_ret, inner_out


class GlancingTransformerDecoder(NATDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)

        # hyper-parameter
        self.glat_training = args.glat_training

        if self.glat_training and self.training:
            print("Training the NAT decoder with glancing trick")

        self.glancing_mode = args.glancing_mode

        self.input_factor = args.input_factor
        self.ref_factor = args.ref_factor
        self.glancing_by_input = args.glancing_by_input
        self.glancing_ratio = args.start_ratio

        self.ratio_scheduler = StepAnnealScheduler(args)
        self.sampler = GlancingSampler(args)
        self.fix_seed = getattr(args, "fix_seed", False)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--glat-training", action="store_true", default=False)
        parser.add_argument("--glancing-mode", type=str, choices=["glancing", "schedule"], default="glancing")

        parser.add_argument("--glancing-num", type=str, choices=["fixed", "adaptive", "adaptive-uni", "adaptive-rev"],
                            default="adaptive", help="glancing sampling number")
        parser.add_argument("--glancing-sap", type=str, choices=["uniform", "schedule"], default="uniform",
                            help="glancing sampling mode")

        # step annealing scheduler
        parser.add_argument("--start-ratio", type=float, default=0.5)
        parser.add_argument("--end-ratio", type=float, default=0.5)
        parser.add_argument("--anneal-steps", type=int, default=1)
        parser.add_argument("--anneal-start", type=int, default=300000)
        parser.add_argument("--print-ratio-every", type=int, default=10000)

        # for training losses
        parser.add_argument("--glancing-by-input", action="store_true", default=False)
        parser.add_argument("--input-factor", type=float, default=0.0)
        parser.add_argument("--ref-factor", type=float, default=0.0)
        parser.add_argument("--fix-seed", action="store_true", default=False)

    @ensemble_decoder
    def forward(self, normalize, encoder_out, prev_output_tokens, step=0, tgt_tokens=None, **unused):
        features, ret = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            embedding_copy=(step == 0) & self.src_embedding_copy,
            tgt_tokens=tgt_tokens,
            **unused
        )
        decoder_out = self.output_layer(features)
        decoder_out = F.log_softmax(decoder_out, -1) if normalize else decoder_out

        if unused.get("extra_ret", False) and tgt_tokens is not None:
            return decoder_out, ret
        else:
            return decoder_out

    def extract_features(
            self,
            prev_output_tokens,
            encoder_out=None,
            early_exit=None,
            embedding_copy=False,
            tgt_tokens=None,
            **unused
    ):
        # compute decoder inputs
        x, decoder_padding_mask, pos = self.forward_decoder_inputs(prev_output_tokens, encoder_out=encoder_out)

        # first decoding pass
        rand_seed = random.randint(0, 19260817)
        if self.fix_seed:
            with torch_seed(rand_seed):
                if not self.glancing_by_input:
                    features, ret = self._forward_decoding(x, decoder_padding_mask, encoder_out, early_exit)
                else:
                    features, ret = x, {}
        else:
            if not self.glancing_by_input:
                features, ret = self._forward_decoding(x, decoder_padding_mask, encoder_out, early_exit)
            else:
                features, ret = x, {}

        if tgt_tokens is not None and self.glat_training and self.training:
            # glancing the reference information
            decoder_inputs, predict, observed = self.glancing_for_decode(
                features=features, targets=tgt_tokens, mask=decoder_padding_mask, ratio=self.glancing_ratio, inputs=x
            )

            # second decoding pass for update parameters
            if self.fix_seed:
                with torch_seed(rand_seed):
                    output, ret = self._forward_decoding(decoder_inputs, decoder_padding_mask, encoder_out, early_exit)
            else:
                output, ret = self._forward_decoding(decoder_inputs, decoder_padding_mask, encoder_out, early_exit)
            ret["inputs"] = x
            ret["features"] = features
            ret["ref_mask"] = observed
            return output, ret
        else:
            ret["inputs"] = x
            ret["features"] = features
            return features, ret

    def forward_as_submodule(self, inputs, decoder_padding_mask, encoder_out=None, early_exit=None, tgt_tokens=None,
                             normalize=False, **unused):
        """ can be extended as extract_features(), w/ forward decoder inputs & w/o embedding projection """
        x = inputs
        rand_seed = random.randint(0, 19260817)
        if self.fix_seed:
            with torch_seed(rand_seed):
                if not self.glancing_by_input:
                    # projection from the first decoder outputs
                    features, ret = self._forward_decoding(x, decoder_padding_mask, encoder_out, early_exit)
                else:
                    # projection from the decoder inputs
                    features, ret = x, {}
        else:
            if not self.glancing_by_input:
                # projection from the first decoder outputs
                features, ret = self._forward_decoding(x, decoder_padding_mask, encoder_out, early_exit)
            else:
                # projection from the decoder inputs
                features, ret = x, {}

        if tgt_tokens is not None and self.glat_training:
            # second pass decoding
            decoder_inputs, predict, observed = self.glancing_for_decode(
                features=features, targets=tgt_tokens, mask=decoder_padding_mask, ratio=self.glancing_ratio,
                inputs=x, mode=self.glancing_mode
            )
            if self.fix_seed:
                with torch_seed(rand_seed):
                    output, ret = self._forward_decoding(decoder_inputs, decoder_padding_mask, encoder_out, early_exit)
            else:
                output, ret = self._forward_decoding(decoder_inputs, decoder_padding_mask, encoder_out, early_exit)

            # for schedule sampling
            ret["ref_mask"] = observed
            ret["inputs"] = x
            ret["features"] = features

            ret["predict"] = predict
            ret["glancing_input"] = decoder_inputs

            features = output
        else:
            ret["ref_mask"] = decoder_padding_mask
            ret["inputs"] = x
            ret["features"] = features
        decoder_out = self.output_layer(features)

        if ret.get("predict", None) is None:
            ret["predict"] = decoder_out.max(dim=-1)

        decoder_out = F.log_softmax(decoder_out, -1) if normalize else decoder_out

        if tgt_tokens is not None:
            return decoder_out, ret
        else:
            return decoder_out

    def glancing_for_decode(self, features, targets, mask, ratio=0.5, inputs=None, **kwargs):
        """ sampling the reference and mixed the inputs"""
        logits = self.output_layer(features)
        prob, predict = logits.max(dim=-1)
        pred_embed = self.forward_embedding(predict)[0]

        sample = self.sampler.forward(targets=targets, mask=mask, ratio=ratio, logits=logits)
        observed = sample.float().unsqueeze(-1)
        ref_embed = self.forward_embedding(targets)[0]

        decode_inputs = self.sampler.forward_inputs(
            inputs=inputs, ref=ref_embed, observed=observed, pred=pred_embed,
            s_mode=kwargs.get("s_mode", None)
        )

        return decode_inputs, predict, observed

    def _forward_decoding(self, x, decoder_padding_mask, encoder_out=None, early_exit=None):
        """ Transformer decoding function: computing hidden states given encoder outputs and decoder inputs """
        x = x.transpose(0, 1)
        attn = None
        inner_states = [x]

        for i, layer in enumerate(self.layers):
            if (early_exit is not None) and (i >= early_exit):
                break

            x, attn, _ = layer(
                x,
                encoder_out.encoder_out if not self.layerwise_attn else encoder_out.encoder_states[i],
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                self_attn_mask=None,
                self_attn_padding_mask=decoder_padding_mask,
            )
            inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        x = x.transpose(0, 1)  # T x B x C -> B x T x C

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": attn, "inner_states": inner_states}

    def step(self, step_num):
        """ update the glancing ratio """
        self.glancing_ratio = self.ratio_scheduler.forward(step_num)


class StepAnnealScheduler(object):
    def __init__(self, args):
        super().__init__()
        self.start_ratio = args.start_ratio
        self.start_ratio = args.start_ratio
        self.end_ratio = args.end_ratio
        self.anneal_steps = args.anneal_steps
        self.anneal_start = args.anneal_start
        self.anneal_end = self.anneal_start + self.anneal_steps
        self.step_ratio = (self.end_ratio - self.start_ratio) / self.anneal_steps
        self.print_ratio_every = args.print_ratio_every

    def forward(self, step_num):
        if step_num < self.anneal_start:
            return self.start_ratio
        elif step_num >= self.anneal_end:
            return self.end_ratio
        else:
            ratio = self.start_ratio + self.step_ratio * (step_num - self.anneal_start)
            if (step_num + 1) % self.print_ratio_every == 0:
                print("=" * 15, "STEP: {} RATIO:{}".format(step_num + 1, ratio), "=" * 15)
            return ratio


class GlancingSampler(object):
    def __init__(self, args):
        super().__init__()
        self.n_mode = args.glancing_num
        self.s_mode = args.glancing_sap

    def forward(self, targets, mask, ratio=0.5, logits=None):
        """return the positions to be replaced """
        from .utils import glancing_sampling
        return glancing_sampling(
            targets=targets, padding_mask=mask, ratio=ratio, logits=logits, n_mode=self.n_mode,
            s_mode=self.s_mode
        )

    def forward_inputs(self, inputs, ref, observed, pred=None, s_mode=None):
        s_mode = self.s_mode if s_mode is None else s_mode

        if s_mode == "schedule":
            assert pred is not None, "schedule needs prediction"
            inputs = pred
        return (1 - observed) * inputs + observed * ref


def base_architecture(args):
    from nat.vanilla_nat import base_architecture
    base_architecture(args)


@register_model_architecture("glat", "glat_wmt14")
def wmt14_en_de(args):
    base_architecture(args)


@register_model_architecture('glat', 'glat_iwslt16')
def iwslt16_de_en(args):
    from nat.vanilla_nat import nat_iwslt16_de_en
    nat_iwslt16_de_en(args)
    base_architecture(args)


@register_model_architecture('glat', 'glat_iwslt14')
def iwslt14_de_en(args):
    from nat.vanilla_nat import nat_iwslt14_de_en
    nat_iwslt14_de_en(args)
    base_architecture(args)


@register_model_architecture('glat', 'glat_base')
def nat_base(args):
    from nat.vanilla_nat import nat_base
    nat_base(args)
    base_architecture(args)
