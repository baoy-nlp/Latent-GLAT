import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.models import register_model, register_model_architecture

from .glat import init_bert_params, GlancingTransformerDecoder
from .vector_quantization import vq_st, vq_search
from .vnat import VariationalNAT
from .vnat import VariationalNATDecoder

INF = 1e-10


@register_model("vqnat")
class VQNAT(VariationalNAT):
    """
    initialize the decoder inputs with latent variables that learned by vector quantization.
    """

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = VQNATDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    @staticmethod
    def add_args(parser):
        VariationalNAT.add_args(parser)
        VQNATDecoder.add_args(parser)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        length_out = self.decoder.forward_length(normalize=False, encoder_out=encoder_out)
        length_tgt = self.decoder.forward_length_prediction(length_out, encoder_out, tgt_tokens)

        # decoding
        decode_out = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            tgt_tokens=tgt_tokens,
            extra_ret=True,
        )
        mask = tgt_tokens.ne(self.pad)
        model_ret, ext = self._compute_loss(decode_out, tgt_tokens, mask)
        model_ret["length"] = {"out": length_out, "tgt": length_tgt, "factor": self.decoder.length_loss_factor}

        if ext is not None:
            # add latent prediction loss
            if "VQ" in ext["prior"]:
                model_ret["vq-L1"] = ext["prior"]["VQ"]

            # update the latent codes with exponential moving average while update the parameter
            # TODO: only update in the local mini-batch, can not synchronized update in multi-gpu or multi update-freq
            if self.training and getattr(self.args, "vq_ema", False):
                self.decoder.update_code(ext["posterior"])

        return model_ret


class VQNATDecoder(VariationalNATDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)

        # test after VQ-EXP5, should remove below
        if hasattr(self, "posterior"):
            delattr(self, "posterior")
            delattr(self, "prior")

        self.code = self._build_codes(args)

        self.latent_factor = getattr(args, "latent_factor", getattr(args, "vq_predictor_ratio", 0.5))
        self.predictor: GlancingTransformerDecoder = self._build_predictor()

        # used for schedule sampling in the latent spaces
        self.vq_schedule_start = getattr(args, "vq_schedule_start", -1)
        self.vq_schedule_ratio = getattr(args, "vq_schedule_ratio", 0.5)
        self.vq_schedule = self.vq_schedule_ratio > 0.
        self.vq_mask_same = getattr(args, "vq_mask_same", False)
        self.vq_mask_diff = getattr(args, "vq_mask_diff", False)
        self.vq_mask_reverse = getattr(args, "vq_mask_reverse", False)
        self.vq_mix_diff = getattr(args, "vq_mix_diff", False)
        self.condition_on_z = getattr(args, "condition_on_z", False)

    @staticmethod
    def add_args(parser):

        # for vector quantization
        parser.add_argument("--vq-ema", action="store_true")
        parser.add_argument("--code-cls", type=str, default="ema")
        parser.add_argument("--vq-split", action="store_true")
        parser.add_argument("--num-codes", type=int)
        parser.add_argument("--lamda", type=float, default=0.999)

        # schedule sampling for the latent codes
        parser.add_argument("--vq-schedule-start", type=int, default=-1)
        parser.add_argument("--vq-schedule-ratio", type=float, default=0.5)
        parser.add_argument("--vq-mask-same", action="store_true", default=False)
        parser.add_argument("--vq-mask-diff", action="store_true", default=False)
        parser.add_argument("--vq-mask-reverse", action="store_true", default=False)
        parser.add_argument("--vq-mix-diff", action="store_true", default=False)

        # for GLAT-based latent predictor
        parser.add_argument("--vq-glat", action="store_true")
        parser.add_argument("--vq-glancing-mode", type=str)

        parser.add_argument("--vq-glancing-num", type=str, default="adaptive",
                            choices=["fixed", "adaptive", "adaptive-uni", "adaptive-rev"])
        parser.add_argument("--vq-glancing-sap", type=str, choices=["uniform", "schedule"], default="uniform")
        parser.add_argument("--vq-start-ratio", type=float)
        parser.add_argument("--vq-end-ratio", type=float)
        parser.add_argument("--vq-anneal-steps", type=int)
        parser.add_argument("--vq-anneal-start", type=int)
        parser.add_argument("--vq-dropout", type=float)

        # Extend after: 05-02
        parser.add_argument("--condition-on-z", action="store_true")
        parser.add_argument("--share-bottom-layers", action="store_true", help="pursue a less memory-cost model")

    def extract_features(
            self,
            prev_output_tokens,
            encoder_out=None,
            early_exit=None,
            embedding_copy=False,
            tgt_tokens=None,
            **unused
    ):
        # compute the decoder inputs
        x, decoder_padding_mask, pos = self.forward_decoder_inputs(prev_output_tokens, encoder_out=encoder_out)

        # predict the latent codes with decoder inputs
        z, z_ret = self.forward_z(encoder_out, tgt_tokens, inputs=x, decoder_padding_mask=decoder_padding_mask)

        # first pass decoding
        residual = x  # x: B, T, C
        x = self.forward_combine(residual, z) if self.latent_use == "input" else residual
        features, ret = self._forward_decoding(x, decoder_padding_mask, encoder_out, early_exit)
        features = self.forward_combine(features, z) if self.latent_use == "output" else features

        if tgt_tokens is not None and self.glat_training and self.training:
            # glancing the reference information
            decoder_inputs, predict, observed = self.glancing_for_decode(
                features=features,
                targets=tgt_tokens,
                mask=decoder_padding_mask + z_ret["prior"]["ref_mask"] if self.condition_on_z else decoder_padding_mask,
                ratio=self.glancing_ratio,
                inputs=residual,
                mode=self.glancing_mode
            )

            # second decoding pass for update parameters
            decoder_inputs = self.forward_combine(decoder_inputs, z) if self.latent_use == "input" else decoder_inputs
            output, ret = self._forward_decoding(decoder_inputs, decoder_padding_mask, encoder_out, early_exit)
            output = self.forward_combine(output, z) if self.latent_use == "output" else output

            ret["inputs"] = residual  # for input-factor
            ret["features"] = features  # for ref-factor
            ret["ref_mask"] = observed  # for NAT Losses

            ret.update(z_ret)  # for VQ-Losses
            return output, ret
        else:
            ret["inputs"] = residual
            ret["features"] = features
            ret.update(z_ret)
            return features, ret

    def step(self, step_num):
        super().step(step_num)
        self.predictor.step(step_num)
        self.vq_schedule = step_num > self.vq_schedule_start and self.vq_schedule_ratio > 0.

    def update_code(self, posterior):
        """ EMA update """
        self.code.update_code(posterior)

    def forward_z(self, encoder_out, tgt_tokens=None, inputs=None, decoder_padding_mask=None):
        if tgt_tokens is not None:
            # vector quantization from the reference --- non-parameter posterior
            inference_out, idx = self._inference_z(inputs, decoder_padding_mask, tgt_tokens)
        else:
            inference_out, idx = None, None

        if self.predictor is None:
            # a non-parameterize predictor, nearest search with decoder inputs
            predict_out, idx = self._inference_z(inputs, decoder_padding_mask)
        else:
            # a parameterize predictor, we use GLAT here.
            predict_out, idx = self._predict_z(inputs, decoder_padding_mask, encoder_out, tgt=idx, out=inference_out)

        if inference_out is not None:
            q = predict_out["z_input"]
        else:
            q = self.code.forward(indices=idx)

        return q, {"prior": predict_out, "posterior": inference_out}

    def _inference_z(self, inputs, decoder_padding_mask, tgt_tokens=None):
        if tgt_tokens is not None:
            # TODO: switch to a context-aware representation, instead of context-independent embeddings
            inputs = self.forward_embedding(tgt_tokens, add_position=False)[0]

        z_q_st, z_q, idx = self.code.straight_through(inputs)

        return {
                   "z_q_st": z_q_st,
                   "z_q": z_q,
                   "z_e": inputs,
                   "idx": idx,
                   "mask": ~decoder_padding_mask,
               }, idx

    def _predict_z(self, inputs, decoder_padding_mask, encoder_out, tgt=None, out=None):
        """ predict the latent variables """

        ret = self.predictor.forward_as_submodule(inputs, decoder_padding_mask, encoder_out, tgt_tokens=tgt)

        z_out, ext = (ret[0], ret[1]) if isinstance(ret, tuple) else (ret, {})
        z_idx = z_out.max(dim=-1)[1]
        mask = decoder_padding_mask

        if tgt is None:
            # predict latent variables while testing
            return {}, z_idx
        else:
            if self.vq_schedule and self.training:
                # schedule sampling used for latent codes
                z_mix = ext.get("glancing_input", None)

                if self.vq_mask_same:
                    # mixing the predict latent codes and reference latent codes
                    # same to CNAT
                    z_mix = self.predictor.sampler.forward_inputs(
                        inputs=None,
                        ref=self.predictor.forward_embedding(tgt)[0],
                        observed=ext["ref_mask"],
                        s_mode="schedule",  # mixing the predict latent codes and reference latent codes
                        pred=self.predictor.forward_embedding(ext["predict"])[0],
                    )
                elif getattr(self, "vq_mask_diff", False):
                    # mixing the predict latent codes and reference latent codes
                    # same to CNAT
                    z_mix = self.predictor.glancing_for_decode(
                        features=ext["features"],
                        targets=tgt,
                        mask=decoder_padding_mask,
                        ratio=self.vq_schedule_ratio,
                        inputs=inputs,
                        s_mode="schedule"
                    )[0]
                elif self.vq_mask_reverse:
                    # mixing the decoder inputs and latent codes
                    # re-sampling the observed latent codes, different to the GLAT-Z predictor
                    z_mix = self.predictor.glancing_for_decode(
                        features=ext["features"],
                        targets=tgt,
                        mask=ext["ref_mask"].squeeze(-1) > 0. + decoder_padding_mask,
                        ratio=self.vq_schedule_ratio,
                        inputs=inputs
                    )[0]
                elif self.vq_mix_diff:
                    # mixing the decoder inputs and latent codes
                    # re-sampling the observed latent codes
                    z_mix = self.predictor.glancing_for_decode(
                        features=ext["features"],
                        targets=tgt,
                        mask=decoder_padding_mask,
                        ratio=self.vq_schedule_ratio,
                        inputs=inputs,
                    )[0]
            else:
                # using the approximate ground truth of latent codes
                z_mix = out["z_q_st"]

            return {
                       "VQ": {
                           "out": z_out,  # glancing outputs
                           "tgt": tgt,  # reference target
                           "factor": self.latent_factor,
                           "mask": ~(decoder_padding_mask + ext["ref_mask"].squeeze(-1) > 0.)
                       },
                       "z_input": z_mix,
                       "z_pred": ext.get("predict", None),
                       "z_ref": out.get("z_q_st", None),
                       "ref_mask": ext["ref_mask"].squeeze(-1) > 0.
                   }, z_idx

    def _build_codes(self, args):
        code_cls = getattr(args, "code_cls", "ema")
        if code_cls == "ema":
            code: EMACode = EMACode(num_codes=args.num_codes, code_dim=args.decoder_embed_dim, lamda=args.lamda)
        elif code_cls == "soft":
            code: SoftCode = SoftCode(num_codes=args.num_codes, code_dim=args.decoder_embed_dim)
        else:
            raise RuntimeError("code cls:{} is error".format(code_cls))
        return code

    def _build_predictor(self):
        main_args = self.args
        args = copy.deepcopy(main_args)
        args.share_decoder_input_output_embed = not getattr(main_args, "vq_split", not self.share_input_output_embed)
        args.decoder_layers = getattr(main_args, "latent_layers", main_args.decoder_layers)
        args.dropout = getattr(main_args, "vq_dropout", main_args.dropout)

        args.glat_training = getattr(main_args, "vq_glat", False)
        args.glancing_mode = getattr(main_args, "vq_glancing_mode", main_args.glancing_mode)
        args.glancing_num = getattr(main_args, "vq_glancing_num", main_args.glancing_num)
        args.glancing_sap = getattr(main_args, "vq_glancing_sap", main_args.glancing_sap)
        args.start_ratio = getattr(main_args, "vq_start_ratio", main_args.start_ratio)
        args.end_ratio = getattr(main_args, "vq_end_ratio", main_args.end_ratio)
        args.anneal_steps = getattr(main_args, "vq_anneal_steps", main_args.anneal_steps)
        args.anneal_start = getattr(main_args, "vq_anneal_start", main_args.anneal_start)

        latent_decoder = GlancingTransformerDecoder(
            args,
            dictionary=Dictionary(num_codes=args.num_codes),
            embed_tokens=self.code.embedding,
            no_encoder_attn=False
        )

        if getattr(args, "share_bottom_layers", False):
            shared_layers = args.latent_layers if args.decoder_layers > args.latent_layers else args.decoder_layers
            for i in range(shared_layers):
                latent_decoder.layers[i] = self.layers[i]

        return latent_decoder


class Code(nn.Module):
    def __init__(self, num_codes, code_dim):
        super().__init__()
        self.K = num_codes
        self.embedding = nn.Embedding(num_codes, code_dim, padding_idx=-1)
        self.embedding.weight.data.uniform_(-1. / num_codes, 1. / num_codes)

    def forward(self, indices=None):
        embed = self.embedding(indices)
        return embed


class SoftCode(Code):
    def __init__(self, num_codes, code_dim):
        super().__init__(num_codes, code_dim)

    def softmax_for_indexing(self, inputs):
        prob = F.linear(inputs, self.embedding.weight)
        return prob.max(dim=-1)[1]

    def forward(self, indices=None, inputs=None):
        if inputs is not None:
            indices = self.softmax_for_indexing(inputs)

        return super().forward(indices)

    def straight_through(self, z_e_x):
        indices = self.softmax_for_indexing(z_e_x)
        z_bar = self.forward(indices=indices)
        return z_bar, z_bar, indices

    def update_code(self, posterior):
        pass


class EMACode(Code):
    def __init__(self, num_codes, code_dim, lamda=0.999, stop_gradient=False):
        super().__init__(num_codes, code_dim)
        self.lamda = lamda
        self.code_count = nn.Parameter(torch.zeros(num_codes).float(), requires_grad=False)
        self.update = not stop_gradient

    def forward(self, indices=None, inputs=None):
        if inputs is not None:
            return vq_search(inputs, self.embedding.weight)

        return super().forward(indices)

    def straight_through(self, z_e_x):
        z, indices = vq_st(z_e_x, self.embedding.weight.detach())
        z_bar = self.embedding.weight.index_select(dim=0, index=indices)
        z_bar = z_bar.view_as(z_e_x)
        return z, z_bar, indices.view(*z.size()[:-1])

    def update_code(self, posterior):
        z_enc = posterior['z_e'].view(-1, posterior['z_e'].size(-1))  # batch_size, sequence_length, D
        enc_sum = self._count_ema(z_enc, posterior["mask"], posterior["idx"])
        self._code_ema(enc_sum)

    def _code_ema(self, z_repr):
        """ exponential moving average """
        count = self.code_count.view(self.K, -1)  # K,1
        mask = (count > 0.0).float()  # K,1
        code = self.embedding.weight.data
        code = mask * (code * self.lamda + (1 - self.lamda) * z_repr / (count + (1 - mask) * INF)) + (1 - mask) * code

        self.embedding.weight.data = code
        self.embedding.weight.requires_grad = self.update

    def _count_ema(self, enc, mask, idx):
        mask = mask.long()
        idx = idx * mask - (1 - mask)  # set the masked indices is -1

        enc = enc.view(-1, enc.size(-1))
        idx = idx.view(-1)
        z_exp = []
        for i in range(self.K):
            i_hit = idx == i  # batch_size*sequence_length,1
            self.code_count[i] = self.lamda * self.code_count[i] + i_hit.sum().float() * (1 - self.lamda)
            z_i_sum = enc[i_hit].sum(dim=0)
            z_exp.append(z_i_sum)

        return torch.stack(z_exp)


class Dictionary(object):
    # helper class for extend the NAT base
    def __init__(self, num_codes):
        super().__init__()
        self.num_codes = num_codes

    def bos(self):
        return -1

    def eos(self):
        return -1

    def unk(self):
        return -1

    def pad(self):
        return -1

    def __len__(self):
        return self.num_codes


def base_architecture(args):
    from nat.vanilla_nat import base_architecture
    base_architecture(args)


@register_model_architecture("vqnat", "vqnat_wmt14")
def wmt14_en_de(args):
    base_architecture(args)


@register_model_architecture('vqnat', 'vqnat_iwslt16')
def iwslt16_de_en(args):
    from nat.vanilla_nat import nat_iwslt16_de_en
    nat_iwslt16_de_en(args)
    base_architecture(args)


@register_model_architecture('vqnat', 'vqnat_iwslt14')
def iwslt14_de_en(args):
    from nat.vanilla_nat import nat_iwslt14_de_en
    nat_iwslt14_de_en(args)
    base_architecture(args)


@register_model_architecture('vqnat', 'vqnat_base')
def nat_base(args):
    from nat.vanilla_nat import nat_base
    nat_base(args)
    base_architecture(args)
