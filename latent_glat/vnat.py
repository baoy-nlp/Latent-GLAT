import copy

import torch
import torch.nn as nn
from fairseq.models import register_model, register_model_architecture

from .glat import GlancingTransformer
from .glat import GlancingTransformerDecoder
from .glat import init_bert_params
from .utils import GateNet, TransformerEncoderNet, GaussianVariable


@register_model("vnat")
class VariationalNAT(GlancingTransformer):
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = VariationalNATDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    @staticmethod
    def add_args(parser):
        GlancingTransformer.add_args(parser)
        VariationalNATDecoder.add_args(parser)

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
            kl_factor = getattr(self.args, "kl_factor", 1.0)
            model_ret["KL"] = {
                "loss": self.compute_kl_loss(ext["prior"], ext["posterior"]) * kl_factor,
                "factor": kl_factor
            }

        return model_ret

    @classmethod
    def compute_kl_loss(cls, prior_out, posterior_out):
        # prior
        mean1 = prior_out["mean"]
        logv1 = prior_out["logv"]
        var1 = logv1.exp()

        mean2 = posterior_out["mean"]
        logv2 = posterior_out["logv"]
        var2 = logv2.exp()

        # kl = -0.5 * (logv2 - logv1 - (var2 / var1) - (mean2 - mean1).pow(2) / var1 + 1).sum(dim=-1).mean()
        kl = 0.5 * (logv2 - logv1 + (var1 / var2) + (mean2 - mean1).pow(2) / var2 - 1).sum(dim=-1).mean()
        return kl


class VariationalNATDecoder(GlancingTransformerDecoder):
    """
        p(y|z,x)
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)

        self.prior = Prior(args)
        self.posterior = self._build_posterior()
        self.glat_training = getattr(args, "glat_training", False)

        self.gate = GateNet(
            d_model=self.embed_dim * 2,
            d_hidden=self.embed_dim * 4,
            d_output=1 if getattr(args, "use_scalar_gate", True) else self.embed_dim,
            dropout=args.dropout
        ) if getattr(args, "gated_func", "residual") == "residual" else None

        self.latent_use = getattr(args, "latent_use", "input")
        self.latent_factor = getattr(args, "latent_factor", 1.0)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--latent-dim", type=int, default=200)
        parser.add_argument("--latent-layers", type=int, default=5)
        parser.add_argument("--latent-use", type=str, default="input", choices=["input", "output", "layer"])
        parser.add_argument("--gated-func", type=str, default="residual")
        parser.add_argument("--kl-factor", type=float, default=1.0)
        parser.add_argument("--latent-factor", type=float)
        parser.add_argument("--use-scalar-gate", action="store_true", default=False)

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
                features=features, targets=tgt_tokens, mask=decoder_padding_mask, ratio=self.glancing_ratio,
                inputs=residual, mode=self.glancing_mode
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

    def forward_z(self, encoder_out, tgt_tokens=None, inputs=None, decoder_padding_mask=None):
        prior_out = self.prior.forward(inputs=encoder_out.encoder_out, mask=~encoder_out.encoder_padding_mask)
        inner_states = {"prior": prior_out}

        z = prior_out["rec"]  # batch_size, hidden
        if tgt_tokens is not None:
            y_mask = tgt_tokens.ne(self.padding_idx)
            y_embed = self.forward_embedding(tgt_tokens)[0]
            posterior_out = self.posterior.forward(
                x_embed=encoder_out.encoder_out,
                y_embed=y_embed,
                x_mask=~encoder_out.encoder_padding_mask,
                y_mask=y_mask
            )
            inner_states["posterior"] = posterior_out

            z = posterior_out["rec"]
        z = z.unsqueeze(1).contiguous().expand(-1, inputs.size(1), -1)
        return z, inner_states

    def forward_combine(self, inputs, z):
        if self.gate is not None:
            g = self.gate(torch.cat([inputs, z], dim=-1)).sigmoid()
            inputs = inputs * g + z * (1 - g)
        else:
            inputs = inputs + z
        return inputs

    def _build_posterior(self):
        model_args = self.args
        args = copy.deepcopy(model_args)
        args.encoder_layers = getattr(model_args, "latent_layers", model_args.decoder_layers)
        return Posterior(args)


class Prior(nn.Module):
    """
        p(z|x): mapping enc(x) to mean and logv
    """

    def __init__(self, args):
        super().__init__()
        self.latent = GaussianVariable(
            input_dim=args.encoder_embed_dim,
            latent_dim=getattr(args, "latent_dim", 200),
            output_dim=args.encoder_embed_dim
        )

    def forward(self, inputs, mask=None):
        inputs = inputs.transpose(0, 1)
        if mask is not None:
            h_f = (inputs * mask.unsqueeze(-1).float()).sum(dim=1) / mask.sum(dim=-1).float().unsqueeze(-1)
        else:
            h_f = inputs.mean(dim=1)

        return self.latent.forward(inputs=h_f)


class Posterior(nn.Module):
    """
        q(z|x,y): enc(y) and enc(x), mapping enc(x,y) to mean and logv
    """

    def __init__(self, args):
        super().__init__()

        self.y_encoder = TransformerEncoderNet(args)

        self.latent = GaussianVariable(
            input_dim=args.encoder_embed_dim * 2,
            latent_dim=getattr(args, "latent_dim", 200),
            output_dim=args.encoder_embed_dim
        )

    def forward(self, x_embed, y_embed, x_mask=None, y_mask=None):
        def _compute_inputs(inputs, mask=None):
            if mask is not None:
                h = (inputs * mask.unsqueeze(-1).float()).sum(dim=1) / mask.sum(dim=-1).float().unsqueeze(-1)
            else:
                h = inputs.mean(dim=1)
            return h

        x_output = x_embed.transpose(0, 1)
        h_f = _compute_inputs(x_output, x_mask)

        # encoding y
        y_output = self.y_encoder.forward(y_embed, ~y_mask).encoder_out
        y_output = y_output.transpose(0, 1)
        h_e = _compute_inputs(y_output, y_mask)

        # concatenate x and y
        h = torch.cat([h_f, h_e], dim=-1)
        return self.latent.forward(inputs=h)


def base_architecture(args):
    from nat.vanilla_nat import base_architecture
    base_architecture(args)


@register_model_architecture("vnat", "vnat_wmt14")
def wmt14_en_de(args):
    base_architecture(args)


@register_model_architecture('vnat', 'vnat_iwslt16')
def iwslt16_de_en(args):
    from nat.vanilla_nat import nat_iwslt16_de_en
    nat_iwslt16_de_en(args)
    base_architecture(args)


@register_model_architecture('vnat', 'vnat_iwslt14')
def iwslt14_de_en(args):
    from nat.vanilla_nat import nat_iwslt14_de_en
    nat_iwslt14_de_en(args)
    base_architecture(args)


@register_model_architecture('vnat', 'vnat_base')
def nat_base(args):
    from nat.vanilla_nat import nat_base
    nat_base(args)
    base_architecture(args)
