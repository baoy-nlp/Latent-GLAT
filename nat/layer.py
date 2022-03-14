from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules.transformer_layer import (
    TransformerDecoderLayer,
    MultiheadAttention,
    TransformerEncoderLayer,
    LayerNorm
)
from torch import Tensor

from .modules import FeedForward, RelativeSelfAttention, RelativePositionEmbeddings


def build_relative_embeddings(args, embedding_dim=None):
    if embedding_dim is None:
        embedding_dim = args.decoder_embed_dim // getattr(args, "decoder_attention_heads")
    return RelativePositionEmbeddings(
        max_rel_positions=getattr(args, "max_rel_positions", 4),
        embedding_dim=embedding_dim,
        direction=True,
        dropout=args.dropout
    )


class BlockedEncoderLayer(TransformerEncoderLayer):
    def __init__(self, args, relative_keys=None, relative_vals=None):
        super().__init__(args)
        self.ffn_block = FeedForward(
            d_model=self.embed_dim,
            d_hidden=args.decoder_ffn_embed_dim,
            dropout=args.dropout
        ) if args.enc_block_cls == "highway" else None

        self.relative_keys = relative_keys
        self.relative_vals = relative_vals

        self.self_attn = self.build_self_attention(self.embed_dim, args)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, eps=getattr(args, "layer_norm_eps", 1e-5))
        self.final_layer_norm = LayerNorm(self.embed_dim, eps=getattr(args, "layer_norm_eps", 1e-5))

    def build_self_attention(self, embed_dim, args):
        if getattr(args, "enc_self_attn_cls", "abs") == "abs":
            return MultiheadAttention(
                embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                self_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
            )
        else:
            return RelativeSelfAttention(
                embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                self_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
            )

    def forward(self, x, encoder_padding_mask, attn_mask: Optional[Tensor] = None):
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.bool(), -1e8)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.relative_keys is None:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask,
            )
        else:
            index = utils.new_arange(x, x.size(0))
            pos_key_embed = self.relative_keys(index)
            pos_val_embed = self.relative_vals(index)
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                pos_key=pos_key_embed,
                pos_val=pos_val_embed,
                key_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask,
            )

        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)

        if self.ffn_block is None:
            x = residual + x
        else:
            g = self.ffn_block(residual).sigmoid()
            x = residual * g + x * (1 - g)

        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x


class BlockedDecoderLayer(TransformerDecoderLayer):
    def __init__(
            self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False,
            relative_keys=None, relative_vals=None
    ):
        super().__init__(args, no_encoder_attn, add_bias_kv, add_zero_attn)
        self.ffn_block = FeedForward(
            d_model=self.embed_dim,
            d_hidden=args.decoder_ffn_embed_dim,
            dropout=args.dropout
        ) if args.block_cls == "highway" else None

        self.relative_keys = relative_keys
        self.relative_vals = relative_vals

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, eps=getattr(args, "layer_norm_eps", 1e-5))
        self.final_layer_norm = LayerNorm(self.embed_dim, eps=getattr(args, "layer_norm_eps", 1e-5))
        if self.encoder_attn_layer_norm is not None:
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, eps=getattr(args, "layer_norm_eps", 1e-5))

    def build_self_attention(self, embed_dim, args, add_bias_kv=False, add_zero_attn=False):
        if getattr(args, "self_attn_cls", "abs") == "abs":
            return MultiheadAttention(
                embed_dim,
                args.decoder_attention_heads,
                dropout=args.attention_dropout,
                add_bias_kv=add_bias_kv,
                add_zero_attn=add_zero_attn,
                self_attention=not getattr(args, "cross_self_attention", False),
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
            )
        else:
            return RelativeSelfAttention(
                embed_dim,
                args.decoder_attention_heads,
                dropout=args.attention_dropout,
                add_bias_kv=add_bias_kv,
                add_zero_attn=add_zero_attn,
                self_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
            )

    def forward(
            self,
            x,
            encoder_out: Optional[torch.Tensor] = None,
            encoder_padding_mask: Optional[torch.Tensor] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            prev_self_attn_state: Optional[List[torch.Tensor]] = None,
            prev_attn_state: Optional[List[torch.Tensor]] = None,
            self_attn_mask: Optional[torch.Tensor] = None,
            self_attn_padding_mask: Optional[torch.Tensor] = None,
            need_attn: bool = False,
            need_head_weights: bool = False,
    ):

        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
                incremental_state is not None
                and _self_attn_input_buffer is not None
                and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        if self.relative_keys is None:
            x, attn = self.self_attn(
                query=x,
                key=y,
                value=y,
                key_padding_mask=self_attn_padding_mask,
                incremental_state=incremental_state,
                need_weights=False,
                attn_mask=self_attn_mask,
            )
        else:
            index = utils.new_arange(x, x.size(0))
            pos_key_embed = self.relative_keys(index)
            pos_val_embed = self.relative_vals(index)
            x, attn = self.self_attn(
                query=x,
                key=y,
                value=y,
                pos_key=pos_key_embed,
                pos_val=pos_val_embed,
                key_padding_mask=self_attn_padding_mask,
                incremental_state=incremental_state,
                need_weights=False,
                attn_mask=self_attn_mask,
            )

        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = residual + x
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)

        if self.ffn_block is None:
            x = residual + x
        else:
            g = self.ffn_block(residual).sigmoid()
            x = residual * g + x * (1 - g)

        if not self.normalize_before:
            x = self.final_layer_norm(x)

        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None


class ContentTrainer(nn.Module):
    def __init__(self, args, out, share_state_out=False, share_layer_out=False):
        super().__init__()
        dim_embed, num_embed = out.weight.size(1), out.weight.size(0)
        if share_state_out:
            self.state_proj = out
        else:
            self.state_proj = nn.Linear(dim_embed, num_embed, bias=False)
        self.aggregate_func = getattr(args, "layer_aggregate_func", "mean")
        if share_layer_out:
            self.layer_proj = out
        else:
            self.layer_proj = nn.Linear(dim_embed, num_embed, bias=False)

    def forward(self, x, tgt_tokens, mask=None, window_size=-1, layer_weight=0.0, state_weight=0.0, layer=0):
        loss_ret = {}
        if layer_weight > 0.:
            loss_ret["L{}-layer".format(layer)] = {
                "loss": self._compute_layer_loss(x, tgt_tokens, mask).mean() * layer_weight,
                "factor": layer_weight,
                "no-acc": True
            }
        if state_weight > 0.:
            if window_size == 0:
                loss_ret["L{}-state-W{}".format(layer, window_size)] = {
                    "out": self.state_proj(x),
                    "tgt": tgt_tokens,
                    "mask": mask,
                    "factor": state_weight,
                }
            else:
                loss_ret["L{}-state-W{}".format(layer, window_size)] = {
                    "loss": self._compute_state_loss(x, tgt_tokens, mask, window_size).mean() * state_weight,
                    "factor": state_weight,
                    "no-acc": True
                }
        return loss_ret

    def _compute_state_loss(self, x, tgt_tokens, mask=None, window_size=-1):
        score = self.state_proj(x)  # batch_size, seq_len, vocab_size
        if window_size == -1:
            # means: sequential bag-of-word loss
            return self.sequential_bow(score, tgt_tokens, mask)

        batch_size, seq_len = tgt_tokens.size()
        index = utils.new_arange(score, seq_len)  # [0, ... Seq-LEN]
        left = utils.new_arange(score, window_size) + 1  # [1, ... Window]
        shift = torch.cat([-left, utils.new_arange(score, window_size + 1)], dim=-1)

        _index = index[:, None] + shift[None, :]
        _mask = _index.ge(0) * _index.lt(seq_len)
        _index = _index.clamp(0, seq_len - 1)

        window_index = _index.unsqueeze(0).expand(batch_size, -1, -1).contiguous().view(batch_size, -1)
        _mask = _mask.unsqueeze(0).expand(batch_size, -1, -1).contiguous().view(batch_size, -1)

        window_tgt = tgt_tokens.gather(dim=-1, index=window_index)  # batch_size, -1
        window_mask = mask.gather(dim=-1, index=window_index) * _mask  # batch_size, -1
        return self.window_bow(
            score=score,
            tgt=window_tgt.contiguous().view(batch_size, seq_len, -1),
            mask=window_mask.contiguous().view(batch_size, seq_len, -1)
        )

    def _compute_layer_loss(self, x, tgt_tokens, mask=None):
        feat = x.mean(dim=1) if self.aggregate_func == "mean" else x.sum(dim=1)
        score = self.layer_proj(feat)
        return self.bow(score=score, tgt=tgt_tokens, mask=mask)

    @classmethod
    def bow(cls, score, tgt, mask=None):
        """

        :param score: batch_size, vocab_size
        :param tgt: batch_size, seq_len
        :param mask: batch_size, seq_len
        :param reduction
        :return:
        """
        inputs = -score.log_softmax(dim=-1)
        loss = inputs.gather(dim=-1, index=tgt)  # batch_size, seq_len
        # if reduction:
        if mask is not None:
            mask = mask.float()
            length = mask.sum(dim=-1)
            loss = (loss * mask).sum(dim=-1) / length
        else:
            length = inputs.size(1)
            loss = loss / length
        return loss

    @classmethod
    def window_bow(cls, score, tgt, mask=None):
        """

        :param score: batch_size, seq_len, vocab_size
        :param tgt: batch_size, seq_len, window_size
        :param mask: batch_size, seq_len, window_size
        :return:
        """
        batch_size, seq_len, vocab_size = score.size()
        window_size = tgt.size(-1)

        # # flatten the inputs in dimension 1
        flat_score = score.contiguous().view(-1, vocab_size)
        flat_tgt = tgt.contiguous().view(-1, window_size)
        flat_mask = mask.contiguous().view(-1, window_size) if mask is not None else None

        flat_loss = - flat_score.log_softmax(dim=-1).gather(dim=-1, index=flat_tgt)
        flat_loss *= (flat_mask.float())
        return flat_loss.sum() / (flat_mask.float().sum())

    @classmethod
    def sequential_bow(cls, score, tgt, mask=None):
        """

        :param score: batch_size, seq_len, vocab_size
        :param tgt: batch_size, seq_len
        :param mask: batch_size, seq_len
        :return:
        """
        seq_len = score.size(1)
        flat_tgt = tgt.unsqueeze(1).expand(-1, seq_len, -1).contiguous()  # batch_size, seq_len, seq_len
        flat_mask = mask.unsqueeze(1).expand(-1, seq_len, -1).contiguous() if mask is not None else None
        # batch_size, seq_len, seq_len

        return cls.window_bow(score=score, tgt=flat_tgt, mask=flat_mask)
