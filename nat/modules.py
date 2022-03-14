from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.modules.multihead_attention import MultiheadAttention
from torch import Tensor
from torch.nn import Parameter


class RelativePositionEmbeddings(nn.Module):
    """
    learned relative position embedding for self-attention with relative position of shaw et al
    """

    def __init__(self, max_rel_positions, embedding_dim, dropout=0.0, direction=True, **params):
        super().__init__()
        self.window_size = max_rel_positions
        self.embedding_dim = embedding_dim
        self.direction = direction

        num_embeddings = max_rel_positions * 2 + 1 if self.direction else max_rel_positions + 1
        self.embeddings = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def map_to_index(self, distance, shift_to_zero=True):
        max_rel_len = self.window_size
        if max_rel_len is None:
            distance = distance
        else:
            distance = distance.clamp(-max_rel_len, max_rel_len)

        if self.direction:
            if shift_to_zero and max_rel_len is not None:
                distance = distance + max_rel_len
            else:
                distance = distance
        else:
            distance = distance.abs()
        return distance

    def forward(self, inputs):
        """
        :param inputs: length, length, num_embeddings or length
        :return:
        """
        if inputs.dim() > 2:
            embed = inputs @ self.embeddings.weight
            embed = self.dropout(embed)
            return embed
        elif inputs.dim() == 2:
            distance = inputs
        else:
            inputs = inputs.squeeze()
            distance = inputs[:, None] - inputs[None, :]

        distance = self.map_to_index(distance)
        embed = self.embeddings(distance)
        embed = self.dropout(embed)
        return embed


def matmul(x, y):
    if x.dim() == y.dim():
        return x @ y
    if x.dim() == y.dim() - 1:
        return (x.unsqueeze(-2) @ y).squeeze(-2)
    return (x @ y.unsqueeze(-1)).squeeze(-1)


def shaw_attention(query, key, pos_key):
    """

        :param query:
        :param key:
        :param pos_key: length, length, depth
        :return:
        """
    bsize, heads, length, depth = key.size()

    q_dot_k = matmul(query, key.contiguous().transpose(-1, -2))  # batch, heads, length, length

    query_for_pos = query.contiguous().permute(2, 0, 1, 3).view(length, bsize * heads, depth)
    pos_for_att = pos_key.contiguous().transpose(-2, -1)  # length, depth, length

    q_dot_p = matmul(query_for_pos, pos_for_att)  # length, batch*heads, length
    q_dot_p = q_dot_p.contiguous().permute(1, 0, 2).view(bsize, heads, length, length)

    return q_dot_k + q_dot_p


def shaw_combine(probs, value, pos_val):
    """

    :param probs:
    :param value:
    :param pos_val: length, length, depth
    :return:
    """
    bsize, heads, length, depth = value.size()

    w_dot_v = matmul(probs, value)  # batch, head, length, depth

    w_for_comb = probs.contiguous().permute(2, 0, 1, 3).view(length, bsize * heads, length)
    w_dot_p = matmul(w_for_comb, pos_val)  # length,batch*heads, depth
    w_dot_p = w_dot_p.contiguous().permute(1, 0, 2).view(bsize, heads, length, depth)

    return w_dot_v + w_dot_p


class RelativeSelfAttention(MultiheadAttention):
    """Multi-headed attention with relative attentions.

    See "Self Attention with relative positions" for more details.
    """

    @classmethod
    def relative_attention(cls, query, key, pos_key):
        if pos_key.dim() == 3:
            return shaw_attention(query, key, pos_key)

    @classmethod
    def relative_combine(cls, probs, value, pos_val):
        if pos_val.dim() == 3:
            return shaw_combine(probs, value, pos_val)

    def forward(
            self,
            query,
            key: Optional[Tensor],
            value: Optional[Tensor],
            pos_key=None,
            pos_val=None,
            key_padding_mask: Optional[Tensor] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            need_weights: bool = True,
            static_kv: bool = False,
            attn_mask: Optional[Tensor] = None,
            before_softmax: bool = False,
            need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:

        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        # self-attention
        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)

        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                    ],
                    dim=1,
                )

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if k is not None:
            k = (k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1))

        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: Optional[Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k is not None and v is not None
            key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=bsz,
                src_len=k.size(1),
                static_kv=static_kv,
            )

            saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        assert k is not None
        src_len = k.size(1)

        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0), 1).type_as(
                            key_padding_mask
                        ),
                    ],
                    dim=1,
                )

        attn_weights = self.relative_attention(
            q.contiguous().view(bsz, self.num_heads, -1, self.head_dim),
            k.contiguous().view(bsz, self.num_heads, -1, self.head_dim),
            pos_key,
        ).contiguous().view(bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if not self.tpu:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf")
                )
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float('-inf'))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = utils.softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace
        )
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        attn = self.relative_combine(
            probs=attn_probs.contiguous().view(bsz, self.num_heads, tgt_len, src_len),
            value=v.contiguous().view(bsz, self.num_heads, -1, self.head_dim),
            pos_val=pos_val
        ).contiguous().view(bsz * self.num_heads, -1, self.head_dim)

        if self.onnx_trace and attn.size(1) == 1:
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights: Optional[Tensor] = None

        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights


class FFNAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias=False):
        super(FFNAttention, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.q_proj = nn.Linear(input_dim, hidden_dim)
        self.k_proj = nn.Conv1d(input_dim, hidden_dim, 1, 1)
        self.out = nn.Linear(hidden_dim, 1, bias=bias)
        self._inf = Parameter(torch.Tensor([-1e18]), requires_grad=False)
        self.inf = None

        # Initialize vector V
        nn.init.uniform_(self.out.weight, -1, 1)

    def forward(self, query, key, mask=None):
        query = self.q_proj(query).unsqueeze(2).expand(-1, -1, key.size(1))  # (batch, hidden, seq_len)
        key = key.permute(0, 2, 1)  # (batch, hidden, seq_len)
        key = self.k_proj(key)  # (batch, hidden, seq_len)

        attn_weight = self.out((query + key).permute(0, 2, 1)).squeeze(-1)  # (batch, seq_len)
        if mask is not None and len(attn_weight[mask]) > 0:
            attn_weight[mask] = self.inf[mask]

        attn_prob = attn_weight.softmax(dim=-1)
        attn = torch.bmm(key, attn_prob.unsqueeze(2)).squeeze(2)
        return attn, attn_weight

    def init_inf(self, mask_size):
        self.inf = self._inf.unsqueeze(1).expand(*mask_size)


class DotProductAttention(nn.Module):
    """ Attention model for Pointer-Net """

    def __init__(self, ninp, nhid):
        """
        Initiate Attention

        :param int ninp: Input's diamention
        :param int nhid: Number of hidden units in the attention
        """

        super(DotProductAttention, self).__init__()

        self.input_dim = ninp
        self.hidden_dim = nhid

        self.input_linear = nn.Linear(ninp, nhid)
        self.context_linear = nn.Conv1d(ninp, nhid, 1, 1)
        self.V = Parameter(torch.FloatTensor(nhid), requires_grad=True)
        self._inf = Parameter(torch.FloatTensor([-1e18]), requires_grad=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.inf = None

        # Initialize vector V
        nn.init.uniform_(self.V, -1, 1)

    def forward(self, inputs, context, mask):
        """
        Attention - Forward-pass

        :param Tensor inputs: Hidden state h
        :param Tensor context: Attention context
        :param ByteTensor mask: Selection mask
        :return: tuple of - (Attentioned hidden state, Alphas)
        """

        # (batch, hidden_dim, seq_len)
        inp = self.input_linear(inputs).unsqueeze(2).expand(-1, -1, context.size(1))

        # (batch, hidden_dim, seq_len)
        context = context.permute(0, 2, 1)
        ctx = self.context_linear(context)

        # (batch, 1, hidden_dim)
        V = self.V.unsqueeze(0).expand(context.size(0), -1).unsqueeze(1)

        # (batch, seq_len)
        attn_weight = torch.bmm(V, self.tanh(inp + ctx)).squeeze(1)
        if mask is not None and len(attn_weight[mask]) > 0:
            attn_weight[mask] = self.inf[mask]
        attn_prob = self.softmax(attn_weight)

        attn = torch.bmm(ctx, attn_prob.unsqueeze(2)).squeeze(2)

        return attn, attn_weight

    def init_inf(self, mask_size):
        self.inf = self._inf.unsqueeze(1).expand(*mask_size)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_hidden, dropout=0.0):
        super().__init__()
        # dropout = 0.0 # means 17
        self.input_to_hidden = nn.Linear(d_model, d_hidden)
        self.hidden_to_output = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        h = F.relu(self.input_to_hidden(inputs))
        h = self.dropout(h)
        return self.hidden_to_output(h)
