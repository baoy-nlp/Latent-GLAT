import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models.transformer import (
    FairseqDropout, LayerDropModuleList, LayerNorm, TransformerEncoderLayer
)

try:
    from fairseq.models.transformer import EncoderOut
except ImportError:
    from fairseq.models.fairseq_encoder import EncoderOut


def glancing_sampling(targets, padding_mask, ratio=0.5, logits=None, n_mode="adaptive", s_mode="uniform"):
    """return the positions to be replaced """
    if n_mode == "fixed":
        number = targets.size(1) * ratio + 1
    elif n_mode == "adaptive":
        # E * f_ratio: proposed by Qian et al.
        assert logits is not None, "logits should not be None"
        predict = logits.max(dim=-1)[1]
        distance = (predict.ne(targets) * ~padding_mask).float().sum(dim=-1)
        number = distance * ratio + 1
    elif n_mode == "adaptive-uni":
        # E * random ratio: Uniform sampling ratio for the model.
        assert logits is not None, "logits should not be None"
        ratio = random.random()
        predict = logits.max(dim=-1)[1]
        distance = (predict.ne(targets) * ~padding_mask).float().sum(dim=-1)
        number = distance * ratio + 1
    elif n_mode == "adaptive-rev":
        # E * (1-E/N): The more predicting error, the more sampling token
        predict = logits.max(dim=-1)[1]
        distance = (predict.ne(targets) * ~padding_mask).float().sum(dim=-1)
        ratio = 1.0 - distance / ((~padding_mask).float())
        number = distance * ratio + 1
    else:
        number = None

    score = targets.clone().float().uniform_()

    if s_mode == "uniform":
        # select replaced token from uniform distributions
        assert number is not None, "number should be decided before sampling"
        score.masked_fill_(padding_mask, 2.0)
        rank = score.sort(1)[1]
        cutoff = utils.new_arange(rank) < number[:, None].long()
        sample = cutoff.scatter(1, rank, cutoff)  # batch_size, sequence_length
    elif s_mode == "schedule":
        # select the replaced token with its modeled y probability
        assert logits is not None, "logits should not be None"
        prob = logits.softmax(dim=-1)
        ref_score = prob.view(-1, targets.size(-1)).contiguous().gather(1, targets.view(-1, 1)).view(*targets.size())
        sample = score.lt(ref_score) * (~padding_mask)
    else:
        sample = None

    return sample


def reparameterize(mean, var, is_logv=False, sample_size=1):
    if sample_size > 1:
        mean = mean.contiguous().unsqueeze(1).expand(-1, sample_size, -1).reshape(-1, mean.size(-1))
        var = var.contiguous().unsqueeze(1).expand(-1, sample_size, -1).reshape(-1, var.size(-1))

    if not is_logv:
        sigma = torch.sqrt(var + 1e-10)
    else:
        sigma = torch.exp(0.5 * var)

    epsilon = torch.randn_like(sigma)
    z = mean + epsilon * sigma
    return z


class GaussianVariable(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.mean = nn.Linear(input_dim, latent_dim)
        self.logv = nn.Linear(input_dim, latent_dim)
        self.rec = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, output_dim),
            nn.Tanh(),
        )

    def forward(self, inputs, max_posterior=False, **kwargs):
        """
        :param inputs:  batch_size,input_dim
        :param max_posterior:
        :return:
            mean: batch_size, latent_dim
            logv: batch_size, latent_dim
            z: batch_size, latent_dim
            rec: batch_size, output_dim
        """
        mean, logv, z = self.posterior(inputs, max_posterior=max_posterior)

        rec = self.rec(z)

        return {"mean": mean, "logv": logv, "z": z, "rec": rec}

    def posterior(self, inputs, max_posterior=False):
        mean = self.mean(inputs)
        logv = self.logv(inputs)
        z = reparameterize(mean, logv, is_logv=True) if not max_posterior else mean
        return mean, logv, z

    def prior(self, inputs, n=-1):
        if n < 0:
            n = inputs.size(0)
        z = torch.randn([n, self.latent_dim])

        if inputs is not None:
            z = z.to(inputs)

        return z


class GateNet(nn.Module):
    def __init__(self, d_model, d_hidden, d_output, dropout=0.0):
        super().__init__()
        self.input_to_hidden = nn.Linear(d_model, d_hidden)
        self.hidden_to_output = nn.Linear(d_hidden, d_output)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        h = F.relu(self.input_to_hidden(inputs))
        h = self.dropout(h)
        return self.hidden_to_output(h)


class TransformerEncoderNet(nn.Module):
    """
    remove embedding layer
    The args need includes:
        - dropout
        - encoder_layer_drop
        - encoder_layers

        - embed_dim or encoder_embed_dim
        - encoder_attention_heads
        - attention_dropout
        - encoder_normalize_before
        - encoder_ffn_embed_dim

    """

    def __init__(self, args):
        super().__init__()
        self.dropout_module = FairseqDropout(args.dropout, module_name=self.__class__.__name__)
        self.encoder_layerdrop = args.encoder_layerdrop

        self.embed_dim = getattr(args, "embed_dim", args.encoder_embed_dim)

        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend([TransformerEncoderLayer(args) for _ in range(args.latent_layers)])
        self.num_layers = len(self.layers)

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(self.embed_dim)
        else:
            self.layer_norm = None

    def forward(self, embedding, padding_mask, return_all_hiddens: bool = False):

        x = embedding.transpose(0, 1)
        encoder_states = [] if return_all_hiddens else None

        # encoder layers
        for layer in self.layers:
            x = layer(x, padding_mask)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=padding_mask,  # B x T
            encoder_embedding=embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=None,
            src_lengths=None,
        )
