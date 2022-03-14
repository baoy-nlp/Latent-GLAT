import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.models.nat.nonautoregressive_transformer import (
    DecoderOut,
    NATransformerModel,
    NATransformerDecoder,
    ensemble_decoder,
    init_bert_params,
    register_model_architecture,
    register_model,
    utils
)
from fairseq.models.transformer import TransformerEncoder
from fairseq.modules import LayerDropModuleList

from .layer import BlockedDecoderLayer, ContentTrainer, BlockedEncoderLayer
from .modules import RelativePositionEmbeddings

INF = 1e10

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


def _softcopy_assignment(src_lens, trg_lens, tau=0.3):
    max_trg_len = trg_lens.max()
    max_src_len = src_lens.max()
    index_s = utils.new_arange(src_lens, max_src_len).float()
    index_t = utils.new_arange(trg_lens, max_trg_len).float()
    diff = -(index_t[:, None] - index_s[None, :]).abs()  # max_trg_len, max_src_len
    diff = diff.unsqueeze(0).expand(trg_lens.size(0), *diff.size())
    mask = (src_lens[:, None] - 1 - index_s[None, :]).lt(0).float()  # batch_size, max_src_lens
    logits = (diff / tau - INF * mask[:, None, :])
    prob = logits.softmax(-1)
    return prob


def _interpolate_assignment(src_lens, tgt_lens, tau=0.3):
    max_tgt_len = tgt_lens.max()
    max_src_len = src_lens.max()
    steps = src_lens.float() / tgt_lens.float()
    index_s = utils.new_arange(tgt_lens, max_src_len).float()  # max_src_len
    index_t = utils.new_arange(tgt_lens, max_tgt_len).float()  # max_trg_len

    index_t = steps[:, None] @ index_t[None, :]  # batch x max_trg_len
    index = (index_s[None, None, :] - index_t[:, :, None]) ** 2
    src_mask = (src_lens[:, None] - index_s[None, :]).lt(0).float()
    index = (-index.float() / tau - INF * (src_mask[:, None, :].float())).softmax(dim=-1)
    return index


def _interpolate(src_masks, tgt_masks, tau=0.3):
    max_src_len = src_masks.size(1)
    max_tgt_len = tgt_masks.size(1)
    src_lens = src_masks.sum(-1).float()
    tgt_lens = tgt_masks.sum(-1).float()
    index_t = utils.new_arange(tgt_masks, max_tgt_len).float()
    index_s = utils.new_arange(tgt_masks, max_src_len).float()
    steps = src_lens / tgt_lens
    index_t = steps[:, None] @ index_t[None, :]  # batch x max_trg_len
    index = (index_s[None, None, :] - index_t[:, :, None]) ** 2
    index = (-index.float() / tau - INF * (1 - src_masks[:, None, :].float())).softmax(dim=-1)
    return index


def build_relative_embeddings(args, embedding_dim=None):
    if embedding_dim is None:
        embedding_dim = args.decoder_embed_dim // getattr(args, "decoder_attention_heads")
    return RelativePositionEmbeddings(
        max_rel_positions=getattr(args, "max_rel_positions", 4),
        embedding_dim=embedding_dim,
        direction=True,
        dropout=args.dropout
    )


@register_model("nat_base")
class NAT(NATransformerModel):
    @staticmethod
    def add_args(parser):
        NATransformerModel.add_args(parser)
        parser.add_argument("--use-ptrn-encoder", action="store_true", help="use pretrained encoder")
        parser.add_argument("--use-ptrn-decoder", action="store_true", help="use pretrained decoder")
        parser.add_argument("--use-ptrn-embed", action="store_true", help="use the word encoder output")
        parser.add_argument("--ptrn-encoder-mode", type=int, default=0, help="0 is None, 1 is finetuning, 2 is fixed")
        parser.add_argument("--ptrn-embed-mode", type=int, default=0, help="0 is None, 1 is finetuning, 2 is fixed")
        parser.add_argument("--no-share-dec-input-output", action='store_true')
        parser.add_argument("--block-cls", type=str, default="None")
        parser.add_argument("--self-attn-cls", type=str, default="abs")
        parser.add_argument("--enc-block-cls", type=str, default="abs")
        parser.add_argument("--enc-self-attn-cls", type=str, default="abs")
        parser.add_argument("--dec-block-cls", type=str, default="abs")
        parser.add_argument("--dec-self-attn-cls", type=str, default="abs")
        parser.add_argument("--max-rel-positions", type=int, default=4)
        parser.add_argument("--share-rel-embeddings", action='store_true')
        parser.add_argument("--layer-norm-eps", type=float, default=1e-5)
        NATDecoder.add_args(parser)

    def fine_tuning_mode(self, args):
        parameters = []
        if getattr(args, "finetune_length_pred", False):
            for p in self.decoder.embed_length.parameters():
                parameters.append(p)
        for p in self.parameters():
            if p in parameters:
                p.requires_grad = True and p.requires_grad
            else:
                p.requires_grad = False

    def forward(
            self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs
    ):
        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, return_all_hiddens=True, **kwargs)

        # decoding
        word_ins_out, other = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            extra_ret=True
        )

        losses = {
            "word_ins": {
                "out": word_ins_out,
                "tgt": tgt_tokens,
                "mask": tgt_tokens.ne(self.pad),
                "ls": self.args.label_smoothing,
                "nll_loss": True
            }
        }
        # length prediction
        if self.decoder.length_loss_factor > 0:
            length_out = self.decoder.forward_length(normalize=False, encoder_out=encoder_out)
            length_tgt = self.decoder.forward_length_prediction(length_out, encoder_out, tgt_tokens)
            losses["length"] = {
                "out": length_out,
                "tgt": length_tgt,
                "factor": self.decoder.length_loss_factor
            }

        # content prediction
        if self.decoder.content_weight > 0.:
            losses = self.decoder.compute_content_loss(
                losses=losses,
                inputs=other,
                tgt_tokens=tgt_tokens,
                mask=tgt_tokens.ne(self.pad)
            )

        return losses

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        ptrn_model = cls.load_pretrained_model(args)

        if getattr(args, "use_ptrn_encoder", False) and ptrn_model is not None:
            encoder = ptrn_model.encoder
        elif getattr(args, "use_ptrn_embed", False) and ptrn_model is not None:
            encoder = cls.build_encoder(args, src_dict, ptrn_model.encoder.embed_tokens)
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            decoder_embed_tokens = encoder.embed_tokens
            args.share_decoder_input_output_embed = True and not getattr(args, "no_share_dec_input_output", False)
        else:
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        model = cls(args, encoder, decoder)
        if getattr(args, "finetune", False):
            print("Fine tune model is started")
            assert ptrn_model is not None, 'ptrn model should not be None while finetuning'
            model.load_state_dict(state_dict=ptrn_model.state_dict(), strict=False)
            model.fine_tuning_mode(args)
        return model

    @classmethod
    def load_pretrained_model(cls, args):
        if getattr(args, "ptrn_model_path", None) is None:
            return None

        from fairseq import checkpoint_utils
        models, _ = checkpoint_utils.load_model_ensemble(
            utils.split_paths(args.ptrn_model_path),
            task=None,
            suffix=getattr(args, "checkpoint_suffix", ""),
        )
        return models[0]

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        if (
                getattr(args, "enc_block_cls", "None") == "highway"
                or getattr(args, "enc_self_attn_cls", "abs") != "abs"
                or getattr(args, "layer_norm_eps", 1e-5) != 1e-5
        ):
            encoder = NATEncoder(args, src_dict, embed_tokens)
        else:
            encoder = TransformerEncoder(args, src_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            encoder.apply(init_bert_params)
        return encoder

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = NATDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):
        """
         used for decoding.
        """
        step = decoder_out.step
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history

        # execute the decoder
        output_masks = output_tokens.ne(self.pad)

        extra_ret, tgt_tokens = kwargs.get("extra_ret", False), kwargs.get("tgt_tokens", None)
        if extra_ret:
            out, ret = self.decoder.forward(normalize=False, prev_output_tokens=output_tokens, encoder_out=encoder_out,
                                            step=step, tgt_tokens=tgt_tokens, extra_ret=extra_ret, )
        else:
            out = self.decoder.forward(
                normalize=False,
                prev_output_tokens=output_tokens,
                encoder_out=encoder_out,
                step=step,
                tgt_tokens=tgt_tokens,
                extra_ret=extra_ret,
            )

        _scores, _tokens = out.max(-1)
        # try:
        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])
        # except:
        #     assert output_masks.size() == _tokens.size(), "there are LENGTH error"
        if history is not None:
            history.append(output_tokens.clone())

        orig = decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history
        )
        if extra_ret:
            return orig, ret
        else:
            return orig

    def initialize_beam_output_tokens(self, encoder_out, src_tokens, tgt_tokens=None, beam_size=1):
        if beam_size == 1:
            return self.initialize_output_tokens(encoder_out, src_tokens, tgt_tokens)

        length_out = self.decoder.forward_length(normalize=True, encoder_out=encoder_out)
        length_tgt = self.decoder.forward_length_prediction(length_out, encoder_out, tgt_tokens, beam_size=beam_size)

        max_length = length_tgt.clamp_(min=2).max()
        idx_length = utils.new_arange(src_tokens, max_length)

        initial_output_tokens = src_tokens.new_zeros(src_tokens.size(0) * beam_size, max_length).fill_(self.pad)
        initial_output_tokens.masked_fill_(idx_length[None, :] < length_tgt[:, None], self.unk)
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)

        initial_output_scores = initial_output_tokens.new_zeros(*initial_output_tokens.size()).type_as(
            encoder_out.encoder_out)

        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None
        )

    def initialize_output_tokens(self, encoder_out, src_tokens, tgt_tokens=None):
        # length prediction
        if isinstance(self.decoder, NATDecoder):
            length_tgt = self.decoder.forward_length_prediction(
                length_out=self.decoder.forward_length(normalize=True, encoder_out=encoder_out),
                encoder_out=encoder_out,
                tgt_tokens=tgt_tokens,
                use_true_length=True,
            )
        else:
            length_tgt = self.decoder.forward_length_prediction(
                length_out=self.decoder.forward_length(normalize=True, encoder_out=encoder_out),
                encoder_out=encoder_out,
                tgt_tokens=tgt_tokens,
            )

        max_length = length_tgt.clamp_(min=2).max()
        idx_length = utils.new_arange(src_tokens, max_length)

        initial_output_tokens = src_tokens.new_zeros(
            src_tokens.size(0), max_length
        ).fill_(self.pad)
        initial_output_tokens.masked_fill_(
            idx_length[None, :] < length_tgt[:, None], self.unk
        )
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out.encoder_out)

        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None
        )

    def regenerate_length_beam(self, decoder_out, beam_size):
        output_tokens = decoder_out.output_tokens
        length_tgt = output_tokens.ne(self.pad).sum(1)
        length_tgt = length_tgt[:, None] + utils.new_arange(length_tgt, 1, beam_size) - beam_size // 2
        length_tgt = length_tgt.view(-1).clamp_(min=2)
        max_length = length_tgt.max()
        idx_length = utils.new_arange(length_tgt, max_length)

        initial_output_tokens = output_tokens.new_zeros(
            length_tgt.size(0), max_length
        ).fill_(self.pad)
        initial_output_tokens.masked_fill_(
            idx_length[None, :] < length_tgt[:, None], self.unk
        )
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(decoder_out.output_scores)

        return decoder_out._replace(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores
        )


class NATEncoder(TransformerEncoder):

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        if getattr(args, "enc_self_attn_cls", "abs") != "abs":
            rel_keys = build_relative_embeddings(args) if getattr(args, "share_rel_embeddings", False) else None
            rel_vals = build_relative_embeddings(args) if getattr(args, "share_rel_embeddings", False) else None
            if self.encoder_layerdrop > 0.0:
                self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
            else:
                self.layers = nn.ModuleList([])
            self.layers.extend(
                [
                    self.build_encoder_layer(args, rel_keys, rel_vals)
                    for _ in range(args.encoder_layers)
                ]
            )

    def build_encoder_layer(self, args, rel_keys=None, rel_vals=None):
        if getattr(args, "enc_self_attn_cls", "abs") == "abs":
            return BlockedEncoderLayer(args)
        else:
            return BlockedEncoderLayer(
                args,
                relative_keys=rel_keys if rel_keys is not None else build_relative_embeddings(args),
                relative_vals=rel_vals if rel_vals is not None else build_relative_embeddings(args),
            )


class NATDecoder(NATransformerDecoder):
    """
    Implementation of vanilla nat models, support vary the decoder inputs:
        - softmax copy
        - positional attention copy
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)

        # decoder inputs
        self.map_func = getattr(args, "mapping_func", "uniform")
        self.map_use = getattr(args, "mapping_use", "embed")

        if getattr(args, "softmax_copy", False):
            self.map_func = "soft"
        if getattr(args, "use_encoder_out", False):
            self.map_use = "output"

        # layer-wise attention
        self.layerwise_attn = getattr(args, "layerwise_attn", False)

        # content trainer
        if self.content_weight > 0:
            self.content = nn.ModuleList(self.build_content_trainer(args, self.output_projection))
            self.state_weight = args.content_state_weight
            self.layer_weight = args.content_layer_weight
            self.window_size = args.content_window_size

        if getattr(args, "self_attn_cls", "abs") != "abs":
            self.embed_positions = None  # TODO check

            rel_keys = build_relative_embeddings(args) if getattr(args, "share_rel_embeddings", False) else None
            rel_vals = build_relative_embeddings(args) if getattr(args, "share_rel_embeddings", False) else None
            if self.decoder_layerdrop > 0.0:
                self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
            else:
                self.layers = nn.ModuleList([])
            self.layers.extend(
                [
                    self.build_decoder_layer(args, no_encoder_attn, rel_keys, rel_vals)
                    for _ in range(args.decoder_layers)
                ]
            )

    @staticmethod
    def add_args(parser):
        parser.add_argument("--softmax-copy", action="store_true",
                            help="softmax-copy replace the uniform copy")
        parser.add_argument("--use-encoder-out", action="store_true",
                            help="use the word encoder output")
        parser.add_argument("--layerwise-attn", action='store_true')

        # input funcition
        parser.add_argument("--mapping-func", type=str, choices=["soft", "uniform", "interpolate"])
        parser.add_argument("--mapping-use", type=str, choices=["embed", "output"])

        # content-training
        parser.add_argument("--content-window-size", nargs="*", type=int, default=[-1, ])
        parser.add_argument("--content-layer-weight", nargs="*", type=float, default=[0.0, ])
        parser.add_argument("--content-state-weight", nargs="*", type=float, default=[0.0, ])
        parser.add_argument("--content-share-state", nargs="*", type=int, default=[0, ])
        parser.add_argument("--content-share-layer", nargs="*", type=int, default=[0, ])
        parser.add_argument("--share-content-all", action="store_true")
        parser.add_argument("--layer-aggregate-func", type=str, default="mean")

    def compute_content_loss(self, losses, inputs, tgt_tokens, mask=None):
        """
        including:
            - Layer Bag-of-word Loss
            - Window Bag-of-word Loss
        """

        inner_states = inputs['inner_states']
        for i, trainer in enumerate(self.content):
            if trainer is not None:
                ret = trainer.forward(
                    x=inner_states[i + 1].transpose(0, 1),
                    tgt_tokens=tgt_tokens,
                    padding_mask=mask,
                    window_size=self.window_size[i],
                    layer_weight=self.layer_weight[i],
                    state_weight=self.state_weight[i],
                    layer=i
                )
                losses.update(ret)
        return losses

    @property
    def content_weight(self):
        return sum(getattr(self.args, "content_layer_weight", [0.0, ])) + sum(
            getattr(self.args, "content_state_weight", [0.0, ]))

    @classmethod
    def build_content_trainer(cls, args, out):
        num_layer = args.decoder_layers

        if len(args.content_share_state) <= num_layer:
            args.content_share_state += ([0] * (num_layer - len(args.content_share_state)))

        if len(args.content_share_layer) <= num_layer:
            args.content_share_layer += ([0] * (num_layer - len(args.content_share_layer)))

        if len(args.content_layer_weight) <= num_layer:
            args.content_layer_weight += ([0.0] * (num_layer - len(args.content_layer_weight)))

        if len(args.content_state_weight) <= num_layer:
            args.content_state_weight += ([0.0] * (num_layer - len(args.content_state_weight)))

        if len(args.content_window_size) <= num_layer:
            args.content_window_size += ([-1] * (num_layer - len(args.content_window_size)))

        if getattr(args, "share_content_all", False):
            trainer = ContentTrainer(
                args=args,
                out=out,
                share_state_out=bool(sum(args.content_share_state)),
                share_layer_out=bool(sum(args.content_share_layer))
            )
        else:
            trainer = None

        contents = []
        for i in range(num_layer):
            if (args.content_layer_weight[i] + args.content_state_weight[i]) > 0:
                _trainer = trainer if trainer is not None else ContentTrainer(
                    args=args,
                    out=out,
                    share_state_out=bool(args.content_share_state[i]),
                    share_layer_out=args.content_share_layer[i]
                )
                contents.append(_trainer)

        return contents

    def build_decoder_layer(self, args, no_encoder_attn=False, rel_keys=None, rel_vals=None):
        if getattr(args, "block_cls", "None") == "highway" or getattr(args, "self_attn_cls", "abs") != "abs":
            if getattr(args, "self_attn_cls", "abs") == "abs":
                return BlockedDecoderLayer(args, no_encoder_attn)
            else:
                return BlockedDecoderLayer(
                    args, no_encoder_attn=no_encoder_attn,
                    relative_keys=rel_keys if rel_keys is not None else build_relative_embeddings(args),
                    relative_vals=rel_vals if rel_vals is not None else build_relative_embeddings(args),
                )

        return super().build_decoder_layer(args, no_encoder_attn)

    @ensemble_decoder
    def forward(self, normalize, encoder_out, prev_output_tokens, step=0, extra_ret=False, **unused):
        features, other = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            embedding_copy=(step == 0) & self.src_embedding_copy,
        )
        decoder_out = self.output_layer(features)
        decoder_out = F.log_softmax(decoder_out, -1) if normalize else decoder_out
        if extra_ret:
            return decoder_out, other
        else:
            return decoder_out

    def extract_features(
            self,
            prev_output_tokens,
            encoder_out=None,
            early_exit=None,
            embedding_copy=False,
            **unused
    ):

        x, decoder_padding_mask, _ = self.forward_decoder_inputs(prev_output_tokens, encoder_out=encoder_out)

        x = x.transpose(0, 1)
        attn = None
        inner_states = [x]

        # decoder layers
        for i, layer in enumerate(self.layers):

            # early exit from the decoder.
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

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": attn, "inner_states": inner_states}

    def forward_decoder_inputs(self, prev_output_tokens, encoder_out=None, add_position=True, **unused):
        # forward source representation
        mapping_use = self.map_use
        mapping_func = self.map_func

        if mapping_use == 'embed':
            src_embed = encoder_out.encoder_embedding
        else:
            src_embed = encoder_out.encoder_out.contiguous().transpose(0, 1)

        src_mask = encoder_out.encoder_padding_mask
        src_mask = (
            ~src_mask
            if src_mask is not None
            else prev_output_tokens.new_ones(*src_embed.size()[:2]).bool()
        )
        tgt_mask = prev_output_tokens.ne(self.padding_idx)

        states = None  # indicate to the

        if mapping_func == 'uniform':
            states = self.forward_copying_source(
                src_embed, src_mask, prev_output_tokens.ne(self.padding_idx)
            )

        if mapping_func == "soft":
            length_sources = src_mask.sum(1)
            length_targets = tgt_mask.sum(1)
            mapped_logits = _softcopy_assignment(length_sources, length_targets)  # batch_size, tgt_len, src_len
            states = torch.bmm(mapped_logits, src_embed)

        if mapping_func == "interpolate":
            # length_sources = src_mask.sum(1)
            # length_targets = tgt_mask.sum(1)
            # mapped_logits = _interpolate_assignment(length_sources, length_targets)  # batch_size, tgt_len, src_len
            mapped_logits = _interpolate(src_mask, tgt_mask)
            states = torch.bmm(mapped_logits, src_embed)

        x, decoder_padding_mask, positions = self.forward_embedding(prev_output_tokens, states, add_position)
        return x, decoder_padding_mask, positions

    def forward_embedding(self, prev_output_tokens, states=None, add_position=True):
        # embed positions
        positions = (
            self.embed_positions(prev_output_tokens)
            if self.embed_positions is not None
            else None
        )

        # embed tokens and positions
        if states is None:
            x = self.embed_scale * self.embed_tokens(prev_output_tokens)
            if self.project_in_dim is not None:
                x = self.project_in_dim(x)
        else:
            x = states

        if positions is not None and add_position:
            x += positions
        x = self.dropout_module(x)
        decoder_padding_mask = prev_output_tokens.eq(self.padding_idx)
        return x, decoder_padding_mask, positions

    def forward_length_prediction(self, length_out, encoder_out, tgt_tokens=None, use_true_length=False, beam_size=1):
        if tgt_tokens is not None and use_true_length:  # only used for oracle training.
            length_tgt = tgt_tokens.ne(self.padding_idx).sum(1).long()
            return length_tgt

        enc_feats = encoder_out.encoder_out  # T x B x C
        src_masks = encoder_out.encoder_padding_mask  # B x T or None
        src_lengs = None
        if self.pred_length_offset:
            if src_masks is None:
                src_lengs = enc_feats.new_ones(enc_feats.size(1)).fill_(enc_feats.size(0))
            else:
                src_lengs = (~src_masks).transpose(0, 1).type_as(enc_feats).sum(0)
            src_lengs = src_lengs.long()

        if tgt_tokens is not None:
            # obtain the length target
            tgt_lengs = tgt_tokens.ne(self.padding_idx).sum(1).long()
            if self.pred_length_offset:
                length_tgt = tgt_lengs - src_lengs + 128
            else:
                length_tgt = tgt_lengs
            length_tgt = length_tgt.clamp(min=0, max=255)

        else:
            if beam_size == 1:
                pred_lengs = length_out.max(-1)[1]
                if self.pred_length_offset:
                    length_tgt = pred_lengs - 128 + src_lengs
                else:
                    length_tgt = pred_lengs
            else:
                pred_lengs = length_out.topk(beam_size)[1]  # batch_size, beam_size
                if self.pred_length_offset:
                    src_lengs = src_lengs.unsqueeze(-1).expand(-1, beam_size)
                    length_tgt = pred_lengs - 128 + src_lengs
                else:
                    length_tgt = pred_lengs
                length_tgt = length_tgt.squeeze(0)

        return length_tgt


def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # --- special arguments ---
    args.sg_length_pred = getattr(args, "sg_length_pred", False)
    args.pred_length_offset = getattr(args, "pred_length_offset", False)
    args.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
    args.src_embedding_copy = getattr(args, "src_embedding_copy", False)
    args.softmax_copy = getattr(args, "softmax_copy", False)
    args.use_encoder_out = getattr(args, "use_encoder_out", False)
    from fairseq.models.transformer import base_architecture
    base_architecture(args)


@register_model_architecture('nat_base', 'nat_iwslt16')
def nat_iwslt16_de_en(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 278)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 507)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 2)
    args.encoder_layers = getattr(args, 'encoder_layers', 5)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 278)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 507)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 2)
    args.decoder_layers = getattr(args, 'decoder_layers', 5)
    base_architecture(args)


@register_model_architecture('nat_base', 'nat_iwslt14_en_de')
def nat_iwslt14_de_en(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    base_architecture(args)


@register_model_architecture('nat_base', 'nat_small')
def nat_small(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 256)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 5)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 256)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 5)
    base_architecture(args)


@register_model_architecture('nat_base', 'nat_base')
def nat_base(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 512)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    base_architecture(args)


@register_model_architecture('nat_base', 'nat_iwslt')
def nat_iwslt(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 400)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 800)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 3)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 400)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 800)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    base_architecture(args)


@register_model_architecture('nat_base', 'nat_iwslt14')
def nat_iwslt14(args):
    nat_iwslt14_de_en(args)


@register_model_architecture(
    "nat_base", "nat_wmt_en_de"
)
def nat_wmt_en_de(args):
    base_architecture(args)
