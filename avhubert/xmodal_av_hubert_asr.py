# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import sys,logging
import contextlib
import tempfile
from argparse import Namespace
from typing import Any, Optional
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass import ChoiceEnum
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import BaseFairseqModel, FairseqEncoder, FairseqEncoderDecoderModel, register_model
from fairseq.modules import LayerNorm, MultiheadAttention, SamePad
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.utils import buffered_arange, index_put, is_xla_tensor

DBG=True if len(sys.argv) == 1 else False

if DBG:
    from decoder import TransformerDecoder
    from hubert_asr import AVHubertAsrConfig
    from xmodal_layer_modules import XmodalSimpleAttentionLayer, XmodalTransformerSentenceEncoderLayer
    from xmodal_temporal_modules import TemporalFlowPredictor, TemporalVAFlowPredictor, TemporalDirectionPredictor, TemporalSpeedPredictor
else:
    from .decoder import TransformerDecoder
    from .hubert_asr import AVHubertAsrConfig
    from .xmodal_layer_modules import XmodalSimpleAttentionLayer, XmodalTransformerSentenceEncoderLayer
    from .xmodal_temporal_modules import TemporalFlowPredictor, TemporalVAFlowPredictor, TemporalDirectionPredictor, TemporalSpeedPredictor

logger = logging.getLogger(__name__)


@dataclass
class XmodalAVHubertSeq2SeqConfig(AVHubertAsrConfig):
    decoder_embed_dim: int = field(
        default=768, metadata={"help": "decoder embedding dimension"}
    )
    decoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "decoder embedding dimension for FFN"}
    )
    decoder_layers: int = field(
        default=6, metadata={"help": "num of decoder layers"}
    )
    decoder_layerdrop: float = field(
        default=0.0, metadata={"help": "decoder layerdrop chance"}
    )
    decoder_attention_heads: int = field(
        default=4, metadata={"help": "num decoder attention heads"}
    )
    decoder_learned_pos: bool = field(
        default=False,
        metadata={"help": "use learned positional embeddings in the decoder"},
    )
    decoder_normalize_before: bool = field(
        default=False,
        metadata={"help": "apply layernorm before each decoder block"},
    )
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if set, disables positional embeddings "
            "(outside self attention)"
        },
    )
    decoder_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability in the decoder"}
    )
    decoder_attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights "
            "inside the decoder"
        },
    )
    decoder_activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN "
            "inside the decoder"
        },
    )
    max_target_positions: int = field(
        default=2048, metadata={"help": "max target positions"}
    )
    share_decoder_input_output_embed: bool = field(
        default=False,
        metadata={"help": "share decoder input and output embeddings"},
    )
    no_scale_embedding: bool = field(default=True, metadata={'help': 'scale embedding'})

    # cross-modal attention
    xmodal_layers: int = field(
        default=3, metadata={"help": "num encoder layers in the transformer"}
    )
    xmodal_layers_type: str = field(
        default='sa-ca', metadata={"help": "xmodal transformer layers type"}
    )
    xmodal_embed_dim: int = field(
        default=1024, metadata={"help": "encoder embedding dimension"}
    )
    xmodal_ffn_embed_dim: int = field(
        default=4096, metadata={"help": "encoder embedding dimension for FFN"}
    )
    xmodal_attention_heads: int = field(
        default=8, metadata={"help": "num encoder attention heads"}
    )
    xmodal_attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights"
        },
    )
    xmodal_activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN"
        },
    )
    xmodal_activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="gelu", metadata={"help": "activation function to use"}
    )
    xmodal_conv_pos: int = field(
        default=128,
        metadata={"help": "number of filters for convolutional positional embeddings"},
    )
    xmodal_conv_pos_groups: int = field(
        default=16,
        metadata={"help": "number of groups for convolutional positional embedding"},
    )
    temporal_flow: bool = field(
        default=False, metadata={"help": "whether to use video-video order temporal flow loss"}
    )
    temporal_va_flow: bool = field(
        default=False, metadata={"help": "whether to use video-audio order temporal flow loss"}
    )
    temporal_direction: bool = field(
        default=False, metadata={"help": "whether to use temporal direction loss"}
    )
    temporal_speed: bool = field(
        default=False, metadata={"help": "whether to use temporal speed loss"}
    )
    temp_len: int = field(
        default=1, metadata={"help": "temporal bulk length"}
    )
    dir_temp_len: int = field(
        default=1, metadata={"help": "temporal bulk length - direction & speed"}
    )
    temp_hidden_dim: int = field(
        default=1024, metadata={"help": "temporal hidden dimension"}
    )
    xmodal_encoder_type: str = field(
        default="xmodaltransformerencoder",
        metadata={"help": "xmodal transformer encoder type"},
    )
    xmodal_encoder_layer_type: str = field(
        default="simpleattention",
        metadata={"help": "xmodal transformer encoder layer type"},
    )
    use_aggregator: bool = field(
        default=True, metadata={"help": "use aggregator conv1d"}
    )
    aggregator_kernel_size: int = field(
        default=3, metadata={"help": "aggregator kernel size"}
    )


class TemporaryGrad(object):
    def __enter__(self):
        self.prev = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
    def __exit__(self, exc_type: any, exc_value: any, traceback: any) -> None:
        torch.set_grad_enabled(self.prev)

class HubertEncoder(FairseqEncoder):
    def __init__(self, cfg: AVHubertAsrConfig, tgt_dict=None):
        self.apply_mask = cfg.apply_mask

        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
        }

        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(
                cfg.w2v_path, arg_overrides
            )
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            cfg.w2v_args = w2v_args
        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(
                    w2v_args
                )

        assert cfg.normalize == w2v_args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for "
            "both pre-training and here"
        )

        w2v_args.task.data = cfg.data

        task = tasks.setup_task(w2v_args.task)
        model = task.build_model(w2v_args.model)

        if state is not None and not cfg.no_pretrained_weights:
            # set strict=False because we omit some modules
            model.load_state_dict(state["model"], strict=False)

        model.remove_pretraining_modules()

        super().__init__(task.source_dictionary)

        d = model.encoder.embedding_dim

        self.w2v_model = model

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

        if tgt_dict is not None:
            self.proj = Linear(d, len(tgt_dict))
        elif getattr(cfg, "decoder_embed_dim", d) != d:
            self.proj = Linear(d, cfg.decoder_embed_dim)
        else:
            self.proj = None

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, source, padding_mask, tbc=True, **kwargs):

        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }
        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            x, padding_mask = self.w2v_model.extract_finetune(**w2v_args)

            if tbc:
                # B x T x C -> T x B x C
                x = x.transpose(0, 1)

        x = self.final_dropout(x)

        if self.proj:
            x = self.proj(x)

        return {
            "encoder_out": x,  # T x B x C
            "encoder_padding_mask": padding_mask,  # B x T
            "padding_mask": padding_mask,
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out[
                "encoder_out"
            ].index_select(1, new_order)
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict


class HubertEncoderWrapper(FairseqEncoder):
    def __init__(self, cfg, w2v_model, xmodal_encoders):
        super().__init__(None)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.w2v_model = w2v_model
        self.xmodal_encoder_video, self.xmodal_encoder_audio = xmodal_encoders

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, source, padding_mask, **kwargs):
        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
        }
        ft = self.freeze_finetune_updates <= self.num_updates if hasattr(self, 'num_updates') else False
        with torch.no_grad() if not ft else contextlib.ExitStack():
            features_dict = self.w2v_model.extract_finetune_features(output_clean=True, **w2v_args)
        
            f_v, f_a, f_a_c = features_dict["video"], features_dict["audio"], features_dict["clean_audio"]            
            features_dict["audio"], layer_results_audio = self.xmodal_encoder_audio(f_a.clone(), f_v.clone(), padding_mask)

            f_a_n = layer_results_audio["xattn_feat"].clone().detach()
            features_dict["video"], layer_results_video = self.xmodal_encoder_video(f_v.clone(), f_a_n.transpose(1, 2), padding_mask)

            if f_a_c is not None:
                if self.xmodal_encoder_audio.xmodal_layers_type != 'no-aud':
                    layer_results_audio["feat"] = f_a_c.clone().transpose(1, 2)  # B x C x T -> B x T x C
            else:  # clean-aud setting
                layer_results_audio = {}

            w2v_args["features"] = features_dict
            x, padding_mask = self.w2v_model.extract_finetune_encoder(**w2v_args)

            # B x T x C -> T x B x C
            x = x.transpose(0, 1)

        return {
            "encoder_out": x,  # T x B x C
            "encoder_padding_mask": padding_mask,  # B x T
            "padding_mask": padding_mask,
            "layer_results": (layer_results_video, layer_results_audio)
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out[
                "encoder_out"
            ].index_select(1, new_order)
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(0, new_order)
        if encoder_out["padding_mask"] is not None:
            encoder_out["padding_mask"] = encoder_out[
                "padding_mask"
            ].index_select(0, new_order)
        return encoder_out
    

class XmodalTransformerEncoder(nn.Module):
    def __init__(self, modality, args):
        super().__init__()
        self.modality = modality
        self.dropout = 0.1
        self.layerdrop = 0.0
        self.xmodal_layer_norm_first = True
        self.embedding_dim = args.xmodal_embed_dim
        self.temporal_flow = args.temporal_flow
        self.temporal_va_flow = args.temporal_va_flow
        self.temporal_direction = args.temporal_direction
        self.temporal_speed = args.temporal_speed
        self.temp_len = args.temp_len
        self.dir_temp_len = args.dir_temp_len
        self.temp_hidden_dim = args.temp_hidden_dim
        self.use_aggregator = args.use_aggregator
        self.aggregator_kernel_size = args.aggregator_kernel_size
        self.xmodal_layers_type = args.xmodal_layers_type
        self.xmodal_encoder_layer_type = args.xmodal_encoder_layer_type

        self.pos_conv = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=args.xmodal_conv_pos,
            padding=args.xmodal_conv_pos // 2,
            groups=args.xmodal_conv_pos_groups,
        )
        dropout = 0
        std = math.sqrt((4 * (1.0 - dropout)) / (args.xmodal_conv_pos * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(args.xmodal_conv_pos), nn.GELU())

        if args.xmodal_layers_type == 'ca':
            use_cross_attention = [True]
        elif args.xmodal_layers_type == 'sa':
            use_cross_attention = [False]
        elif args.xmodal_layers_type == 'sa-ca':
            use_cross_attention = [False]*(args.xmodal_layers-1) + [True]
        else:
            raise NotImplementedError
        
        if self.xmodal_encoder_layer_type == "simpleattention":
            xmodal_encoder_layer = XmodalSimpleAttentionLayer
        elif self.xmodal_encoder_layer_type == "bottleneck":
            xmodal_encoder_layer = XmodalTransformerSentenceEncoderLayer
        else:
            raise NotImplementedError
        self.layers = nn.ModuleList(
            [
                xmodal_encoder_layer(
                    cross_attention=uca,
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=args.xmodal_ffn_embed_dim,
                    num_attention_heads=args.xmodal_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=args.xmodal_attention_dropout,
                    activation_dropout=args.xmodal_activation_dropout,
                    activation_fn=args.xmodal_activation_fn,
                    layer_norm_first=self.xmodal_layer_norm_first,
                )
                for i, uca in zip(range(args.xmodal_layers), use_cross_attention)
            ]
        )

        self.layer_norm = LayerNorm(self.embedding_dim)

        if self.modality == 'video':
            if self.temporal_flow:
                self.flow_predictor = TemporalFlowPredictor(
                    self.embedding_dim, 
                    temp_len=self.temp_len, 
                    temp_hidden_dim=self.temp_hidden_dim,
                    use_aggregator=self.use_aggregator,
                    aggregator_kernel_size=self.aggregator_kernel_size,
                )
            if self.temporal_va_flow:
                self.flow_va_predictor = TemporalVAFlowPredictor(
                    self.embedding_dim, 
                    temp_len=self.temp_len, 
                    temp_hidden_dim=self.temp_hidden_dim,
                    use_aggregator=self.use_aggregator,
                    aggregator_kernel_size=self.aggregator_kernel_size,
                )
            if self.temporal_direction:
                self.direction_predictor = TemporalDirectionPredictor(
                    self.embedding_dim, 
                    temp_len=self.dir_temp_len,
                    temp_hidden_dim=self.temp_hidden_dim,
                    use_aggregator=self.use_aggregator,
                    aggregator_kernel_size=self.aggregator_kernel_size,
                )
            if self.temporal_speed:
                self.speed_predictor = TemporalSpeedPredictor(
                    self.embedding_dim, 
                    temp_len=self.dir_temp_len,
                    temp_hidden_dim=self.temp_hidden_dim,
                    use_aggregator=self.use_aggregator,
                    aggregator_kernel_size=self.aggregator_kernel_size,
                )

        self.apply(init_bert_params)

    def forward(self, x, y, padding_mask=None, layer=None):
        # B x C x T -> B x T x C
        x, y = x.transpose(1, 2), y.transpose(1, 2)
        x, layer_results = self.extract_features(x, y, padding_mask, layer)
        # B x T x C -> B x C x T
        x = x.transpose(1, 2)

        return x, layer_results

    def extract_features(self, x, y, padding_mask=None, tgt_layer=None):
        if padding_mask is not None:
            x = index_put(x, padding_mask, 0)

        y = y.clone().detach()
        
        with TemporaryGrad():
            y_conv = self.pos_conv(y.transpose(1, 2))
            y_conv = y_conv.transpose(1, 2)
            y = y + y_conv

            # B x T x C -> T x B x C
            x, y = x.transpose(0, 1), y.transpose(0, 1)

            res = x
            layer_results = dict()
            for i, layer in enumerate(self.layers):
                if i == 0:
                    x, z = layer(x.clone().detach(), y, self_attn_padding_mask=padding_mask, need_weights=False)
                else:
                    x, z = layer(x, y, self_attn_padding_mask=padding_mask, need_weights=False)

            if self.modality == 'video':
                if self.temporal_flow:
                    if x.size(0) > self.flow_predictor.temp_len:
                        layer_results["temp_flow"] = self.flow_predictor(x.transpose(0, 1))
                if self.temporal_va_flow:
                    if x.size(0) > self.flow_va_predictor.temp_len:
                        layer_results["temp_va_flow"] = self.flow_va_predictor(x.transpose(0, 1), y.transpose(0, 1))
                if self.temporal_direction:
                    if x.size(0) > self.direction_predictor.temp_len-1 + self.direction_predictor.aggregator_kernel_size:
                        layer_results["temp_direction"] = self.direction_predictor(x.transpose(0, 1))
                if self.temporal_speed:
                    if x.size(0) > self.speed_predictor.speed * (self.speed_predictor.temp_len-1) + self.speed_predictor.aggregator_kernel_size:
                        layer_results["temp_speed"] = self.speed_predictor(x.transpose(0, 1))

            layer_results["feat"] = res.clone().transpose(0, 1)
            layer_results["xattn_feat"] = x.transpose(0, 1)
        
        x = res + x

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, layer_results

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict


@register_model("xmodal_av_hubert_seq2seq", dataclass=XmodalAVHubertSeq2SeqConfig)
class XmodalAVHubertSeq2Seq(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder, tgt_dict, cfg):
        super().__init__(encoder, decoder)
        self.cfg = cfg
        self.freeze_finetune_updates = cfg.freeze_finetune_updates

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""

        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
        }

        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(
                cfg.w2v_path, arg_overrides
            )
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            cfg.w2v_args = w2v_args
        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(
                    w2v_args
                )

        assert cfg.normalize == w2v_args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for "
            "both pre-training and here"
        )

        w2v_args.task.data = cfg.data

        task_pretrain = tasks.setup_task(w2v_args.task)
        if state is not None:
            task_pretrain.load_state_dict(state['task_state'])

        if cfg.xmodal_encoder_type == "xmodaltransformerencoder":
            xmodal_encoder_modality = XmodalTransformerEncoder
        else:
            raise NotImplementedError
        xmodal_encoder_video = xmodal_encoder_modality(modality='video', args=cfg)
        xmodal_encoder_audio = xmodal_encoder_modality(modality='audio', args=cfg)

        encoder_ = task_pretrain.build_model(w2v_args.model)

        encoder = HubertEncoderWrapper(cfg, encoder_, (xmodal_encoder_video, xmodal_encoder_audio))
        if state is not None and not cfg.no_pretrained_weights:
            # set strict=False because we omit some modules
            del state['model']['mask_emb']
            encoder.w2v_model.load_state_dict(state["model"], strict=False)

        encoder.w2v_model.remove_pretraining_modules()

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx=padding_idx)
            return emb

        decoder_embed_tokens = build_embedding(tgt_dict, cfg.decoder_embed_dim)
        decoder = TransformerDecoder(cfg, tgt_dict, decoder_embed_tokens)

        return XmodalAVHubertSeq2Seq(encoder, decoder, tgt_dict, cfg)

    def forward(self, **kwargs):
        output = self.encoder(**kwargs)
        decoder_out = self.decoder(prev_output_tokens=kwargs['prev_output_tokens'], encoder_out=output)
        return decoder_out, output["layer_results"]

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m
