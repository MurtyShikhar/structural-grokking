
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers import AutoConfig, T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Stack
from layers.tied_embedding import TiedEmbedding
from layers.transformer import Transformer
from layers import PositionalEncoding

import torch.nn.functional as F
import torch
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import copy
import math
import warnings

from models import TransformerResult

from layers.transformer.transformer import TransformerDecoderLayer, TransformerDecoderWithLayer
__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""

from transformers.file_utils import is_torch_fx_proxy

class PreTrainedEncoderWithVanillaDecoder(nn.Module):
    def __init__(self, encoder, pad_token_id, start_token_id):
        super().__init__()
        self.encoder = encoder
        self.pad_token_id = pad_token_id
        self.mode = 'enc_dec'
        self.start_token_id = start_token_id
        
        input_embeddings = encoder.get_input_embeddings()
        
        vocab_size = input_embeddings.num_embeddings
        d_model = encoder.config.hidden_size
         
        self.output_embedding = nn.Embedding(vocab_size, d_model)
        self.output_map = TiedEmbedding(self.output_embedding.weight)


        # define hyperparams
        num_decoder_layers = 6
        nhead = 8
        dim_feedforward = 2048
        dropout = 0.1 
        activation = F.relu
        self.decoder = TransformerDecoderWithLayer()(num_decoder_layers, d_model, 
                                                     nhead, dim_feedforward, dropout, activation)
        self.pos = PositionalEncoding(d_model, max_len=5000, batch_first=True, scale=1.0)
        self.register_buffer('int_seq', torch.arange(1024, dtype=torch.long))
        self.scale_mode = 'opennmt'


    def output_embed(self, x):
        return self.output_embedding(x)

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.start_token_id
        pad_token_id = self.pad_token_id
        
        assert (
            decoder_start_token_id is not None
        ), "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. See T5 docs for more information"

        # shift inputs to the right
        if is_torch_fx_proxy(input_ids):
            # Item assignment is not supported natively for proxies.
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        assert torch.all(shifted_input_ids >= 0).item(), "Verify that `shifted_input_ids` has only positive values"
        return shifted_input_ids

    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(sz, sz, dtype=torch.bool, device=device), diagonal=1)

    def pos_embed(self, t: torch.Tensor, offset: int, scale_offset: int) -> torch.Tensor:
        if self.scale_mode == "opennmt":
            t = t * math.sqrt(t.shape[-1])
        
        return self.pos(t, offset)

    def generate_len_mask(self, max_len: int, len: torch.Tensor) -> torch.Tensor:
        return self.int_seq[: max_len] >= len.unsqueeze(-1)


    def forward(self, input_ids=None, input_len=None, labels=None, target_len=None):
        with torch.no_grad():
            encoder_outputs = self.encoder(input_ids=input_ids, return_dict=True)
            hidden_states = encoder_outputs.last_hidden_state
        
        #encoder_outputs = self.encoder(input_ids=input_ids, return_dict=True)
        #hidden_states = encoder_outputs.last_hidden_state

        in_len_mask = self.generate_len_mask(input_ids.shape[1], input_len)
        
        decoder_input_ids = self._shift_right(labels)
        target = self.output_embed(decoder_input_ids)
        target = self.pos_embed(target, 0, 1)

        tgt_mask = self.generate_square_subsequent_mask(target.shape[1], input_ids.device)
        res = self.decoder(target, hidden_states, tgt_mask, in_len_mask)
            
        return TransformerResult.create(self.output_map(res), target_len)


class PretrainedEncoderWithRandomDecoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder 
        input_embeddings = encoder.get_input_embeddings()
        vocab_size = input_embeddings.num_embeddings
        self.d_model = input_embeddings.embedding_dim


        # TODO: might need extra glue...
        # use a decoder that's just a standard    
        decoder_config = AutoConfig.from_pretrained('t5-small')
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = 6
        self.decoder_config = decoder_config
        self.decoder = T5Stack(decoder_config, input_embeddings)
        self.lm_head = nn.Linear(self.d_model, vocab_size, bias=False)

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.decoder_config.decoder_start_token_id
        pad_token_id = self.decoder_config.pad_token_id

        assert (
            decoder_start_token_id is not None
        ), "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. See T5 docs for more information"

        # shift inputs to the right
        if is_torch_fx_proxy(input_ids):
            # Item assignment is not supported natively for proxies.
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        assert torch.all(shifted_input_ids >= 0).item(), "Verify that `shifted_input_ids` has only positive values"
        return shifted_input_ids

    def forward(
        self,
        input_ids=None,
        labels=None,
    ):
        
        encoder_outputs = self.encoder(input_ids=input_ids, return_dict=True) 
        hidden_states = encoder_outputs.last_hidden_state

        
        
        decoder_input_ids = self._shift_right(labels)



        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=hidden_states,
            use_cache=True,
            return_dict=True
        )

        sequence_output = decoder_outputs[0]
        # Rescale output before projecting on vocab
        sequence_output = sequence_output * (self.d_model ** -0.5)
        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666


        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
    


class T5ForConditionalGenerationRoleMarkers(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.role_embeddings = nn.Embedding(config.num_roles, config.d_model)
        self.project_down = nn.Linear(2*config.d_model, config.d_model, bias=False)

    def forward(
        self,
        input_ids=None,
        role_marker_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]
        # we use the role_markers now!
        if role_marker_ids is not None:
            role_marker_vecs = self.role_embeddings(role_marker_ids)
            try:
                hidden_states = self.project_down(torch.cat([hidden_states, role_marker_vecs], axis=-1))
            except:
                import pdb; pdb.set_trace();

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
