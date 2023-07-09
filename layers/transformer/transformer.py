import torch
import torch.nn
import torch.nn.functional as F
from .multi_head_attention import MultiHeadAttention, AttentionMask
from typing import Optional, Callable, Dict
from dataclasses import dataclass

# This file is based on PyTorch's internal implementation

ActivationFunction = Callable[[torch.Tensor], torch.Tensor]


class TransformerEncoderLayer(torch.nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation: ActivationFunction = F.relu,
    ):
        super(TransformerEncoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model)

        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.activation = activation
        self.reset_parameters()

    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        pos_mask: Optional[torch.Tensor] = None,
        full_target=None,
        **kwargs,
    ) -> torch.Tensor:
        if "get_attn_scores" in kwargs:
            src2, weights = self.self_attn(
                src, src, AttentionMask(mask, pos_mask), need_weights=True
            )
        else:
            src2 = self.self_attn(
                src,
                src if full_target is None else full_target,
                AttentionMask(mask, pos_mask),
            )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        if "get_attn_scores" in kwargs:
            return src, weights
        else:
            return src

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(
            self.linear1.weight,
            gain=torch.nn.init.calculate_gain("relu")
            if self.activation is F.relu
            else 1.0,
        )
        torch.nn.init.xavier_uniform_(self.linear2.weight)


class TransformerDecoderLayerPreLN(torch.nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation: ActivationFunction = F.relu,
    ):
        super(TransformerDecoderLayerPreLN, self).__init__()

        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward, bias=False)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model, bias=False)

        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.norm3 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.dropout3 = torch.nn.Dropout(dropout)

        self.activation = activation
        self.reset_parameters()

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        full_target: Optional[torch.Tensor] = None,
        pos_offset: int = 0,
    ) -> torch.Tensor:

        assert pos_offset == 0 or tgt_mask is None
        tgt = self.norm1(tgt)
        tgt2 = self.self_attn(
            tgt,
            tgt if full_target is None else full_target,
            mask=AttentionMask(None, tgt_mask),
        )
        tgt = tgt + self.dropout1(tgt2)

        tgt = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            tgt, memory, mask=AttentionMask(memory_key_padding_mask, None)
        )
        tgt = tgt + self.dropout2(tgt2)

        tgt = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(
            self.linear1.weight,
            gain=torch.nn.init.calculate_gain("relu")
            if self.activation is F.relu
            else 1.0,
        )
        torch.nn.init.xavier_uniform_(self.linear2.weight)


class TransformerDecoderLayer(torch.nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation: ActivationFunction = F.relu,
    ):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model)

        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.norm3 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.dropout3 = torch.nn.Dropout(dropout)

        self.activation = activation
        self.reset_parameters()

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        full_target: Optional[torch.Tensor] = None,
        pos_offset: int = 0,
    ) -> torch.Tensor:

        assert pos_offset == 0 or tgt_mask is None
        tgt2 = self.self_attn(
            tgt,
            tgt if full_target is None else full_target,
            mask=AttentionMask(None, tgt_mask),
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            tgt, memory, mask=AttentionMask(memory_key_padding_mask, None)
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(
            self.linear1.weight,
            gain=torch.nn.init.calculate_gain("relu")
            if self.activation is F.relu
            else 1.0,
        )
        torch.nn.init.xavier_uniform_(self.linear2.weight)


class TransformerDecoderBase(torch.nn.Module):
    @dataclass
    class State:
        step: int
        state: Dict[int, torch.Tensor]

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def create_state(
        self, batch_size: int, max_length: int, device: torch.device
    ) -> State:
        return self.State(
            0,
            {
                i: torch.empty([batch_size, max_length, self.d_model], device=device)
                for i in range(len(self.layers))
            },
        )

    def one_step_forward(self, state: State, data: torch.Tensor, *args, **kwargs):
        assert (
            data.shape[1] == 1
        ), f"For one-step forward should have one timesteps, but shape is {data.shape}"
        assert state.step < state.state[0].shape[1]

        for i, l in enumerate(self.layers):
            state.state[i][:, state.step : state.step + 1] = data
            data = l(
                data,
                *args,
                **kwargs,
                full_target=state.state[i][:, : state.step + 1],
                pos_offset=state.step,
            )

        state.step += 1
        return data


class TransformerEncoder(torch.nn.Module):
    @dataclass
    class State:
        step: int
        state: Dict[int, torch.Tensor]

    def __init__(self, layer, n_layers: int, *args, **kwargs):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [layer(*args, **kwargs) for _ in range(n_layers)]
        )

        self.d_model = self.layers[0].d_model

    def create_state(
        self, batch_size: int, max_length: int, device: torch.device
    ) -> State:
        return self.State(
            0,
            {
                i: torch.zeros([batch_size, max_length, self.d_model], device=device)
                for i in range(len(self.layers))
            },
        )

    def attn_matrices(self, data: torch.Tensor, src_length_mask, pos_mask):
        attn_matrices = []
        for idx, l in enumerate(self.layers):
            _, mat = l(
                data, mask=src_length_mask, get_attn_scores=True, pos_mask=pos_mask
            )
            attn_matrices.append(mat)
        return attn_matrices

    def forward(self, data: torch.Tensor, *args, **kwargs):

        if "src_length_mask" in kwargs:
            mask = kwargs["src_length_mask"]
        elif len(args) > 0:
            mask = args[0]
        else:
            mask = None

        if "layer_id" in kwargs:
            layer_id = kwargs["layer_id"]
        else:
            layer_id = -1

        if "get_all_layers" in kwargs:
            all_data = [data]

        for idx, l in enumerate(self.layers):
            if type(mask) == list:
                mask_curr = mask[idx]
            else:
                mask_curr = mask
            data = l(data, mask=mask_curr, **kwargs)
            if layer_id == idx:
                break
            if "get_all_layers" in kwargs:
                all_data.append(data)
        if "get_all_layers" in kwargs:
            return all_data
        else:
            return data

    def one_step_forward(self, state: State, data: torch.Tensor, *args, **kwargs):
        assert (
            data.shape[1] == 1
        ), f"For one-step forward should have one timesteps, but shape is {data.shape}"
        assert state.step < state.state[0].shape[1]
        if "src_length_mask" in kwargs:
            mask = kwargs["src_length_mask"]
        else:
            mask = None

        for i, l in enumerate(self.layers):
            state.state[i][:, state.step : state.step + 1] = data
            data = l(
                data,
                mask=mask,
                full_target=state.state[i][:, : state.step + 1],
                pos_offset=state.step,
            )

        state.step += 1
        return data


class TransformerDecoder(TransformerDecoderBase):
    def __init__(self, layer, n_layers: int, d_model: int, *args, **kwargs):
        super().__init__(d_model)
        self.layers = torch.nn.ModuleList(
            [layer(d_model, *args, **kwargs) for _ in range(n_layers)]
        )
        # self.norm = torch.nn.LayerNorm(d_model)
        # self.final_dropout = torch.nn.Dropout(0.1)

    def forward(self, data: torch.Tensor, *args, **kwargs):
        for l in self.layers:
            data = l(data, *args, **kwargs)
        return data
        # return self.final_dropout(self.norm(data))


def TransformerEncoderWithLayer(layer=TransformerEncoderLayer):
    return lambda *args, **kwargs: TransformerEncoder(layer, *args, **kwargs)


def TransformerDecoderWithLayer(layer=TransformerDecoderLayer):
    return lambda *args, **kwargs: TransformerDecoder(layer, *args, **kwargs)


class Transformer(torch.nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: ActivationFunction = F.relu,
        encoder_layer=TransformerEncoderWithLayer(),
        decoder_layer=TransformerDecoderWithLayer(),
        is_null_encoder=False,
        **kwargs,
    ):
        super().__init__()

        if is_null_encoder:
            self.encoder = lambda src, src_length_mask: src
            self.num_encoder_layers = 0
        else:
            self.encoder = encoder_layer(
                num_encoder_layers, d_model, nhead, dim_feedforward, dropout, activation
            )
            self.num_encoder_layers = num_encoder_layers
        self.decoder = decoder_layer(
            num_decoder_layers, d_model, nhead, dim_feedforward, dropout, activation
        )

    def get_hidden_states(self, src, src_length_mask=None, layer_id=-1, is_lm=False):
        if is_lm:
            if type(src_length_mask) == list:
                pos_mask = self.generate_square_subsequent_mask(
                    src_length_mask[0].shape[1], device=src.device
                )
            else:
                pos_mask = self.generate_square_subsequent_mask(
                    src_length_mask.shape[1], device=src.device
                )
            memory = self.encoder(
                src,
                src_length_mask=src_length_mask,
                layer_id=layer_id,
                pos_mask=pos_mask,
            )
        else:
            memory = self.encoder(
                src,
                src_length_mask=src_length_mask,
                layer_id=layer_id,
            )

        return memory

    def get_attn_matrices(self, src, tgt, src_length_mask=None):
        if tgt is None:
            pos_mask = self.generate_square_subsequent_mask(
                src_length_mask.shape[1], device=src.device
            )
        else:
            pos_mask = None
        attn_mask = self.encoder.attn_matrices(
            src, src_length_mask=src_length_mask, pos_mask=pos_mask
        )
        return attn_mask

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_length_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if tgt is None:
            # run as a left to right language model
            pos_mask = self.generate_square_subsequent_mask(
                src_length_mask.shape[1], device=src.device
            )
            return self.encoder(
                src, src_length_mask=src_length_mask, pos_mask=pos_mask, **kwargs
            )
        else:
            memory = self.encoder(src, src_length_mask)
            return self.decoder(tgt, memory, tgt_mask, src_length_mask)

    def generate_square_subsequent_mask(
        self, sz: int, device: torch.device
    ) -> torch.Tensor:
        return torch.triu(
            torch.ones(sz, sz, dtype=torch.bool, device=device), diagonal=1
        )
