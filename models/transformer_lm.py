import torch
import random
import math
from layers import Transformer, TiedEmbedding, PositionalEncoding
from typing import Callable, Optional


class DotDict(dict):
    def __getattr__(self, item):
        if item not in self:
            raise AttributeError
        return self.get(item)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class TransformerResult(DotDict):
    data: torch.Tensor
    length: torch.Tensor

    @staticmethod
    def create(data: torch.Tensor, length: torch.Tensor):
        return TransformerResult({"data": data, "length": length})


class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        n_input_tokens: int,
        state_size: int = 512,
        ff_multiplier: float = 1,
        max_len: int = 5000,
        transformer=Transformer,
        tied_embedding: bool = False,
        pos_embeddig: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
        encoder_sos: bool = True,
        embedding_init: str = "pytorch",
        scale_mode: str = "none",
        **kwargs
    ):
        """
        Transformer encoder-decoder.

        :param n_input_tokens: Number of channels for the input vectors
        :param n_out_tokens: Number of channels for the output vectors
        :param state_size: The size of the internal state of the transformer
        """
        super().__init__()

        assert scale_mode in ["none", "opennmt", "down"]
        assert embedding_init in ["pytorch", "xavier", "kaiming"]

        self.tied_embedding = tied_embedding

        self.encoder_eos = n_input_tokens
        self.encoder_sos = n_input_tokens + 1 if encoder_sos else None
        self.state_size = state_size
        self.embedding_init = embedding_init
        self.ff_multiplier = ff_multiplier
        self.n_input_tokens = n_input_tokens
        self.scale_mode = scale_mode
        self.pos = pos_embeddig or PositionalEncoding(
            state_size,
            max_len=max_len,
            batch_first=True,
            scale=(1.0 / math.sqrt(state_size)) if scale_mode == "down" else 1.0,
        )

        self.register_buffer("int_seq", torch.arange(max_len, dtype=torch.long))
        self.construct(transformer, **kwargs)
        self.reset_parameters()
        # need this flag for the training loop helpers
        self.mode = "lm"

    def construct(self, transformer, **kwargs):
        self.input_embedding = torch.nn.Embedding(
            self.n_input_tokens + 1 + int(self.encoder_sos is not None),
            self.state_size,
        )

        if self.tied_embedding:
            self.output_map = TiedEmbedding(self.input_embedding.weight)
        else:
            self.output_map = torch.nn.Linear(
                self.state_size,
                self.n_input_tokens + 1 + int(self.encoder_sos is not None),
            )

        self.trafo = transformer(
            d_model=self.state_size,
            dim_feedforward=int(self.ff_multiplier * self.state_size),
            **kwargs
        )

    def input_embed(self, x: torch.Tensor) -> torch.Tensor:
        src = self.input_embedding(x.long())
        return src

    def reset_parameters(self):
        if self.embedding_init == "xavier":
            torch.nn.init.xavier_uniform_(self.input_embedding.weight)
        elif self.embedding_init == "kaiming":
            torch.nn.init.kaiming_normal_(self.input_embedding.weight)
        if not self.tied_embedding:
            torch.nn.init.xavier_uniform_(self.output_map.weight)

    def generate_len_mask(self, max_len: int, len: torch.Tensor) -> torch.Tensor:
        return self.int_seq[:max_len] >= len.unsqueeze(-1)

    def run_teacher_forcing(
        self,
        src: torch.Tensor,
        src_len: torch.Tensor,
    ) -> TransformerResult:
        in_len_mask = self.generate_len_mask(src.shape[1], src_len)
        res = self.trafo(src, tgt=None, src_length_mask=in_len_mask)
        return TransformerResult.create(self.output_map(res), src_len)

    def pos_embed(self, t: torch.Tensor, offset: int) -> torch.Tensor:
        if self.scale_mode == "opennmt":
            t = t * math.sqrt(t.shape[-1])

        return self.pos(t, offset)

    def get_encoder_layers(self):
        return self.trafo.num_encoder_layers

    def get_attention_sparsity(self, src, src_len):
        mask = self.generate_len_mask(src.shape[1], src_len)
        src = self.pos_embed(self.input_embed(src), 0)
        attn_matrices = self.trafo.get_attn_matrices(
            src, tgt=None, src_length_mask=mask
        )

        total_entropy = 0.0
        for mat in attn_matrices:
            for clen, batch_obj in zip(src_len, mat):
                curr_att_mat = batch_obj[:clen, :clen]
                for idx, attns in enumerate(curr_att_mat):
                    total_entropy += torch.distributions.Categorical(
                        attns[: idx + 1]
                    ).entropy()
        return total_entropy / len(attn_matrices)

    def encoder_only(self, src, mask, layer_id=-1, gaussian_noise=None):
        src = self.pos_embed(self.input_embed(src), 0)
        if gaussian_noise is not None:
            src += gaussian_noise

        return self.trafo.get_hidden_states(src, mask, layer_id=layer_id, is_lm=True)

    def run_greedy(
        self, src: torch.Tensor, src_len: torch.Tensor, max_len: int
    ) -> TransformerResult:
        batch_size = src.shape[0]
        n_steps = src.shape[1]

        src = self.pos_embed(self.input_embed(src), 0)
        in_len_mask = self.generate_len_mask(src.shape[1], src_len)
        processed = self.trafo(
            src, tgt=None, src_length_mask=in_len_mask, get_all_layers=True
        )

        running = torch.ones([batch_size], dtype=torch.bool, device=src.device)
        out_len = torch.zeros_like(running, dtype=torch.long)

        last_embeddings = self.output_map(
            torch.cat(
                [curr[l - 1].unsqueeze(0) for curr, l in zip(processed[-1], src_len)]
            )
        )

        pred_words = torch.argmax(last_embeddings, -1)
        next_tgt = torch.cat(
            [
                self.pos_embed(
                    self.input_embed(pred_words[idx : idx + 1]).unsqueeze(1), slen
                )
                for idx, slen in enumerate(src_len)
            ]
        )

        all_outputs = [last_embeddings.unsqueeze(1)]
        state = self.trafo.encoder.create_state(
            src.shape[0], n_steps + max_len, src.device
        )

        for idx in range(len(processed) - 1):
            state.state[idx][:, :n_steps] = processed[idx]

        state.step = n_steps
        # pos masking not implemented
        curr_mask = in_len_mask
        for i in range(max_len):
            curr_mask = torch.cat([curr_mask, ~running.unsqueeze(1)], dim=1)
            output = self.trafo.encoder.one_step_forward(
                state, next_tgt, src_length_mask=curr_mask
            )

            output = self.output_map(output)
            all_outputs.append(output)
            out_token = torch.argmax(output[:, -1], -1)
            running &= out_token != self.encoder_eos
            out_len[running] = i + 1
            next_tgt = torch.cat(
                [
                    self.pos_embed(
                        self.input_embed(out_token[idx : idx + 1]).unsqueeze(1),
                        i + 1 + slen,
                    )
                    for idx, slen in enumerate(src_len)
                ]
            )

        return TransformerResult.create(torch.cat(all_outputs, 1), out_len)

    def forward(
        self,
        src: torch.Tensor,
        src_len: torch.Tensor,
    ) -> TransformerResult:
        """
        Run transformer encoder-decoder on some input/output pair

        :param src: source features. Shape: [N, S, D], where S in the in sequence length, N is the batch size
        :param src_len: length of source sequences. Shape: [N], N is the batch size
        :param target: target tensor. Shape: [N, S], where T in the in sequence length, N is the batch size
        :param target_len: length of target sequences. Shape: [N], N is the batch size
        :param teacher_forcing: use teacher forcing or greedy decoding
        :param max_len: overwrite autodetected max length. Useful for parallel execution
        :return: prediction of the target tensor. Shape [N, T, C_out]
        """
        src = self.pos_embed(self.input_embed(src), 0)
        ### we are only going to ever use this LM to measure perplexity / surprisal, so it's ok
        return self.run_teacher_forcing(src, src_len)
