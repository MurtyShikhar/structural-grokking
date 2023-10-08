import torch.nn
from layers.transformer import Transformer
from layers.transformer.transformer import TransformerDecoderWithLayer
from models import TransformerEncDecModel, TransformerDecModel
from interfaces import (
    TransformerEncDecInterface,
    TransformerDecOnlyInterface,
    TransformerLMInterface,
)
from models.transformer_lm import TransformerLM


def create_lm(in_vocab_size, vec_dim, n_heads, encoder_n_layers) -> torch.nn.Module:
    args = dict(embedding_init="xavier", scale_mode="opennmt")
    return TransformerLM(
        in_vocab_size, vec_dim, n_heads, num_encoder_layers=encoder_n_layers, **args
    )


def create_model(
    in_vocab_size,
    out_vocab_size,
    vec_dim,
    n_heads,
    encoder_n_layers,
    decoder_n_layers,
    is_null_encoder=False,
    mode="enc_dec",
) -> torch.nn.Module:
    args = dict(embedding_init="xavier", scale_mode="opennmt", mode=mode)
    if is_null_encoder:
        return TransformerDecModel(
            in_vocab_size,
            out_vocab_size,
            vec_dim,
            n_heads,
            num_encoder_layers=encoder_n_layers,
            num_decoder_layers=decoder_n_layers,
            tied_embedding=True,
            **args
        )
    else:
        return TransformerEncDecModel(
            in_vocab_size,
            out_vocab_size,
            vec_dim,
            n_heads,
            num_encoder_layers=encoder_n_layers,
            num_decoder_layers=decoder_n_layers,
            tied_embedding=True,
            **args
        )


def create_model_interface(
    model, label_smoothing=0.0, is_null_encoder=False, is_lm=False
):
    if is_null_encoder:
        return TransformerDecOnlyInterface(model, label_smoothing=label_smoothing)
    elif is_lm:
        return TransformerLMInterface(model, label_smoothing=label_smoothing)
    else:
        return TransformerEncDecInterface(model, label_smoothing=label_smoothing)


#### Similar interfaces for pretrained models...
