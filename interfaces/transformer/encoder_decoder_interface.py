import torch
import torch.nn
from typing import Dict, Tuple
from models.encoder_decoder import add_eos
from models.transformer_enc_dec import TransformerResult
from ..model_interface import ModelInterface
import layers

from ..encoder_decoder import EncoderDecoderResult


class TransformerDecOnlyInterface(ModelInterface):
    def __init__(self, model: torch.nn.Module, label_smoothing: float = 0.0):
        self.model = model
        self.label_smoothing = label_smoothing

    def loss(
        self, outputs: TransformerResult, ref: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        l = layers.cross_entropy(
            outputs.data, ref, reduction="none", smoothing=self.label_smoothing
        )
        l = l.reshape_as(ref) * mask
        l = l.sum() / mask.sum()
        return l

    def decode_outputs(
        self, outputs: EncoderDecoderResult
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return outputs.outputs, outputs.out_lengths

    def __call__(
        self, data: Dict[str, torch.Tensor], train_eos: bool = True
    ) -> EncoderDecoderResult:
        out_len = data["out_len"].long()
        out_with_eos = add_eos(
            data["out"].transpose(0, 1), data["out_len"], self.model.decoder_sos_eos
        ).transpose(0, 1)
        out_len += 1

        res = self.model(
            data["in"],
            data["in_len"].long(),
            out_with_eos,
            out_len,
            teacher_forcing=self.model.training,
            max_len=out_len.max().item(),
        )

        res.data = res.data.transpose(0, 1)
        len_mask = ~self.model.generate_len_mask(
            out_with_eos.shape[1], out_len if train_eos else (out_len - 1)
        ).transpose(0, 1)

        loss = self.loss(res, out_with_eos.transpose(0, 1), len_mask)
        return EncoderDecoderResult(res.data, res.length, loss)


class TransformerEncDecInterface(ModelInterface):
    def __init__(self, model: torch.nn.Module, label_smoothing: float = 0.0):
        self.model = model
        self.label_smoothing = label_smoothing

    def loss(
        self, outputs: TransformerResult, ref: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        l = layers.cross_entropy(
            outputs.data, ref, reduction="none", smoothing=self.label_smoothing
        )
        l = l.reshape_as(ref) * mask
        l = l.sum() / mask.sum()
        return l

    def decode_outputs(
        self, outputs: EncoderDecoderResult
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return outputs.outputs, outputs.out_lengths

    def __call__(
        self,
        data: Dict[str, torch.Tensor],
        train_eos: bool = True,
        teacher_force_always=False,
    ) -> EncoderDecoderResult:

        if "out_len" in data:
            in_len = data["in_len"].long()
            in_with_eos = add_eos(data["in"], data["in_len"], self.model.encoder_eos)
            in_len += 1
            out_len = data["out_len"].long()
            out_with_eos = add_eos(
                data["out"], data["out_len"], self.model.decoder_sos_eos
            )
            out_len += 1
            res = self.model(
                in_with_eos.transpose(0, 1),
                in_len,
                out_with_eos.transpose(0, 1),
                out_len,
                teacher_forcing=self.model.training,
                max_len=out_len.max().item(),
            )
        else:
            in_len = data["in_len"].long()
            in_with_eos = add_eos(
                data["in"].transpose(0, 1), data["in_len"], self.model.encoder_eos
            ).transpose(0, 1)
            in_len += 1
            out_with_eos = None
            out_len = None
            res = self.model(
                in_with_eos,
                in_len,
                out_with_eos,
                out_len,
                teacher_forcing=self.model.training,
                max_len=None,
            )

        if self.model.mode == "mlm":
            ### take the in_len mask
            mask_logits = res[:, 1:-1, :]
            loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
            len_mask = ~self.model.generate_len_mask(
                data["labels"].shape[0], data["in_len"]
            )
            loss = layers.cross_entropy(
                mask_logits, data["labels"].transpose(0, 1), reduction="none"
            )
            loss = loss * len_mask
            return EncoderDecoderResult(
                mask_logits, None, (loss.sum() / len_mask.sum())
            )
        elif self.model.mode == "classifier":
            logits = res
            loss = layers.cross_entropy(logits, data["labels"], reduction="none")
            return EncoderDecoderResult(logits, None, loss.mean())
        else:
            res.data = res.data.transpose(0, 1)
            len_mask = ~self.model.generate_len_mask(
                out_with_eos.shape[0], out_len if train_eos else (out_len - 1)
            ).transpose(0, 1)

            loss = self.loss(res, out_with_eos, len_mask)
            return EncoderDecoderResult(res.data, res.length, loss)


class PreTrainedEncoderVanillaDecoderInterface(TransformerEncDecInterface):
    def __call__(
        self,
        data: Dict[str, torch.Tensor],
        train_eos: bool = True,
        teacher_force_always=False,
    ) -> EncoderDecoderResult:
        in_len = data["in_len"].long()
        out_len = data["out_len"].long()

        inp_dict = {
            "input_ids": data["in"].transpose(0, 1).contiguous(),
            "input_len": in_len,
            "target_len": out_len,
            "labels": data["out"].transpose(0, 1).contiguous(),
        }
        # to keep the same evaluation etc, we want the output to be (seq_len x batch_size)?
        res = self.model(**inp_dict)
        res.data = res.data.transpose(0, 1)
        len_mask = ~self.model.generate_len_mask(
            data["out"].shape[0], out_len if train_eos else (out_len - 1)
        ).transpose(0, 1)

        loss = self.loss(res, data["out"], len_mask)
        return EncoderDecoderResult(res.data, res.length, loss)


class T5EncDecInterface(TransformerEncDecInterface):
    def __call__(
        self,
        data: Dict[str, torch.Tensor],
        train_eos: bool = True,
        teacher_force_always=False,
    ) -> EncoderDecoderResult:
        in_len = data["in_len"].long()
        out_len = data["out_len"].long()

        if "role_marker" in data:
            inp_dict = {
                "input_ids": data["in"].transpose(0, 1).contiguous(),
                "role_marker_ids": data["role_marker"].transpose(0, 1).contiguous(),
                "labels": data["out"].transpose(0, 1).contiguous(),
            }
        else:
            inp_dict = {
                "input_ids": data["in"].transpose(0, 1).contiguous(),
                "labels": data["out"].transpose(0, 1).contiguous(),
            }
        # to keep the same evaluation etc, we want the output to be (seq_len x batch_size)?
        if self.model.training or teacher_force_always:
            res = self.model(**inp_dict)
            loss = res.loss
            logits = res.logits
            return EncoderDecoderResult(logits.transpose(0, 1), data["out_len"], loss)
        else:
            output = self.model.generate(
                data["in"].transpose(0, 1),
                max_new_tokens=out_len.max().item(),
                return_dict_in_generate=True,
                output_scores=True,
            )
            logits = torch.stack(output["scores"], axis=1)  # bs x dim

            out_lens = []
            for seq in output["sequences"]:
                out = torch.where(seq == 1)
                if len(out[0]) == 0:
                    out_lens.append(len(seq))
                else:
                    out_lens.append(out[0].item())

            loss = 0.0  # get the loss...
            return EncoderDecoderResult(
                logits.transpose(0, 1), torch.tensor(out_lens).to(logits.device), loss
            )
