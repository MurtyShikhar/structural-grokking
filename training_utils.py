import torch
from mlm_collator import MaskCollator
import sequence
from tqdm import tqdm
import os
import wandb

### NOTE: change this to your own wandb project and entity!
wandb.init(project="structural-grokking", entity="shikharmurty")
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

from transformers.data.data_collator import DataCollatorWithPadding
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
)

import collate
import wandb
from plot import CustomPlot


def get_grad_norm(model):
    total_norm = 0
    parameters = [
        p for p in model.parameters() if p.grad is not None and p.requires_grad
    ]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


def get_opt(lr, model):
    if type(model) != torch.nn.Module:
        model = model.model
    no_decay = ["bias", "LayerNorm.weight"]
    weight_decay = 0.0
    adam_epsilon = 1e-7
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=lr,
        eps=adam_epsilon,
    )
    return optimizer


def get_scheduler(opt, t_total):
    num_warmup_steps = 10000
    scheduler = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
    )
    return scheduler


def eval_lm(model_interface, val_datasets, best_accs, device, num_steps, collator):
    def helper(validation):
        model_interface.model.eval()
        loss_curr = 0
        total = 0
        with torch.no_grad():
            for batch in tqdm(validation):
                batch_gpu = {}
                for key in batch:
                    batch_gpu[key] = batch[key].to(device)
                res = model_interface(batch_gpu, normalize=True)
                loss_curr += res.loss.cpu().numpy()
                total += 1
        return loss_curr / total

    eval_batch_size = 32
    plots = {}
    curr_accs = {}
    for key, val_dataset in val_datasets.items():
        validation = DataLoader(
            val_dataset,
            sampler=SequentialSampler(val_dataset),
            batch_size=eval_batch_size,
            collate_fn=collator,
        )
        curr_accs[key] = helper(validation)
        plots["curr-{}-ppl".format(key)] = curr_accs[key]
    best_accs = {key: min(curr_accs[key], best_accs[key]) for key in curr_accs}
    plots.update({"best/{}": v for k, v in best_accs.items()})
    plotting_util(plots, num_steps)
    return best_accs, curr_accs


def plotting_util(dict_of_elems, step):
    wandbdict = {}
    for k, v in dict_of_elems.items():
        if isinstance(v, CustomPlot):
            v = v.to_wandb()
            if v is None:
                continue

            if isinstance(v, dict):
                for k2, v2 in v.items():
                    wandbdict[k + "/" + k2] = v2
            else:
                wandbdict[k] = v
        elif isinstance(v, (int, float)):
            wandbdict[k] = v
        else:
            assert False, f"Invalid data type {type(v)}"
    wandbdict["iteration"] = step
    wandb.log(wandbdict)


def eval_func(model, validation, tokenizer, best_acc, device):
    def get_decoding_acc(outputs, labels):
        acc = 0
        for out, label in zip(outputs, labels):
            dec_str = tokenizer.decode(out, skip_special_tokens=True)
            label = [(l if l != -100 else tokenizer.pad_token_id) for l in label]
            orig_str = tokenizer.decode(label, skip_special_tokens=True)
            acc += dec_str == orig_str
        return acc

    curr_acc = 0
    total = 0
    if type(model) != torch.nn.Module:
        model.model.eval()
    else:
        model.eval()
    with torch.no_grad():
        for batch in tqdm(validation):
            batch_gpu = {}
            for key in batch:
                batch_gpu[key] = batch[key].to(device)
            curr_acc += get_decoding_acc(
                model.generate(batch_gpu["input_ids"]).cpu().tolist(),
                batch["labels"].cpu().tolist(),
            )
            total += len(batch["labels"])

    curr_acc /= 1.0 * total
    print("Current Accuracy: {:.4f}".format(curr_acc))
    if curr_acc > best_acc:
        return curr_acc
    else:
        return best_acc


def eval_callback(
    args,
    model,
    val_datasets,
    tokenizer,
    best_accs,
    device,
    num_steps,
    train_data_collator,
):
    assert model.model.mode == "lm"
    best_accs, curr_accs = eval_lm(
        model,
        val_datasets,
        best_accs,
        device,
        num_steps,
        train_data_collator,
    )
    return best_accs, curr_accs


def train_loop(
    args,
    model,
    train_dataset,
    val_datasets,
    device,
    save_dir,
    tokenizer=None,
    metric="acc",
    in_vocab=None,
    callback_fn=None,
):
    num_steps = 0
    max_grad_norm = 1
    train_batch_size = 8
    accum_steps = 1
    eval_every = 10000
    max_steps = 2000000

    opt = get_opt(args.lr, model)
    scheduler = get_scheduler(opt, max_steps)

    if tokenizer is not None:
        train_data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    else:
        train_data_collator = collate.VarLengthCollate(tokenizer)

    best_ppl = {key: 10000.0 for key in val_datasets}

    while True:
        train_dataloader = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=train_batch_size,
            collate_fn=train_data_collator,
        )
        total_train_sz = len(train_dataset)
        if num_steps > max_steps:
            break
        with torch.enable_grad(), tqdm(total=total_train_sz) as progress_bar:
            losses = []
            for curr_batch_dict in train_dataloader:
                if type(model) != torch.nn.Module:
                    model.model.train()
                else:
                    model.train()
                curr_batch_dict_gpu = {}
                for key in curr_batch_dict:
                    curr_batch_dict_gpu[key] = curr_batch_dict[key].to(device)
                loss_curr = model(curr_batch_dict_gpu).loss
                progress_bar.update(curr_batch_dict["in"].shape[1])
                losses.append(loss_curr.item())

                loss_curr /= accum_steps
                loss_curr.backward()
                if len(losses) == accum_steps:
                    num_steps += 1
                    torch.nn.utils.clip_grad_norm_(
                        model.model.parameters(), max_grad_norm
                    )
                    progress_bar.set_postfix(
                        {"loss": sum(losses), "num_steps": num_steps}
                    )
                    grad_norm = get_grad_norm(model.model)
                    wandb.log(
                        {
                            "loss": sum(losses),
                            "grad_norm": grad_norm,
                            "iteration": num_steps,
                        }
                    )
                    opt.step()
                    scheduler.step()
                    model.model.zero_grad()
                    losses = []
                    if num_steps % eval_every == 0:
                        print("Evaluating at step {}".format(num_steps))
                        best_ppl, curr_ppl = eval_callback(
                            args,
                            model,
                            val_datasets,
                            tokenizer,
                            best_ppl,
                            device,
                            num_steps,
                            train_data_collator,
                        )
                        print(curr_ppl)
                        if callback_fn is not None:
                            val_score = callback_fn("val")
                            test_score = callback_fn("test")
                            print(val_score, test_score)
                            wandbdict = {
                                "iteration": num_steps,
                                "val_aux": val_score,
                                "test_aux": test_score,
                            }
                            wandb.log(wandbdict)

                        if len(save_dir) > 0:
                            torch.save(
                                model.model.state_dict(),
                                os.path.join(
                                    save_dir, "checkpoint_{}.pickle".format(num_steps)
                                ),
                            )
                    if num_steps > max_steps:
                        break
            if losses:
                num_steps += 1
                progress_bar.set_postfix({"loss": sum(losses), "num_steps": num_steps})
                grad_norm = get_grad_norm(model.model)
                wandb.log(
                    {
                        "loss": sum(losses),
                        "grad_norm": grad_norm,
                        "iteration": num_steps,
                    }
                )
                torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_grad_norm)
                opt.step()
                scheduler.step()
                model.model.zero_grad()
                losses = []
                if num_steps > max_steps:
                    break

    print("Best Perplexities,", best_ppl)
    return
