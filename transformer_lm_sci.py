# compute sci / weight norms / attention sparsity for an LM

import argparse
import os
from data_utils import (
    build_datasets_dyck,
    build_datasets_lm,
    build_datasets_tense_inflection,
)
from train_transformers import get_base_transformer_lm
import torch
import numpy as np
from data_utils.dyck_helpers import read_dyck_data
from data_utils.lm_dataset_helpers import read_lm_data
from data_utils.tense_inflection_helpers import read_ti_data

from tree_projections import (
    get_tree_projection,
    get_parsing_accuracy,
    get_sparsity_scores_helper,
)
from tqdm import tqdm
from util import set_seed
import random

from graph_node import Graph
from parse_q_and_tense import parse_question, parse_tense, convert_to_parse


def get_gold_parse(dataset_type, sent):
    if dataset_type == "lm":
        parse = convert_to_parse(sent, parse_question(sent))
    elif dataset_type == "tense":
        parse = convert_to_parse(sent, parse_tense(sent))
    else:
        raise Exception
    return parse


def process(sents, split_by_words):
    def remove_fullstop(sent_list):
        if sent_list[-1] == ".":
            return sent_list[:-1]

    new_sents = []
    target_words = []
    for sent in sents:
        split_word = None
        sent_words = sent.split(" ")
        for word in split_by_words:
            if word in sent_words:
                split_word = word
                break
        if split_word is None:
            continue
        idx = sent_words.index(split_word)
        target_words.append(sent_words[idx + 1])
        new_sents.append(" ".join(remove_fullstop(sent_words[:idx])))
    return new_sents, target_words


def get_norm(model):
    total_norm = 0
    parameters = [p for p in model.parameters()]
    for p in parameters:
        param_norm = p.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


def compute_attention_sparsity(args, model_name):
    args.vec_dim = 512
    args.n_heads = 4
    args.gpu_id = 0

    mname = model_name.split("/")[-2]
    folder_name = "MODEL_SPARSITY/{}_sparsity".format(mname)
    checkpoint = model_name.split("/")[-1].split(".")[0]
    if os.path.exists("{}/{}.txt".format(folder_name, checkpoint)):
        return
    if args.dataset == "dyck":
        _, in_vocab, _ = build_datasets_dyck()
        in_sentences = read_dyck_data([args.split], 20)
        if len(in_sentences) > 500:
            idxs = random.sample(
                [
                    idx
                    for idx, sent in enumerate(in_sentences)
                    if len(sent.split(" ")) < 200
                ],
                k=500,
            )
            in_sentences = [in_sentences[idx] for idx in idxs]
    elif args.dataset == "lm":
        _, in_vocab, _ = build_datasets_lm()
        in_sentences, _ = read_lm_data([args.split])
        in_sentences, targets = process(in_sentences, split_by_words=["quest"])
        if len(in_sentences) > 10000:
            idxs = random.sample([idx for idx, _ in enumerate(in_sentences)], k=10000)
            in_sentences = [in_sentences[idx] for idx in idxs]
    elif args.dataset == "tense":
        _, in_vocab, _ = build_datasets_tense_inflection()
        in_sentences, _ = read_ti_data([args.split])
        in_sentences, targets = process(
            in_sentences, split_by_words=["PRESENT", "PAST"]
        )
        if len(in_sentences) > 10000:
            idxs = random.sample([idx for idx, _ in enumerate(in_sentences)], k=10000)
            in_sentences = [in_sentences[idx] for idx in idxs]
    else:
        raise Exception

    lm, _ = get_base_transformer_lm(args, in_vocab, model_name=model_name)
    device = torch.device("cuda:{}".format(args.gpu_id))
    lm.to(device)

    def tokenizer(s, add_special_tokens=True):
        if add_special_tokens:
            return [lm.encoder_sos] + in_vocab(s)
        else:
            return in_vocab(s)

    attn_sparsity = get_sparsity_scores_helper(lm, tokenizer, in_sentences)

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    with open("{}/{}.txt".format(folder_name, checkpoint), "w") as writer:
        writer.write(str(attn_sparsity))
        writer.write("\n")


def compute_model_norm(args, model_name):
    args.vec_dim = 512
    args.n_heads = 4
    args.gpu_id = 0

    if args.dataset == "dyck":
        _, in_vocab, _ = build_datasets_dyck()
    elif args.dataset == "lm":
        _, in_vocab, _ = build_datasets_lm()
    elif args.dataset == "tense":
        _, in_vocab, _ = build_datasets_tense_inflection()
    else:
        raise Exception

    lm, _ = get_base_transformer_lm(args, in_vocab, model_name=model_name)
    device = torch.device("cuda:{}".format(args.gpu_id))
    lm.to(device)

    model_norm = get_norm(lm)
    mname = model_name.split("/")[-2]
    folder_name = "MODEL_NORM/{}_norm".format(mname)
    checkpoint = model_name.split("/")[-1].split(".")[0]
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    with open("{}/{}.txt".format(folder_name, checkpoint), "w") as writer:
        writer.write(str(model_norm))
        writer.write("\n")


def compute_sci_helper_fn(
    args, in_vocab, model_name, in_sentences, targets, gold_parses, ret_vals
):
    lm, _ = get_base_transformer_lm(args, in_vocab, model_name=model_name)
    device = torch.device("cuda:{}".format(args.gpu_id))
    lm.to(device)

    mname = model_name.split("/")[-2]
    folder_name = "SCI_SCORES/{}_sci".format(mname)
    checkpoint = model_name.split("/")[-1].split(".")[0]
    print(checkpoint)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    if os.path.exists("{}/{}.txt".format(folder_name, checkpoint)) and not ret_vals:
        return

    def tokenizer(s, add_special_tokens=True):
        if add_special_tokens:
            return [lm.encoder_sos] + in_vocab(s)
        else:
            return in_vocab(s)

    total_sci_score = 0.0
    pred_parses = []

    batch_size = 1
    st = 0
    with tqdm(total=len(in_sentences)) as progress_bar:
        while st < len(in_sentences):
            en = min(len(in_sentences), st + batch_size)
            output = get_tree_projection(
                in_sentences[st:en][0],
                lm,
                tokenizer,
                st_threshold=0,
                verbose=True,
                sim_fn="cosine",
                normalize=True,
                layer_id=-1,
                is_lm=True,
            )

            pred_parses += [output["pred_parse"]]
            total_sci_score += np.sum([output["pred_parse_score"]])
            progress_bar.update(en - st)
            st = en

    score = total_sci_score / len(in_sentences)
    if gold_parses is not None:
        parsing_acc = get_parsing_accuracy(pred_parses, gold_parses)["f1"]
    else:
        parsing_acc = 0.0

    if ret_vals:
        return score, pred_parses, gold_parses
    else:
        with open("{}/{}.txt".format(folder_name, checkpoint), "w") as writer:
            writer.write(str(score))
            writer.write("\n")
            writer.write(str(parsing_acc))
            writer.write("\n")
            writer.write("\n")


def compute_sci(args, model_name, ret_vals=False):
    args.vec_dim = 512
    args.n_heads = 4
    args.gpu_id = 0

    if args.dataset == "dyck":
        _, in_vocab, _ = build_datasets_dyck()
        in_sentences = read_dyck_data([args.split], 20)
        if len(in_sentences) > 500:
            idxs = random.sample(
                [idx for idx, sent in enumerate(in_sentences)],
                k=500,
            )
            in_sentences = [in_sentences[idx] for idx in idxs]
        targets = None
        gold_parses = None
    elif args.dataset == "lm":
        _, in_vocab, _ = build_datasets_lm()
        in_sentences, _ = read_lm_data([args.split])
        in_sentences, targets = process(in_sentences, split_by_words=["quest"])
        if len(in_sentences) > 10000:
            idxs = random.sample([idx for idx, _ in enumerate(in_sentences)], k=10000)
            in_sentences = [in_sentences[idx] for idx in idxs]
            targets = [targets[idx] for idx in idxs]
        gold_parses = [
            get_gold_parse("lm", "{} . quest".format(sent)) for sent in in_sentences
        ]
    elif args.dataset == "tense":
        _, in_vocab, _ = build_datasets_tense_inflection()
        in_sentences, _ = read_ti_data([args.split])
        in_sentences, targets = process(
            in_sentences, split_by_words=["PRESENT", "PAST"]
        )
        if len(in_sentences) > 10000:
            idxs = random.sample([idx for idx, _ in enumerate(in_sentences)], k=10000)
            in_sentences = [in_sentences[idx] for idx in idxs]
            targets = [targets[idx] for idx in idxs]
        gold_parses = [
            get_gold_parse("tense", "{} . present".format(sent))
            for sent in in_sentences
        ]

    else:
        raise Exception

    return compute_sci_helper_fn(
        args, in_vocab, model_name, in_sentences, targets, gold_parses, ret_vals
    )


def get_idxs_res(dataset, res):
    if dataset == "dyck":
        return [10000] + [10000 * idx for idx in range(res, 51, res)]
    else:
        return [3000] + [3000 * idx for idx in range(res, 101, res)]


def flatten(l_o_l):
    return [x for l in l_o_l for x in l]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--dataset", type=str, default="dyck")
    parser.add_argument("--encoder_n_layers", default=4, type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument("--compute_norm", action="store_true")
    parser.add_argument("--compute_sparsity", action="store_true")
    args = parser.parse_args()
    # main(args)
    # eval2(args)
    set_seed(args.seed)

    if args.compute_norm:
        idxs = get_idxs_res(args.dataset, args.resolution)
        all_model_names = [
            "{}/checkpoint_{}.pickle".format(args.model_name, idx) for idx in idxs
        ]

        for model_name in all_model_names:
            compute_model_norm(args, model_name)

    elif args.compute_sparsity:
        idxs = get_idxs_res(args.dataset, args.resolution)
        all_model_names = [
            "{}/checkpoint_{}.pickle".format(args.model_name, idx) for idx in idxs
        ]

        for model_name in all_model_names:
            compute_attention_sparsity(args, model_name)
    elif not args.dummy:
        if args.resolution == -1:
            compute_sci(args, args.model_name)
        else:
            ### args.model_name is now the path
            res_list = [10, 5]
            if args.resolution != 1:
                idxs = get_idxs_res(args.dataset, args.resolution)
            else:
                idxs = range(3000, 303000, 3000)
            other_res = set(
                flatten(
                    [
                        get_idxs_res(args.dataset, res)
                        for res in res_list
                        if res > args.resolution
                    ]
                )
            )
            idxs = [idx for idx in idxs if idx not in other_res]
            all_model_names = [
                "{}/checkpoint_{}.pickle".format(args.model_name, idx) for idx in idxs
            ]

            for model_name in all_model_names:
                compute_sci(args, model_name)
