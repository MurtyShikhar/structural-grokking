import os
import torch
import string
from tqdm import tqdm
import random
from datasets import Dataset as HFDataset
from vocabulary import WordVocabulary
from util import test_continuations

DATA_DIR = os.getcwd()

def process(line):
    return line.replace("\t", " ")


#### define a callback function for how to handle model predictions?


def read_lm_data(splits, do_process=True):
    in_sentences = []
    index_map = {split: [] for split in splits}
    for split in splits:
        with open(
            "{}/question_formation_data/question.{}".format(DATA_DIR, split),
            "r",
        ) as reader:
            if do_process:
                sents = [process(line.strip()) for line in reader.readlines()]
            else:
                sents = [line.strip() for line in reader.readlines()]
            for sent in sents:
                index_map[split].append(len(in_sentences))
                in_sentences.append(sent)
    return in_sentences, index_map


def build_datasets_lm():
    def get_subset(elem_list, idx_list):
        return [elem_list[idx] for idx in idx_list]

    splits = ["train", "val", "test"]
    in_sentences, index_map = read_lm_data(splits)
    print("num examples: {}".format(len(in_sentences)))

    in_vocab = WordVocabulary(in_sentences, split_punctuation=False)
    dataset = {}
    for split in splits:
        in_subset = get_subset(in_sentences, index_map[split])
        in_subset_tokenized = [in_vocab(s) for s in in_subset]
        in_lens = [len(s) for s in in_subset_tokenized]
        data = {
            "in": in_subset_tokenized,
            "in_len": in_lens,
            "idxs": index_map[split],
        }
        dataset_curr = HFDataset.from_dict(data)
        dataset[split] = dataset_curr
    return dataset, in_vocab, in_sentences


def eval_lm_callback(lm, in_vocab, split):
    def tokenizer(s):
        return [lm.encoder_sos] + in_vocab(s)

    sents, _ = read_lm_data([split])
    split_into_words = [sent.split(" ") for sent in sents if "quest" in sent]
    q_words = []
    prefixes = []
    for sent_words in split_into_words:
        idx = sent_words.index("quest")
        q_word = sent_words[idx + 1]
        q_words.append(q_word)
        prefixes.append(" ".join(sent_words[: idx + 1]))

    out = test_continuations(tokenizer, lm, prefixes, 0)
    # out = test_continuations_gpt2(tokenizer, lm, prefixes[:100], args.gpu_id)
    closing_words = ["doesn't", "does", "do", "don't"]
    closing_word_idxs = [in_vocab[w] for w in closing_words]
    out = out[:, closing_word_idxs]

    acc = [closing_words[i] == q_word for i, q_word in zip(out.argmax(dim=1), q_words)]
    agg_acc = sum(acc) / len(out)
    print(agg_acc)
    return agg_acc


if __name__ == "__main__":
    dataset, in_vocab, in_sentences = build_datasets_lm()
