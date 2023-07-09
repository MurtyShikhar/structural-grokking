import os
import torch
import string
from tqdm import tqdm
import random
from datasets import Dataset as HFDataset
from vocabulary import WordVocabulary
from util import test_continuations


## CHANGE THIS FOR YOUR PROJECT (cannot use os.getcwd() because of hydra)
DATA_DIR = "/u/scr/smurty/structural_grokking/data_utils"


def read_dyck_data(splits, vocab_size):
    def process(sent):
        words = sent.split(" ")[:-1]
        return " ".join(words)

    in_sentences = []
    for split in splits:
        with open(
            "{}/dyck_data/k-{}_d-10_{}.txt".format(DATA_DIR, vocab_size, split), "r"
        ) as reader:
            sents = [process(line.strip()) for line in reader.readlines()]
            for sent in sents:
                in_sentences.append(sent)
    return in_sentences


def get_opening_brackets_prefix(dyck_str):
    """'
    for every closing bracket, return the index of the opening bracket
    """
    dyck_words = dyck_str.split(" ")
    stack = []
    targets = []
    for idx, word in enumerate(dyck_words):
        if "(" in word:
            stack.append(("(", idx))
        else:
            _, oidx = stack.pop()
            targets.append(oidx)
    return targets


def get_all_prefixes(
    dyck_str, target=False, min_dep_length=None, get_opening_idx=False
):
    #### for every prefix that ends in a closing bracket
    dyck_words = dyck_str.split(" ")

    # idxs for all closing brackets
    closing_idxs = [idx for idx, word in enumerate(dyck_words) if ")" in word]
    prefixes = [" ".join(dyck_words[:idx]) for idx in closing_idxs]
    if min_dep_length:
        o_bracket_idxs = get_opening_brackets_prefix(dyck_str)
        chosen_prefixes = [
            idx
            for idx, _ in enumerate(o_bracket_idxs)
            if closing_idxs[idx] - o_bracket_idxs[idx] >= min_dep_length
        ]

        if get_opening_idx:
            o_targets = [o_bracket_idxs[idx] for idx in chosen_prefixes]
            chosen_prefixes = [prefixes[idx] for idx in chosen_prefixes]
            return chosen_prefixes, o_targets
        else:
            return [prefixes[idx] for idx in chosen_prefixes], [
                dyck_words[closing_idxs[idx]] for idx in chosen_prefixes
            ]

    if target:
        # for every closing bracket what was the opening bracket that causes it to close?
        return prefixes, get_opening_brackets_prefix(dyck_str)
    else:
        return prefixes, [dyck_words[idx] for idx in closing_idxs]


def convert_state_into_s(stack_state):
    ### convert a sequence of push and pop into a dyck
    input_str = []
    stack = []

    stack_states = stack_state.split(",")
    vocab_open = ["(" + v for v in "abcdef"]
    for idx, state in enumerate(stack_states):
        if state == "(":
            curr_s = random.choice(vocab_open)
            stack.append(curr_s)
            input_str.append(curr_s)
        else:
            out = stack.pop()
            input_str.append(out[1] + ")")
    return input_str


def get_stack_state(input_str):
    stack = []
    input_words = input_str.split(" ")
    stack_states = []
    stack_all = []
    for idx, elem in enumerate(input_words):
        if "(" in elem:
            stack.append("(")
            stack_states.append("(")
        else:
            stack.pop()
            stack_states.append(")")
            if len(stack) == 0:
                stack_all.append(",".join(stack_states))
                stack_states = []
    return stack_all


def get_identifier_iterator():
    """Returns an iterator to provide unique ids to bracket types."""
    ids = iter(list(string.ascii_lowercase))
    k = 1
    while True:
        try:
            str_id = next(ids)
        except StopIteration:
            ids = iter(list(string.ascii_lowercase))
            k += 1
            str_id = next(ids)
        yield str_id * k


def get_vocab_of_bracket_types(bracket_types):
    """Returns the vocabulary corresponding to the number of brackets.

    There are bracket_types open brackets, bracket_types close brackets,
    START, and END.
    Arguments:
      bracket_types: int (k in Dyck-(k,m))
    Returns:
      Dictionary mapping symbol string  s to int ids.
    """
    id_iterator = get_identifier_iterator()
    ids = [next(id_iterator) for x in range(bracket_types)]
    vocab = {
        x: c
        for c, x in enumerate(
            ["(" + id_str for id_str in ids]
            + [id_str + ")" for id_str in ids]
            + ["START", "END"]
        )
    }
    return vocab, ids


class DyckPDFA:
    """
    Implements a probabilistic finite automata (PFA) that
    generates the dyck language
    """

    def __init__(self, max_stack_depth, bracket_types):
        self.max_stack_depth = max_stack_depth
        self.bracket_types = bracket_types
        self.vocab, self.ids = get_vocab_of_bracket_types(bracket_types)
        self.vocab_list = list(sorted(self.vocab.keys(), key=lambda x: self.vocab[x]))
        self.distributions = {}
        self.list_hash = {}

    def get_token_distribution_for_state(self, state_vec):
        """
        Given a stack state (list of ids, e.g., ['a', 'b']
        produces the probability distribution over next tokens
        """
        if state_vec in self.distributions:
            return self.distributions[state_vec]
        distrib_vec = torch.zeros(len(self.vocab))
        if len(state_vec) == 0:
            for id_str in self.ids:
                distrib_vec[self.vocab["(" + id_str]] = 1 / len(self.ids)
            distrib_vec[self.vocab["END"]] += 1
        elif len(state_vec) == self.max_stack_depth:
            distrib_vec[self.vocab[state_vec[-1] + ")"]] = 1
        else:
            for id_str in self.ids:
                distrib_vec[self.vocab["(" + id_str]] = 1 / len(self.ids)
            distrib_vec[self.vocab[state_vec[-1] + ")"]] = 1
        self.distributions[tuple(state_vec)] = torch.distributions.Categorical(
            distrib_vec / torch.sum(distrib_vec)
        )
        return self.distributions[state_vec]

    def update_state(self, state_vec, new_char_string):
        """
        Updates the DFA state based on the character new_char_string

        For a valid open/close bracket, pushes/pops as necessary.
        For an invalid open/close bracket, leaves state unchanged.
        """
        state_vec = list(state_vec)
        if ")" in new_char_string:
            bracket_type = new_char_string.strip(")")
            if len(state_vec) > 0 and state_vec[-1] == bracket_type:
                state_vec = state_vec[:-1]
        if "(" in new_char_string:
            bracket_type = new_char_string.strip("(")
            if len(state_vec) < self.max_stack_depth:
                state_vec.append(bracket_type)
        return state_vec

    def get_state_hash(self, state_vector):
        hash = []
        for elem in state_vector:
            if "(" in elem:
                hash.append("(")
            else:
                hash.append(")")
        return " ".join(hash)

    def sample(self, length_min, length_max=-1):
        """
        Returns a sample from the Dyck language, as well
        as the maximum number of concurrently-open brackets,
        and the number of times traversed from empty-stack to
        full-stack and back.
        """
        state_vec = []
        string = []
        max_state_len = 0
        empty_full_empty_traversals = 0
        empty_flag = True
        full_flag = False

        stack_path = []
        stack_paths = []
        while True:
            # probs = torch.distributions.Categorical(self.get_token_distribution_for_state(state_vec))
            probs = self.get_token_distribution_for_state(tuple(state_vec))
            new_char = probs.sample()
            new_char_string = self.vocab_list[int(new_char)]
            # Break from generation if END is permitted and sampled
            if new_char_string == "END":
                if len(string) < length_min:
                    continue
                else:
                    string.append(new_char_string)
                    break
            # Otherwise, update the state vector
            string.append(new_char_string)
            state_vec = self.update_state(state_vec, new_char_string)
            if len(state_vec) == 0:
                stack_path_curr = ",".join(stack_path)
                stack_paths.append(stack_path_curr)
                stack_path = []

            else:
                state_hash = self.get_state_hash(state_vec)
                if state_hash not in self.list_hash:
                    self.list_hash[state_hash] = str(len(self.list_hash))
                stack_path.append(self.list_hash[state_hash])

            max_state_len = max(max_state_len, len(state_vec))
            if len(state_vec) == self.max_stack_depth and empty_flag:
                full_flag = True
                empty_flag = False
            if len(state_vec) == 0 and full_flag:
                full_flag = False
                empty_flag = True
                empty_full_empty_traversals += 1
        if len(stack_path) != 0:
            stack_path_curr = ",".join(stack_path)
            stack_paths.append(stack_path_curr)
        return string, max_state_len, empty_full_empty_traversals, stack_paths


def get_training_data(pfsa, min_len, max_len, data_size=200000):
    unique_paths = []
    training_strings = []
    for idx in tqdm(range(data_size)):
        curr_string, _, _, stack_paths = pfsa.sample(min_len, max_len)
        unique_paths += stack_paths
        training_strings.append(curr_string)
    return training_strings, set(unique_paths)


def get_test_data(
    pfsa, min_len, max_len, training_strings, train_paths, data_size=2000
):
    iid_strings = []
    ood_strings = []
    while True:
        if len(ood_strings) == data_size and len(iid_strings) == data_size:
            break
        curr_string, _, _, paths = pfsa.sample(min_len, max_len)
        seen = all([path in train_paths for path in paths])
        if seen and curr_string not in training_strings:
            if len(iid_strings) < data_size:
                iid_strings.append(curr_string)
        elif not seen:
            if len(ood_strings) < data_size:
                ood_strings.append(curr_string)

        if len(ood_strings) % 100 == 0:
            print(len(ood_strings), len(iid_strings))

    return iid_strings, ood_strings


def write_to_file(fname, dataset):
    with open(fname, "w") as writer:
        for dat in dataset:
            curr_string = " ".join(dat)
            writer.write(curr_string)
            writer.write("\n")
    return


def build_datasets_dyck(vocab=20, stack_depth=10):
    def process(sent):
        words = sent.split(" ")[:-1]
        return " ".join(words)

    def read_data(splits):
        in_sentences = []
        index_map = {split: [] for split in splits}
        for split in splits:
            with open(
                "{}/dyck_data/k-{}_d-{}_{}.txt".format(
                    DATA_DIR, vocab, stack_depth, split
                ),
                "r",
            ) as reader:
                sents = [process(line.strip()) for line in reader.readlines()]
                for sent in sents:
                    index_map[split].append(len(in_sentences))
                    in_sentences.append(sent)
        return in_sentences, index_map

    def get_subset(elem_list, idx_list):
        return [elem_list[idx] for idx in idx_list]

    splits = ["train", "val", "test"]
    in_sentences, index_map = read_data(splits)
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



def eval_callback_dyck(lm, in_vocab, split):
    def tokenizer(s):
        return [lm.encoder_sos] + in_vocab(s)

    prefixes = []
    targets = []

    sents = read_dyck_data([split], 20)
    for sent in sents:
        if split == 'val':
            min_dep_length = None
        else:
            min_dep_length = 10
        prefixes_curr, targets_curr = get_all_prefixes(
            sent, min_dep_length=None, get_opening_idx=False
        )
        prefixes += prefixes_curr
        targets += targets_curr
        if len(prefixes) >= 5000:
            break

    out = test_continuations(tokenizer, lm, prefixes, 0)
    vocab_items_closing_brackets = [
        in_vocab.words[s] for s in in_vocab.words if ")" in s
    ]

    out_closing = out[:, vocab_items_closing_brackets]
    best_closing_entry = [
        vocab_items_closing_brackets[idx] for idx in out_closing.argmax(dim=1)
    ]
    accs = [pred == in_vocab(t)[0] for pred, t in zip(best_closing_entry, targets)]
    agg_acc = sum(accs) / len(prefixes)
    print(agg_acc)
    return agg_acc

if __name__ == "__main__":
    max_depth = 20
    vocab = 10
    dyck_pfsa = DyckPDFA(max_depth, vocab)

    training_strings, paths = get_training_data(dyck_pfsa, 4, 500, data_size=500000)
    iid_val_data, ood_test_data = get_test_data(
        dyck_pfsa, 4, 500, training_strings, paths, data_size=20000
    )

    print(len(training_strings))
    print(len(iid_val_data))

    write_to_file(
        "dyck_data/k-{}_d-{}_train.txt".format(vocab, max_depth), training_strings
    )
    write_to_file("dyck_data/k-{}_d-{}_val.txt".format(vocab, max_depth), iid_val_data)

    write_to_file(
        "dyck_data/k-{}_d-{}_test.txt".format(vocab, max_depth), ood_test_data
    )
