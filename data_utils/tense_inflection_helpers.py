### Code borrowed from McCoy et al: Does syntax need to grow on trees? Sources of hierarchical inductive bias in sequence-to-sequence networks

from vocabulary import WordVocabulary
from datasets import Dataset as HFDataset
from util import run_lm_decoding
import random
import os

# Determines whether the main verb of the sentence is correct
### target is first, pred is second
def main_right_tense(senta, sentb):

    if not right_pos(senta, sentb):
        return False

    wordsa = senta.split()
    wordsb = sentb.split()

    pos_tags = sent_to_pos(senta)

    if pos_tags[2] == "R":
        seen_v = 0
        for index, tag in enumerate(pos_tags):
            if tag == "V":
                if seen_v:
                    ind_v = index
                    break
                else:
                    seen_v = 1

    else:
        for index, tag in enumerate(pos_tags):
            if tag == "V":
                ind_v = index
                break

    verba = wordsa[ind_v]
    verbb = wordsb[ind_v]

    return verbb == verba


# Converting a sentence to a list of part-of-speech tags
DATA_DIR = os.path.abspath(os.path.dirname(__file__))
posDictTense = {}
fi = open("{}/tense_inflection_data/pos_tense.txt".format(DATA_DIR), "r")
for line in fi:
    parts = line.split("\t")
    posDictTense[parts[0].strip()] = parts[1].strip()

posDict = {}
fi = open("{}/tense_inflection_data/pos.txt".format(DATA_DIR), "r")
for line in fi:
    parts = line.split("\t")
    posDict[parts[0].strip()] = parts[1].strip()


def sent_to_pos(sent):
    words = sent.split()

    pos_tags = []

    for word in words:
        if word in posDict:
            pos_tags.append(posDict[word])
        else:
            pos_tags.append(posDictTense[word])

    return pos_tags


# Does the sentence have the correct sequence of
# part of speech tags?
def right_pos(senta, sentb):
    pos_tags_a = sent_to_pos(senta)
    pos_tags_b = sent_to_pos(sentb)

    if pos_tags_a == pos_tags_b:
        return True
    else:
        return False
nouns_sg = ["newt", "orangutan", "peacock", "quail", "raven", "salamander", "tyrannosaurus", "unicorn", "vulture", "walrus", "xylophone", "yak", "zebra"]
nouns_pl = ["newts", "orangutans", "peacocks", "quails", "ravens", "salamanders", "tyrannosauruses", "unicorns", "vultures", "walruses", "xylophones", "yaks", "zebras"]

verbs_sg = ["giggles", "smiles", "sleeps", "swims", "waits", "moves", "changes", "reads", "eats", "entertains", "amuses", "high_fives", "applauds", "confuses", "admires", "accepts", "remembers", "comforts"]
verbs_pl = ["giggle", "smile", "sleep", "swim", "wait", "move", "change", "read", "eat", "entertain", "amuse", "high_five", "applaud", "confuse", "admire", "accept", "remember", "comfort"]

auxes_sg = ["does", "doesn't"]
auxes_pl = ["do", "don't"]

# Given an input past tense sentence, outputs
# what the present-tense version would be if
# verbs agreed with the most recent noun instead
# of with their subjects.
def tense_nearest(sent):
    new_words = []
    words = sent.split()
    tense_agr = "sg"
    for word in words:
        if word in nouns_sg:
            tense_agr = "sg"
            new_words.append(word)
        elif word in nouns_pl:
            tense_agr = "pl"
            new_words.append(word)
        elif word in verbs_sg:
            verb_ind = verbs_sg.index(word)
            if tense_agr == "sg":
                new_words.append(verbs_sg[verb_ind])
            else:
                new_words.append(verbs_pl[verb_ind])
        elif word in verbs_pl:
            verb_ind = verbs_pl.index(word)
            if tense_agr == "sg":
                new_words.append(verbs_sg[verb_ind])
            else:
                new_words.append(verbs_pl[verb_ind])
        else:
            new_words.append(word)
    return " ".join(new_words)



def process(line):
    return line.replace("\t", " ")


def read_ti_data(splits, do_process=True):
    in_sentences = []
    index_map = {split: [] for split in splits}
    for split in splits:
        with open(
            "{}/tense_inflection_data/tense.{}".format(DATA_DIR, split),
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


def build_datasets_tense_inflection():
    def get_subset(elem_list, idx_list):
        return [elem_list[idx] for idx in idx_list]

    splits = ["train", "val", "test"]
    in_sentences, index_map = read_ti_data(splits)
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


def eval_callback_tense_inflection(lm, in_vocab, split):
    def tokenizer(s):
        return [lm.encoder_sos] + in_vocab(s)

    def remove_fullstop(s):
        if s[-1] == ".":
            return " ".join(s.split(" ")[:-1])
        else:
            return s

    sents, _ = read_ti_data([split])
    if len(sents) > 2000:
        sents = random.sample(sents, k=2000)

    split_into_words = [sent.split(" ") for sent in sents]
    input_sents = []
    target_sents = []
    for sent_words in split_into_words:
        if "PRESENT" in sent_words:
            word = "PRESENT"
        elif "PAST" in sent_words:
            word = "PAST"
        idx = sent_words.index(word)
        input_sents.append(" ".join(sent_words[: idx + 1]))
        target_sents.append(" ".join(sent_words[idx + 1 :]))

    out = run_lm_decoding(tokenizer, lm, input_sents, 0)

    main_correct = 0.0
    for target_sent, our_pred in zip(target_sents, out):
        pred = remove_fullstop(" ".join(in_vocab(our_pred)))
        target = remove_fullstop(target_sent)
        main_correct += main_right_tense(target, pred)
    ### check exact match acc
    return main_correct / len(target_sents)

