# Code borrowed from McCoy et al: Does syntax need to grow on trees? Sources of hierarchical inductive bias in sequence-to-sequence networks
# Various functions to enable parsing of sentences
# from our artificial grammars
import os

# Load part-of-speech labels
posDict = {}
DIR = os.path.abspath(os.path.dirname(__file__))
fi = open("{}/data_utils/tense_inflection_data/pos.txt".format(DIR), "r")  # MIGHT NEED TO CHANGE BACK
for line in fi:
    parts = line.split("\t")
    posDict[parts[0].strip()] = parts[1].strip()

posDict2 = {}
fi = open("{}/data_utils/tense_inflection_data/pos2.txt".format(DIR), "r")
for line in fi:
    parts = line.split("\t")
    posDict2[parts[0].strip()] = parts[1].strip()


# Conert a sentence to part-of-speech tags
def sent_to_pos(sent):
    words = sent.split()

    pos_tags = []

    for word in words:
        if word in posDict:
            pos_tags.append(posDict[word])
        else:
            pos_tags.append(posDictTense[word])

    return pos_tags


# Convert a sentence to part-of-speech tags
# from the second part-of-speech file
def sent_to_posb(sent):
    words = sent.split()

    pos_tags = []

    for word in words:
        pos_tags.append(posDict2[word])

    return pos_tags


# Convert a sequence of part-of-speech tags into
# a parse. Works by successively grouping together
# neighboring pairs.
def pos_to_parse(pos_seq):
    full_parse = []

    current_parse = []
    current_nodes = pos_seq

    new_nodes = []
    skip_next = 0

    while len(current_nodes) > 1:
        for index, node in enumerate(current_nodes):
            if skip_next:
                skip_next = 0
                continue
            if node == "D" and current_nodes[index + 1] == "N":
                new_nodes.append("NP")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "P" and current_nodes[index + 1] == "NP":
                new_nodes.append("PP")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "P" and current_nodes[index + 1] == "NP_f":
                new_nodes.append("PP")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "NP" and current_nodes[index + 1] == "PP":
                new_nodes.append("NP")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "NP" and current_nodes[index + 1] == "RC":
                new_nodes.append("NP")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "V" and current_nodes[index + 1] == "NP_f":
                new_nodes.append("VP")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "V" and current_nodes[index + 1] == "T":
                new_nodes.append("VP")
                current_parse.append([index])
            elif node == "V" and current_nodes[index + 1] == "A":
                new_nodes.append("VP")
                current_parse.append([index])
            elif node == "V" and current_nodes[index + 1] == "VP":
                new_nodes.append("VP")
                current_parse.append([index])
            elif node == "VP" and current_nodes[index + 1] == "VP_f":
                new_nodes.append("VP_f")
                current_parse.append([index])
            elif node == "NP" and current_nodes[index + 1] == "T":
                new_nodes.append("NP_f")
                current_parse.append([index])
            elif node == "NP" and current_nodes[index + 1] == "VP_f":
                new_nodes.append("S")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "S" and current_nodes[index + 1] == "T":
                new_nodes.append("ROOT")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "A" and current_nodes[index + 1] == "S_bar":
                new_nodes.append("A_S_bar")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "A" and current_nodes[index + 1] == "S":
                new_nodes.append("A_S_bar")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "K" and current_nodes[index + 1] == "ROOT":
                new_nodes.append("ROOT")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "K" and current_nodes[index + 1] == "ROOT":
                new_nodes.append("ROOT")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "A" and current_nodes[index + 1] == "VP":
                new_nodes.append("VP_f")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "C" and current_nodes[index + 1] == "VP":
                new_nodes.append("VP")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "R" and current_nodes[index + 1] == "VP_f":
                new_nodes.append("RC")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "R" and current_nodes[index + 1] == "S":
                new_nodes.append("RC")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "R" and current_nodes[index + 1] == "S_bar":
                new_nodes.append("RC")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "NP" and current_nodes[index + 1] == "A":
                new_nodes.append("NP_f")
                current_parse.append([index])
            elif node == "NP_f" and current_nodes[index + 1] == "VP_f":
                new_nodes.append("S")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "V" and current_nodes[index + 1] == "NP_bar":
                new_nodes.append("VP")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "NP_bar" and current_nodes[index + 1] == "VP":
                new_nodes.append("S_bar")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "NP" and current_nodes[index + 1] == "VP":
                new_nodes.append("NP_bar")
                current_parse.append([index])
            elif node == "A_S_bar" and current_nodes[index + 1] == "T":
                new_nodes.append("ROOT")  # CHANGE
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "ROOT" and current_nodes[index + 1] == "G":
                new_nodes.append("ROOT")
                current_parse.append([index, index + 1])
                skip_next = 1
            else:
                new_nodes.append(node)
                current_parse.append([index])

        current_nodes = new_nodes
        new_nodes = []
        skip_next = 0
        full_parse.append(current_parse)
        current_parse = []

    full_parse.append([[0]])

    return full_parse


# Parse a sentence from the question formation dataset
def parse_question(sent):
    return pos_to_parse(sent_to_posb(sent))


# Create a part-of-speech dictionary for tense reinflection sentences
posDictTense = {}
fi = open("{}/data_utils/tense_inflection_data/pos_tense.txt".format(DIR), "r")
for line in fi:
    parts = line.split("\t")
    posDictTense[parts[0].strip()] = parts[1].strip()

# Convert a tense reinflection sentence into
# a sequence of part-of-speech tags
def sent_to_pos_tense(sent):
    words = sent.split()

    pos_tags = []

    for word in words:
        pos_tags.append(posDictTense[word])

    return pos_tags


# Convert a sequence of part-of-speech tags into a parse.
# Works by successively grouping together
# neighboring pairs.
def pos_to_parse_tense(pos_seq):
    full_parse = []

    current_parse = []
    current_nodes = pos_seq

    new_nodes = []
    skip_next = 0

    while len(current_nodes) > 1:
        for index, node in enumerate(current_nodes):
            if skip_next:
                skip_next = 0
                continue
            if node == "D" and current_nodes[index + 1] == "N":
                new_nodes.append("NP")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "P" and current_nodes[index + 1] == "NP":
                new_nodes.append("PP")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "P" and current_nodes[index + 1] == "NP_f":
                new_nodes.append("PP")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "NP" and current_nodes[index + 1] == "PP":
                new_nodes.append("NP")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "NP" and current_nodes[index + 1] == "RC":
                new_nodes.append("NP")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "V" and current_nodes[index + 1] == "NP_f":
                new_nodes.append("VP_f")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif (
                node == "V"
                and current_nodes[index + 1] == "NP"
                and current_nodes[index + 2] == "VP_f"
            ):
                new_nodes.append("VP_f")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "V" and current_nodes[index + 1] == "T":
                new_nodes.append("VP_f")
                current_parse.append([index])
            elif node == "V" and current_nodes[index + 1] == "VP_f":
                new_nodes.append("VP_f")
                current_parse.append([index])
            elif node == "NP" and current_nodes[index + 1] == "T":
                new_nodes.append("NP_f")
                current_parse.append([index])
            elif node == "NP" and current_nodes[index + 1] == "VP_f":
                new_nodes.append("S")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "S" and current_nodes[index + 1] == "T":
                new_nodes.append("ROOT")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "K" and current_nodes[index + 1] == "ROOT":
                new_nodes.append("ROOT")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "K" and current_nodes[index + 1] == "ROOT":
                new_nodes.append("ROOT")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "R" and current_nodes[index + 1] == "VP_f":
                new_nodes.append("RC")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "R" and current_nodes[index + 1] == "S":
                new_nodes.append("RC")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "NP" and current_nodes[index + 1] == "A":
                new_nodes.append("NP_f")
                current_parse.append([index])
            elif node == "NP_f" and current_nodes[index + 1] == "VP_f":
                new_nodes.append("S")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "V" and current_nodes[index + 1] == "NP_bar":
                new_nodes.append("VP_f")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "NP_bar" and current_nodes[index + 1] == "VP_f":
                new_nodes.append("S_bar")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "NP" and current_nodes[index + 1] == "VP_f":
                new_nodes.append("NP_bar")
                current_parse.append([index])
            elif node == "A_S_bar" and current_nodes[index + 1] == "T":
                new_nodes.append("ROOT")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "ROOT" and current_nodes[index + 1] == "G":
                new_nodes.append("ROOT")
                current_parse.append([index, index + 1])
                skip_next = 1
            else:
                new_nodes.append(node)
                current_parse.append([index])

        current_nodes = new_nodes
        new_nodes = []
        skip_next = 0
        full_parse.append(current_parse)
        current_parse = []

    full_parse.append([[0]])

    return full_parse


# Parse a sentence from the tense reinflection dataset
def parse_tense(sent):
    return pos_to_parse_tense(sent_to_pos_tense(sent))


# Returns a uniformly right-branching parse
def parse_right_branching(sent):
    len_sent = len(sent.split())

    full_parse = []

    for i in range(len_sent):
        if i == 0:
            new_part = [[0]]
        elif i == 1:
            new_part = [[0, 1]]
        else:
            new_part = [[j] for j in range(i - 1)] + [[i - 1, i]]

        full_parse = [new_part] + full_parse

    return full_parse


def convert_parse_to_list(sent, parse):
    words = sent.split(" ")

    def helper():
        return


def convert_to_parse(sent, parse_list, do_postprocess=True):
    words = sent.split(" ")

    def parse_helper(l_idx):
        if l_idx == 0:
            out_prev = words
        else:
            out_prev = parse_helper(l_idx - 1)
        elems = parse_list[l_idx]  ### index into the previous
        out = []
        for elem in elems:
            curr = [out_prev[idx] for idx in elem]
            if len(curr) > 1:
                out.append(tuple(curr))
            else:
                out.append(curr[0])
        return tuple(out)

    parsed_inp = parse_helper(len(parse_list) - 1)
    if len(parsed_inp) == 1:
        parsed_inp = parsed_inp[0]

    if do_postprocess:
        if parsed_inp[-1] in ["quest", "decl", "past", "present"]:
            parsed_inp = parsed_inp[0]
        if parsed_inp[-1] == ".":
            parsed_inp = parsed_inp[0]
    return parsed_inp


def convert_into_layerwise_format(sent, parse):
    def refine_layer(frontier, layer, g_obj):
        so_far = {}
        next_nodes = {}
        parents = g_obj.parent

        next_layer = []
        for idx, _ in enumerate(layer):
            parent = parents[frontier[idx]]
            if parent in so_far:
                index = so_far[parent]
                next_layer[index].append(idx)
            else:
                so_far[parent] = len(so_far)
                next_nodes[len(next_layer)] = (parent, frontier[idx])
                next_layer.append([idx])

        new_frontier = {}
        for key, node in next_nodes.items():
            if len(next_layer[key]) == 1:
                new_frontier[key] = node[1]
            else:
                new_frontier[key] = node[0]
        return new_frontier, next_layer

    from graph_node import Graph

    g_obj = Graph(parse)
    words = sent.split(" ")

    curr_layer = [[idx] for idx in range(len(words))]
    nodes = {idx[0]: g_obj.idx_dict[idx[0]] for idx in curr_layer}

    next_nodes = nodes
    next_layer = curr_layer

    layered_solution = [curr_layer]
    while True:
        next_nodes, next_layer = refine_layer(next_nodes, next_layer, g_obj)
        layered_solution.append(next_layer)
        if len(next_layer) == 1:
            break
    return layered_solution


def parse_tense_patched(sent):
    parse = convert_to_parse(sent, parse_tense(sent), do_postprocess=False)
    return convert_into_layerwise_format(sent, parse)


if __name__ == "__main__":
    from data_utils.lm_dataset_helpers import read_lm_data
    from data_utils.tense_inflection_helpers import read_ti_data

    in_sentences, _ = read_ti_data(["train"], do_process=False)
    import random

    for idx in random.sample([idx for idx in range(10000)], k=1000):
        sent = in_sentences[idx].strip().split("\t")[0].strip().lower()
        parse = convert_to_parse(sent, parse_tense(sent), do_postprocess=False)
        assert parse == convert_to_parse(
            sent, convert_into_layerwise_format(sent, parse), do_postprocess=False
        )
