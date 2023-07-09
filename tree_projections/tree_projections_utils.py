from tqdm import tqdm
import torch
import numpy as np
import random
import pickle
import collate
from scipy.spatial import distance

device = torch.device("cuda")


def read_inputs_and_parses(input_file):
    def binarize(tree):
        if type(tree) == str:
            return tree
        elif len(tree) == 1:
            return binarize(tree[0])
        else:
            lchild = binarize(tree[0])
            rchild = binarize(tree[1:])
            return (lchild, rchild)

    with open(input_file, "rb") as reader:
        data = pickle.load(reader)
    if "ptb" in input_file or "pcfg" in input_file:
        return data[0], data[1]
    else:
        strs, parses = [], []
        for l in data:
            strs.append(l)
            parses.append(data[l])
    return strs, parses


def get_sparsity_scores_helper(model, tokenizer, input_list):
    batch_size = 32
    st = 0
    train_data_collator = collate.VarLengthCollate(None)

    def tokenizer_helper(inp_slice):
        inp_list = [tokenizer(s) for s in inp_slice]
        in_lens = [len(s) for s in inp_list]
        inp_to_collate = [{"in": x, "in_len": y} for x, y in zip(inp_list, in_lens)]
        inp = train_data_collator(inp_to_collate)
        in_len = inp["in_len"].long()
        return inp["in"].transpose(0, 1), in_len

    total_sparsity = 0.0
    with tqdm(total=len(input_list)) as progress_bar:
        while st < len(input_list):
            en = min(len(input_list), st + batch_size)
            cslice = input_list[st:en]
            inputs, input_lens = tokenizer_helper(cslice)
            inputs = inputs.to(device)
            input_lens = input_lens.to(device)
            # input masks specify the inner context
            model.eval()
            with torch.no_grad():
                total_sparsity += model.get_attention_sparsity(inputs, input_lens)
            progress_bar.update(en - st)
            st = en
    return (total_sparsity.item()) / (1.0 * len(input_list))


def get_all_hidden_states(
    model,
    tokenizer,
    input_list,
    input_masks=None,
    layer_id=-1,
    all_layers=False,
    sum_all=False,
    start_relax_layer=0,
    end_relax_layer=0,
    tqdm_disable=False,
    pre_tokenized=None,
):
    def relax_cond(mask, relax_mask, num_layers):
        if input_masks is None:
            return relax_mask
        else:
            ### relax mask only masks padded stuff
            #### relax mask from 0 ... start_relax_layer-1,
            #### relax_mask from num_layers - end_relax_layer to num_layers - 1
            #### mask from start_relax_layer to num_layers - end_layers - 1
            return [relax_mask] * start_relax_layer + [mask] * (
                num_layers - start_relax_layer
            )

    hidden_states_all = []
    # add special tokens because we don't the confounder where removing special tokens causes
    # the model to not work
    batch_size = 4096
    st = 0
    num_layers = model.config.num_hidden_layers
    with tqdm(total=len(input_list), disable=tqdm_disable) as progress_bar:
        while st < len(input_list):
            en = min(len(input_list), st + batch_size)
            cslice = input_list[st:en]
            tokenized = tokenizer(cslice, return_tensors="pt", padding=True)
            inputs = tokenized.input_ids
            inp_len = inputs.shape[1]
            inputs = inputs.to(device)
            # input masks specify the inner context
            if input_masks is not None:
                masks_curr = input_masks[st:en]
                masks_padded = []
                for mask in masks_curr:
                    mask_padded = mask + [0] * (inp_len - len(mask))
                    masks_padded.append(mask_padded)
                mask = torch.tensor(masks_padded).to(device)
            else:
                mask = tokenized.attention_mask.to(device)

            model.eval()
            with torch.no_grad():
                outputs = model(
                    input_ids=inputs,
                    attention_mask=relax_cond(
                        mask, tokenized.attention_mask.to(device), num_layers
                    ),
                    output_hidden_states=True,
                ).hidden_states
            # remove vectors for masked stuff
            mask_mult = mask.unsqueeze(-1)
            outputs = [hs * mask_mult for hs in outputs]
            for idx, _ in enumerate(cslice):
                if all_layers:
                    hidden_states = [
                        hs[idx].cpu().numpy() for hs in outputs
                    ]  # list of size num layers. each layer has D sub-word vectors.
                else:
                    hidden_states = [outputs[layer_id][idx].cpu().numpy()]
                if sum_all:
                    # the first thing is the [CLS] or [start id] which we ignore
                    # the secnd thing is the [EOS] token which we also ignore.
                    hidden_states = [hs[1:-1].sum(axis=0) for hs in hidden_states]
                hidden_states_all.append(hidden_states)
            progress_bar.update(en - st)
            st = en
    if sum_all:
        return hidden_states_all
    else:
        return get_word_vecs_from_subwords(
            input_list, hidden_states_all, tokenizer, pre_tokenized
        )


def get_all_hidden_states_scratch(
    model,
    tokenizer,
    input_list,
    input_masks=None,
    sum_all=False,
    tqdm_disable=False,
    pre_tokenized=None,
    start_relax_layer=0,
    layer_id=-1,
    is_lm=False,
):
    def relax_cond(mask, relax_mask, num_layers):
        ### relax mask only masks padded stuff
        ### mas masked everything
        #### relax mask from 0 ... start_relax_layer-1,
        #### relax_mask from num_layers - end_relax_layer to num_layers - 1
        #### mask from start_relax_layer to num_layers - end_layers - 1
        return [relax_mask] * start_relax_layer + [mask] * (
            num_layers - start_relax_layer
        )

    hidden_states_all = []
    batch_size = 256
    st = 0

    train_data_collator = collate.VarLengthCollate(None)

    def tokenizer_helper(inp_slice):
        inp_list = [tokenizer(s) for s in inp_slice]
        in_lens = [len(s) for s in inp_list]

        inp_to_collate = [{"in": x, "in_len": y} for x, y in zip(inp_list, in_lens)]
        inp = train_data_collator(inp_to_collate)
        in_len = inp["in_len"].long()
        return inp["in"].transpose(0, 1), in_len

    num_layers = model.get_encoder_layers()
    with tqdm(total=len(input_list), disable=tqdm_disable) as progress_bar:
        while st < len(input_list):
            en = min(len(input_list), st + batch_size)
            cslice = input_list[st:en]
            inputs, input_lens = tokenizer_helper(cslice)
            inputs = inputs.to(device)
            input_lens = input_lens.to(device)
            inp_len = inputs.shape[1]
            # input masks specify the inner context
            if input_masks is not None:
                masks_curr = input_masks[st:en]
                masks_padded = []
                for mask in masks_curr:
                    mask_padded = mask + [True] * (inp_len - len(mask))
                    masks_padded.append(mask_padded)
                tree_mask = torch.tensor(masks_padded).to(device)
                relax_mask = model.generate_len_mask(inp_len, input_lens).to(device)
                mask = relax_cond(tree_mask, relax_mask, num_layers)
                mask_mult = tree_mask.unsqueeze(-1)
            else:
                mask = model.generate_len_mask(inp_len, input_lens).to(device)
                mask_mult = mask.unsqueeze(-1)
            model.eval()
            with torch.no_grad():
                outputs = [model.encoder_only(inputs, mask, layer_id=layer_id)]

            # remove vectors for masked stuff
            # REMEMBER: mask is 1 if the token is not attended to, and 0 if the token is attended to.
            outputs = [hs * (~mask_mult) for hs in outputs]
            for idx, _ in enumerate(cslice):
                hidden_states = [outputs[0][idx].cpu().numpy()]
                if sum_all:
                    # the first thing is the [CLS] or [start id] which we ignore
                    # for non-LMS, the secnd thing is the [EOS] token which we also ignore.
                    if is_lm:
                        hidden_states = [hs[1:].sum(axis=0) for hs in hidden_states]
                    else:
                        hidden_states = [hs[1:-1].sum(axis=0) for hs in hidden_states]
                hidden_states_all.append(hidden_states)
            progress_bar.update(en - st)
            st = en

    if sum_all:
        return hidden_states_all
    else:
        return get_word_vecs_from_subwords(
            input_list, hidden_states_all, tokenizer, pre_tokenized
        )


def get_cumulants(hs, idx_list):
    all_vecs = []
    for st, en in idx_list:
        curr_vec = hs[st:en].sum(axis=0)  # sum up all the BPE token representations!
        all_vecs.append(curr_vec)
    try:
        return np.stack(all_vecs, axis=0)
    except:
        import pdb

        pdb.set_trace()


def get_idxs(phrase_tokens, sent_tokens, st):
    while st < len(sent_tokens):
        en = st + len(phrase_tokens)
        if sent_tokens[st:en] == phrase_tokens:
            return (st, en)
        st += 1
    raise Exception


def get_word_vecs_from_subwords(
    input_list, hidden_states_all, tokenizer, pre_tokenized=None
):
    cumulants = []
    if pre_tokenized:
        sent_token_list, word_tokens_all = pre_tokenized
    else:
        sent_token_list = tokenizer(input_list, padding=True).input_ids
    for idx, input_str in enumerate(input_list):
        sent_tokens = sent_token_list[idx]
        curr_hidden_states = hidden_states_all[idx]
        if pre_tokenized:
            idxs = word_tokens_all[idx]
        else:
            words = input_str.split(" ")
            idxs = []
            # go in order.
            st = 0
            for word in words:
                word_tokenized = tokenizer(word, add_special_tokens=False).input_ids
                st_curr, en_curr = get_idxs(word_tokenized, sent_tokens, st)
                idxs.append((st_curr, en_curr))
                st = en_curr

        hs_cumulated = [get_cumulants(hs, idxs) for hs in curr_hidden_states]
        cumulants.append(hs_cumulated)
    return cumulants


def mask_all_possible(input_str, tokenizer, masking_type="attention"):
    """
    masking_type can be token or attention
    """
    all_tokens = input_str.split(" ")
    tokenized_inp = tokenizer(input_str)
    word_tokenized = [tokenizer(word, add_special_tokens=False) for word in all_tokens]

    # the starting point of each word
    st_p = 0
    while st_p < len(tokenized_inp):
        en = st_p + len(word_tokenized[0])
        if tokenized_inp[st_p:en] == word_tokenized[0]:
            break
        st_p += 1
    cumulants = [st_p]
    for w_tokenized in word_tokenized:
        cumulants.append(cumulants[-1] + len(w_tokenized))

    # the final thing might be a special token too. 0_0
    en_p = len(tokenized_inp) - cumulants[-1]
    # en_p is 0 for an LM, and non-zero if there are special tokens at the end
    assert en_p >= 0

    def generate_attention_mask(st, en):
        if masking_type == "attention":
            block_len = cumulants[en] - cumulants[st]
            out = (
                [False] * st_p
                + [True] * (cumulants[st] - st_p)
                + [False] * block_len
                + [True] * (cumulants[-1] - cumulants[en])
                + [False] * en_p
            )
            assert len(out) == len(tokenized_inp)
        else:
            # 0...st-1 is masked st... en-1 is open, en...len(all_tokens)-1
            mask_token = tokenizer.mask_token
            masked_tokens = (
                [mask_token] * st
                + all_tokens[st:en]
                + [mask_token] * (len(all_tokens) - en)
            )
            out = " ".join(masked_tokens)
        return out

    sz = len(all_tokens)
    # l = [1, ... sz]
    # st = [0, 1, 2, ..., sz - l-1]
    all_inputs = {}
    # get new attention masks.
    for l in range(1, sz + 1):
        for st in range(sz - l + 1):
            en = st + l
            # only the stuff from st:en can be attended to.
            all_inputs[(st, en - 1)] = generate_attention_mask(st, en)
    return all_inputs


def get_masking_info(tokenizer, input_strs, masking_type="attention"):
    masked_strs = []
    curr = 0
    sentence2idx_tuple = []
    if masking_type == "attention":
        input_masks = []
    else:
        input_masks = None
    for inp in input_strs:
        input_dict = mask_all_possible(inp, tokenizer, masking_type=masking_type)
        curr_keys = [k for k in input_dict]
        if masking_type == "attention":
            masked_strs += [inp] * len(input_dict)
            input_masks += [input_dict[key] for key in curr_keys]
        else:
            masked_strs += [input_dict[key] for key in curr_keys]
        relative_idxs = [(curr + p, key) for p, key in enumerate(curr_keys)]
        curr += len(curr_keys)
        sentence2idx_tuple.append(relative_idxs)

    return sentence2idx_tuple, masked_strs, input_masks


def measure_sim_factory(distance_fn):
    def measure_sim(m1, m2):
        if m1.ndim == 2:
            assert len(m1) == len(m2)
            return sum(measure_sim(m1[idx], m2[idx]) for idx in range(len(m1))) / (
                1.0 * len(m1)
            )
        elif distance_fn == distance.cosine:
            return 1.0 - distance_fn(m1, m2)
        else:
            return -1.0 * distance_fn(m1, m2)

    return measure_sim


def approximation_error(chart_values, input_str, parse):
    num_words = len(input_str.split(" "))

    def score_recurse(st, en):
        if st == en:
            return 0.0
        split_point = parse[(st, en)]
        s = chart_values[(st, split_point)] + chart_values[(split_point + 1, en)]
        return s + score_recurse(st, split_point) + score_recurse(split_point + 1, en)

    return score_recurse(0, num_words - 1)


def tree_projection(
    chart_values,
    input_str,
    get_score_parse=False,
    normalize=False,
    is_leaf_fn=None,
    is_invalid_fn=None,
):
    num_words = len(input_str.split(" "))

    def tree_projection_recurse(word_list, st, en, randomize=False):
        if is_leaf_fn is not None and is_leaf_fn(word_list, st, en):
            return " ".join(word_list[st : en + 1]), 0.0
        elif st == en:
            return word_list[st], 0.0
        else:
            curr_split = st
            best_val = -10000
            if randomize:
                curr_split = random.choice(range(st, en))
            else:
                for k in range(st, en):
                    if is_invalid_fn is not None and is_invalid_fn(
                        word_list, st, k, en
                    ):
                        continue
                    curr_val = chart_values[(st, k)] + chart_values[(k + 1, en)]
                    if curr_val > best_val:
                        best_val = curr_val
                        curr_split = k
            p1, s1 = tree_projection_recurse(word_list, st, curr_split)
            p2, s2 = tree_projection_recurse(word_list, curr_split + 1, en)
            if normalize:
                rand_split = random.choice(range(st, en))
                rand_val = (
                    chart_values[(st, rand_split)] + chart_values[(rand_split + 1, en)]
                )
                best_val -= rand_val

            return (p1, p2), s1 + s2 + best_val

    word_list = input_str.split(" ")
    parse, score = tree_projection_recurse(word_list, 0, num_words - 1)

    if get_score_parse:
        return score
    else:
        return chart_values, parse, score


def get_pre_tokenized_info(input_str, tokenizer, is_roberta=False):
    sent_tokens = tokenizer(input_str)
    if is_roberta:
        sent_tokens = sent_tokens["input_ids"]
    words = input_str.split(" ")
    idxs = []
    # go in order.
    st = 0
    for idx, word in enumerate(words):
        if is_roberta and idx != 0:
            word = " " + word
        word_tokenized = tokenizer(word, add_special_tokens=False)
        if is_roberta:
            word_tokenized = word_tokenized["input_ids"]
        st_curr, en_curr = get_idxs(word_tokenized, sent_tokens, st)
        idxs.append((st_curr, en_curr))
        st = en_curr
    return sent_tokens, idxs


def get_tree_projection(
    input_str,
    model,
    tokenizer,
    st_threshold=0,
    verbose=False,
    sim_fn="euclidean",
    normalize=False,
    is_leaf_fn=None,
    is_invalid_fn=None,
    layer_id=-1,
    is_lm=False,
):
    sent_tokens, idxs = get_pre_tokenized_info(input_str, tokenizer)
    sentence2idx_tuple, masked_strs, input_masks = get_masking_info(
        tokenizer, [input_str], masking_type="attention"
    )
    outer_context_vecs = get_all_hidden_states_scratch(
        model,
        tokenizer,
        [input_str],
        tqdm_disable=True,
        pre_tokenized=([sent_tokens], [idxs]),
        layer_id=layer_id,
        is_lm=is_lm,
    )
    inner_context_vecs = get_all_hidden_states_scratch(
        model,
        tokenizer,
        masked_strs,
        input_masks,
        sum_all=True,
        start_relax_layer=st_threshold,
        tqdm_disable=True,
        pre_tokenized=([sent_tokens] * len(masked_strs), [idxs] * len(masked_strs)),
        layer_id=layer_id,
        is_lm=is_lm,
    )
    keys = sentence2idx_tuple[0]

    if sim_fn == "euclidean":
        measure_sci = measure_sim_factory(distance.euclidean)
    else:
        measure_sci = measure_sim_factory(distance.cosine)

    num_layers = len(inner_context_vecs[0])
    sci_chart = [{} for _ in range(num_layers)]
    scores = [0.0 for _ in range(num_layers)]
    parses = []
    for layer in range(num_layers):
        all_vector_idxs = outer_context_vecs[0][
            layer
        ]  # this is the hidden state of the fully contextualized model
        for idx, key in keys:
            st, en = key

            fully_contextual_vectors = all_vector_idxs[st : en + 1].sum(
                axis=0
            )  # comsiders everything
            inner_context_vectors = inner_context_vecs[idx][
                layer
            ]  # only consider the words inside the context
            sci_chart[layer][(st, en)] = measure_sci(
                inner_context_vectors, fully_contextual_vectors
            )
        chart, parse, score = tree_projection(
            sci_chart[layer],
            input_str,
            get_score_parse=False,
            normalize=normalize,
            is_leaf_fn=is_leaf_fn,
            is_invalid_fn=is_invalid_fn,
        )
        parses.append(parse)
        sci_chart[layer] = chart
        scores[layer] = score
    if verbose:
        return {
            "sci_chart": sci_chart[-1],
            "pred_parse_score": scores[-1],
            "pred_parse": parses[-1],
        }
    else:
        return parses[-1]
