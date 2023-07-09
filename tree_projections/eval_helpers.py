def get_parsing_accuracy(predicted_parses, gold_parses):
    def get_brackets(parse):
        p_set = set()

        def get_brackets_helpers(t, st):
            if type(t) == str:
                return 1
            else:
                l1_len = get_brackets_helpers(t[0], st)
                l2_len = get_brackets_helpers(t[1], st + l1_len)
                p_set.add((st, st + l1_len + l2_len - 1))
                return l1_len + l2_len

        get_brackets_helpers(parse, 0)
        return p_set

    if type(gold_parses[0]) != tuple:
        gold_brackets = [get_node_brackets(parse) for parse in gold_parses]
    else:
        gold_brackets = [get_brackets(parse) for parse in gold_parses]

    pred_brackets = [get_brackets(parse) for parse in predicted_parses]

    def get_score(set_1, set_2):
        score = 0.0
        for p in set_2:
            if p in set_1:
                score += 1
        return score

    # to restrict to noun phrases, precision means how many of the NP constituents discovered are actual NPs
    # recall means how many of the NP constituents in the gold are also constituents in our model.
    precision = sum(
        [get_score(gold, pred) for gold, pred in zip(gold_brackets, pred_brackets)]
    )
    recall = sum(
        [get_score(pred, gold) for gold, pred in zip(gold_brackets, pred_brackets)]
    )
    precision /= 1.0 * sum(len(b) for b in pred_brackets)
    recall /= 1.0 * sum(len(b) for b in gold_brackets)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / (precision + recall + 1e-10),
    }
