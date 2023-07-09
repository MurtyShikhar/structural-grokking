def right_branching_parse(input_str):
    input_words = input_str.split(" ")
    num_words = len(input_words)

    def add_brackets(st, en):
        if st == en:
            return input_words[st]
        else:
            p1 = add_brackets(st, st)
            p2 = add_brackets(st + 1, en)
            return (p1, p2)

    return add_brackets(0, num_words - 1)


def left_branching_parse(input_str):
    input_words = input_str.split(" ")
    num_words = len(input_words)

    def add_brackets(st, en):
        if st == en:
            return input_words[st]
        else:
            p1 = add_brackets(st, en - 1)
            p2 = add_brackets(en, en)
            return (p1, p2)

    return add_brackets(0, num_words - 1)


def random_parse(input_str):
    input_words = input_str.split(" ")
    num_words = len(input_words)

    def add_brackets(st, en):
        if st == en:
            return input_words[st]
        else:
            random_point = random.choice([k for k in range(st, en)])
            p1 = add_brackets(st, random_point)
            p2 = add_brackets(random_point + 1, en)
            return (p1, p2)

    return add_brackets(0, num_words - 1)


def balanced_parse(input_str):
    input_words = input_str.split(" ")
    num_words = len(input_words)

    def add_brackets(st, en):
        if st == en:
            return input_words[st]
        else:
            mid_point = (en - st + 1) // 2
            p1 = add_brackets(st, st + mid_point - 1)
            p2 = add_brackets(st + mid_point, en)
            return (p1, p2)

    return add_brackets(0, num_words - 1)
