import argparse

def read_f(fname):
    def process(s):
        all_words = s.split(' ')
        return ' '.join(all_words[:-1] + ['<|endoftext|>'])
    all_lines = [process(l.strip()) for l in open(fname).readlines()]
    data = ' '.join(all_lines)
    fwrite = fname.split('.')[0] + '_gpt.txt'
    with open(fwrite, 'w') as writer:
        writer.write(data)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str)
    args = parser.parse_args()


    read_f(args.file_name)
