from collections import Counter
import argparse
import math
import re


SET_FILE_HEADER_LINE_REGEX = r"^<\w+\s+\d+\s*(.*)>$"
WORD_IS_COUNT_THRESHOLD = 4
LIDSTONE_LAMBDA = 0.5
ITERATION_COUNT = 30
CLUSTERS = range(9)
EPSILON = 0.0001


def get_vocabulary(articles_file_path):
    words_counter = Counter()
    with open(articles_file_path) as articles_file:
        for line in articles_file:
            line = line.strip()
            if not re.match(SET_FILE_HEADER_LINE_REGEX, line):
                words_counter.update(line.split())

    return {y: x for x, y in enumerate(filter(lambda x: words_counter[x] >= WORD_IS_COUNT_THRESHOLD, words_counter))}


def get_articles(articles_file_path, vocabulary):
    articles = []
    with open(articles_file_path) as articles_file:
        for line in articles_file:
            line = line.strip()
            if line == "":
                continue
            header_line_match = re.match(SET_FILE_HEADER_LINE_REGEX, line)
            if header_line_match:
                articles.append({})
                articles[-1]["topics"] = header_line_match.group(1).split()
            else:
                articles[-1]["text"] = " ".join(filter(lambda x: x in vocabulary, line.split()))

    return articles


def build_wti(articles):
    wti = [[0 for _ in CLUSTERS] for _ in articles]
    for t in range(len(articles)):
        wti[t][t % len(CLUSTERS)] = 1.0
    return wti


def build_ntk(articles, vocabulary):
    ntk = [[0 for _ in vocabulary] for _ in articles]
    for t in range(len(articles)):
        word_counter = Counter(articles[t]['text'].split())
        for word in word_counter:
            ntk[t][vocabulary[word]] = word_counter[word]
    return ntk


def compute_wti(articles, vocabulary, ntk, alpha, zti):
    pass


def compute_alpha(articles, wti):
    pass


def compute_pik(articles, vocabulary, wti, ntk):
    pass


def compute_zti(articles, vocabulary, ntk, alpha, pik):
    return [[math.log(alpha[i]) + sum([math.log(pik[i][k]) * ntk[t][k] for k in range(len(vocabulary))])
             for i in range(len(CLUSTERS))] for t in range(len(articles))]


def run_em_initialization(articles, vocabulary):
    wti = build_wti(articles)
    ntk = build_ntk(articles, vocabulary)
    return wti, ntk


def run_e_phase(articles, vocabulary, ntk, alpha, pik):
    zti = compute_zti(articles, vocabulary, ntk, alpha, pik)
    wti = compute_wti(articles, vocabulary, ntk, alpha, zti)
    return wti


def run_m_phase(articles, vocabulary, wti, ntk):
    alpha = compute_alpha(articles, wti)
    pik = compute_pik(articles, vocabulary, wti, ntk)
    return alpha, pik


def run_em(articles, vocabulary):
    wti, ntk = run_em_initialization(articles, vocabulary)
    alpha, pik = run_m_phase(articles, vocabulary, wti, ntk)

    for iteration_number in range(ITERATION_COUNT):
        wti = run_e_phase(articles, vocabulary, ntk, alpha, pik)
        alpha, pik = run_m_phase(articles, vocabulary, wti, ntk)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('develop_file_path', help='The path to the develop.txt file')
    return parser.parse_args()


def main(args):
    vocabulary = get_vocabulary(args.develop_file_path)
    articles = get_articles(args.develop_file_path, vocabulary)
    run_em(articles, vocabulary)


if __name__ == "__main__":
    main(get_arguments())
