from collections import Counter
import argparse
import logging
import math
import re


SET_FILE_HEADER_LINE_REGEX = r"^<\w+\s+\d+\s*(.*)>$"
WORD_IS_COUNT_THRESHOLD = 4
LIDSTONE_LAMBDA = 0.5
ITERATION_COUNT = 30
CLUSTERS = range(9)
EPSILON = 0.0001
K = 10.0


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


def compute_zti(articles, vocabulary, ntk, alpha, pik):
    return [[math.log(alpha[i]) + sum([math.log(pik[i][k]) * ntk[t][k] for k in range(len(vocabulary))])
             for i in range(len(CLUSTERS))] for t in range(len(articles))]


def compute_wti(articles, zti):
    wti = [[0 for _ in CLUSTERS] for _ in articles]
    for t in range(len(articles)):
        m = max(zti[t])
        for i in range(len(CLUSTERS)):
            if zti[t][i] - m >= -K:
                wti[t][i] = \
                    math.exp(zti[t][i] - m) / sum([math.exp(zti[t][j] - m) for j in CLUSTERS if zti[t][j] - m >= -K])
    return wti


def compute_alpha(articles, wti):
    alpha = [0 for _ in CLUSTERS]
    for i in range(len(CLUSTERS)):
        alpha[i] = sum([wti[t][i] for t in range(len(articles))]) / len(articles)
        if alpha[i] == 0:
            alpha[i] = EPSILON

    alpha = [alpha[i] / sum(alpha) for i in range(len(alpha))]
    return alpha


def compute_pik(articles, vocabulary, wti, ntk):
    pik = [[0 for _ in vocabulary] for _ in CLUSTERS]
    for i in range(len(CLUSTERS)):
        denominator = LIDSTONE_LAMBDA * len(vocabulary) + \
                      sum(wti[t][i] * len(articles[t]["text"].split()) for t in range(len(articles)))
        for k in range(len(vocabulary)):
            numerator = LIDSTONE_LAMBDA + sum(wti[t][i] * ntk[t][k] for t in range(len(articles)))
            pik[i][k] = numerator / denominator
    return pik


def compute_log_likelihood(articles, vocabulary, ntk, alpha, pik):
    log_likelihood = 0
    zti = compute_zti(articles, vocabulary, ntk, alpha, pik)
    for t in range(len(articles)):
        m = max(zti[t])
        sum_above_clusters = sum([math.exp(zti[t][i] - m) for i in range(len(CLUSTERS)) if zti[t][i] - m >= -K])
        log_likelihood += m + math.log(sum_above_clusters)
    return log_likelihood


def run_em_initialization(articles, vocabulary):
    wti = build_wti(articles)
    ntk = build_ntk(articles, vocabulary)
    return wti, ntk


def run_e_phase(articles, vocabulary, ntk, alpha, pik):
    zti = compute_zti(articles, vocabulary, ntk, alpha, pik)
    wti = compute_wti(articles, zti)
    return wti


def run_m_phase(articles, vocabulary, wti, ntk):
    alpha = compute_alpha(articles, wti)
    pik = compute_pik(articles, vocabulary, wti, ntk)
    return alpha, pik


def run_em(articles, vocabulary):
    logging.debug("EM start")
    wti, ntk = run_em_initialization(articles, vocabulary)
    alpha, pik = run_m_phase(articles, vocabulary, wti, ntk)

    for iteration_number in range(ITERATION_COUNT):
        logging.debug("Iteration %d", iteration_number)
        wti = run_e_phase(articles, vocabulary, ntk, alpha, pik)
        alpha, pik = run_m_phase(articles, vocabulary, wti, ntk)
        logging.debug("Log likelihood = %f", compute_log_likelihood(articles, vocabulary, ntk, alpha, pik))


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('develop_file_path', help='The path to the develop.txt file')
    return parser.parse_args()


def main(args):
    vocabulary = get_vocabulary(args.develop_file_path)
    articles = get_articles(args.develop_file_path, vocabulary)
    run_em(articles, vocabulary)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main(get_arguments())
