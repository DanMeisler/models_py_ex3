from collections import Counter, defaultdict
import argparse
import logging
import math
import re

SET_FILE_HEADER_LINE_REGEX = r"^<\w+\s+\d+\s*(.*)>$"
WORD_IS_COUNT_THRESHOLD = 4  # filter words that appear less than 4 times in the whole corpus
LIDSTONE_LAMBDA = 0.5  # lidstone smoothing for m step
ITERATION_COUNT = 30
CLUSTERS = range(9)
EPSILON = 0.0001  # alphas cannot be zeros
K = 10.0  # precision parameter for underflow handling

"""
Dictionary:
wti - the weights
ntk - word frequency
alpha - P(cluster i)
pik - pik[i][k] == P(word k | cluster i)
zti - log of the numerator of the resulted wti (wti[t][i])
t - iterates over the articles
k - iterates over the vocabulary
i - iterates over the clusters
"""


def get_vocabulary(articles_file_path):
    words_counter = Counter()
    with open(articles_file_path) as articles_file:
        for line in articles_file:
            line = line.strip()
            if not re.match(SET_FILE_HEADER_LINE_REGEX, line):
                words_counter.update(line.split())

    vocabulary = list(filter(lambda x: words_counter[x] >= WORD_IS_COUNT_THRESHOLD, words_counter))
    word_count = sum([x for x in words_counter.values() if x >= WORD_IS_COUNT_THRESHOLD])
    return vocabulary, word_count


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


def build_k_t_map(articles, vocabulary):
    k_t_map = defaultdict(list)
    for k, word in enumerate(vocabulary):
        k_t_map[k] = list(filter(lambda x: word in articles[x]["text"], range(len(articles))))
    return k_t_map


def build_t_k_map(articles, vocabulary):
    t_k_map = defaultdict(list)
    for t, article in enumerate(articles):
        t_k_map[t] = [vocabulary.index(word) for word in set(article["text"].split())]
    return t_k_map


def build_wti(articles):
    wti = defaultdict(lambda: defaultdict(lambda: 0))
    for t in range(len(articles)):
        wti[t][t % len(CLUSTERS)] = 1.0
    return wti


def build_ntk(articles, vocabulary):
    ntk = defaultdict(lambda: defaultdict(lambda: 0))
    for t in range(len(articles)):
        word_counter = Counter(articles[t]['text'].split())
        for word in word_counter:
            ntk[t][vocabulary.index(word)] = word_counter[word]
    return ntk


def compute_zti(articles, t_k_map, ntk, alpha, pik):
    zti = defaultdict(lambda: defaultdict(lambda: 0))
    for t in range(len(articles)):
        for i in range(len(CLUSTERS)):
            zti[t][i] = math.log(alpha[i]) + sum([math.log(pik[i][k]) * ntk[t][k] for k in t_k_map[t]])
    return zti


def compute_wti(articles, zti):
    wti = defaultdict(lambda: defaultdict(lambda: 0))
    for t in range(len(articles)):
        m = max(zti[t].values())
        for i in range(len(CLUSTERS)):
            if zti[t][i] - m >= -K:
                wti[t][i] = \
                    math.exp(zti[t][i] - m) / sum([math.exp(zti[t][j] - m) for j in CLUSTERS if zti[t][j] - m >= -K])
    return wti


def compute_alpha(articles, wti):
    alpha = defaultdict(lambda: 0)
    for i in range(len(CLUSTERS)):
        alpha[i] = sum([wti[t][i] for t in range(len(articles))]) / len(articles)
        if alpha[i] == 0:
            alpha[i] = EPSILON

    alpha_total = sum(alpha.values())
    for i in range(len(alpha)):
        alpha[i] /= alpha_total

    return alpha


def compute_pik(articles, vocabulary, k_t_map, wti, ntk):
    pik = defaultdict(lambda: defaultdict(lambda: 0))
    for i in range(len(CLUSTERS)):
        denominator = LIDSTONE_LAMBDA * len(vocabulary) + \
                      sum(wti[t][i] * len(articles[t]["text"].split()) for t in range(len(articles)))
        for k in range(len(vocabulary)):
            numerator = LIDSTONE_LAMBDA + sum(wti[t][i] * ntk[t][k] for t in k_t_map[k])
            pik[i][k] = numerator / denominator
    return pik


def compute_log_likelihood(articles, t_k_map, ntk, alpha, pik):
    log_likelihood = 0
    zti = compute_zti(articles, t_k_map, ntk, alpha, pik)
    for t in range(len(articles)):
        m = max(zti[t].values())
        sum_above_clusters = sum([math.exp(zti[t][i] - m) for i in range(len(CLUSTERS)) if zti[t][i] - m >= -K])
        log_likelihood += m + math.log(sum_above_clusters)
    return log_likelihood


def compute_perplexity(log_likelihood, word_count):
    return math.exp(-log_likelihood / word_count)


def run_em_initialization(articles, vocabulary):
    wti = build_wti(articles)
    ntk = build_ntk(articles, vocabulary)
    return wti, ntk


def run_e_phase(articles, t_k_map, ntk, alpha, pik):
    zti = compute_zti(articles, t_k_map, ntk, alpha, pik)
    wti = compute_wti(articles, zti)
    return wti


def run_m_phase(articles, vocabulary, k_t_map, wti, ntk):
    alpha = compute_alpha(articles, wti)
    pik = compute_pik(articles, vocabulary, k_t_map, wti, ntk)
    return alpha, pik


def run_em(articles, vocabulary, word_count, k_t_map, t_k_map):
    logging.debug("EM initialization start")
    wti, ntk = run_em_initialization(articles, vocabulary)
    alpha, pik = run_m_phase(articles, vocabulary, k_t_map, wti, ntk)

    for iteration_number in range(ITERATION_COUNT):
        logging.debug("Iteration %d", iteration_number)
        logging.debug("E start")
        wti = run_e_phase(articles, t_k_map, ntk, alpha, pik)
        logging.debug("M start")
        alpha, pik = run_m_phase(articles, vocabulary, k_t_map, wti, ntk)
        log_likelihood = compute_log_likelihood(articles, t_k_map, ntk, alpha, pik)
        logging.debug("Log likelihood = %f", log_likelihood)
        perplexity = compute_perplexity(log_likelihood, word_count)
        logging.debug("Perplexity = %f", perplexity)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('develop_file_path', help='The path to the develop.txt file')
    return parser.parse_args()


def main(args):
    logging.debug("Start")
    vocabulary, word_count = get_vocabulary(args.develop_file_path)
    logging.debug("word count = %d", word_count)
    articles = get_articles(args.develop_file_path, vocabulary)
    k_t_map = build_k_t_map(articles, vocabulary)
    t_k_map = build_t_k_map(articles, vocabulary)
    run_em(articles, vocabulary, word_count, k_t_map, t_k_map)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main(get_arguments())
