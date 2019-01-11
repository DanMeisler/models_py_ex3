from collections import Counter, defaultdict
import argparse
import logging
import math
import re

TOPICS = ['acq', 'money-fx', 'grain', 'crude', 'trade', 'interest', 'ship', 'wheat', 'corn']
SET_FILE_HEADER_LINE_REGEX = r"^<\w+\s+\d+\s*(.*)>$"
WORD_IS_COUNT_THRESHOLD = 4  # filter words that appear less than 4 times in the whole corpus
STOP_EM_LOG_LIKELIHOOD_THRESHOLD = 10
LIDSTONE_LAMBDA = 0.5  # lidstone smoothing for m step
MAX_ITERATION_COUNT = 35
CLUSTERS = [{} for i in range(9)]
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
    """
    Create the vocabulary from develop.txt file with filtering by word threshold
    :param articles_file_path: path to article.txt file
    :return: Vocabulary in the form of [word_1, word_2, ..., word_k, ..., word_|V|]
    """
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
    """
    Create the articles list
    :param articles_file_path: path to article.txt file
    :param vocabulary: a vocabulary (See get_vocabulary's doc for more info)
    :return: articles in the form of [{"topics": [topic_1, topic_2, ...], "text": article_text}]
    """
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
    """
    Create map for speed up EM algorithm
    :param articles: articles (See get_articles's doc for more info)
    :param vocabulary: a vocabulary (See get_vocabulary's doc for more info)
    :return: a map in the form of {k:[t, t, ...], k:[t, t, ...], ...}
    """
    k_t_map = defaultdict(list)
    for k, word in enumerate(vocabulary):
        k_t_map[k] = list(filter(lambda x: word in articles[x]["text"], range(len(articles))))
    return k_t_map


def build_t_k_map(articles, vocabulary):
    """
    Create map for speed up EM algorithm
    :param articles: articles (See get_articles's doc for more info)
    :param vocabulary: a vocabulary (See get_vocabulary's doc for more info)
    :return: a map in the form of {t:[k, k, ...], t:[k, k, ...], ...}
    """
    t_k_map = defaultdict(list)
    for t, article in enumerate(articles):
        t_k_map[t] = [vocabulary.index(word) for word in set(article["text"].split())]
    return t_k_map


def build_wti(articles):
    """
    (See dictionary above for more info)
    :param articles: articles (See get_articles's doc for more info)
    :return: initialized wti
    """
    wti = defaultdict(lambda: defaultdict(lambda: 0))
    for t in range(len(articles)):
        wti[t][t % len(CLUSTERS)] = 1.0
    return wti


def build_ntk(articles, vocabulary):
    """
    (See dictionary above for more info)
    :param articles: articles (See get_articles's doc for more info)
    :param vocabulary: a vocabulary (See get_vocabulary's doc for more info)
    :return: ntk
    """
    ntk = defaultdict(lambda: defaultdict(lambda: 0))
    for t in range(len(articles)):
        word_counter = Counter(articles[t]['text'].split())
        for word in word_counter:
            ntk[t][vocabulary.index(word)] = word_counter[word]
    return ntk


def build_clusters(articles, wti):
    """
    Each cluster will be in the form of {["topics":Counter({topic_1: topic_1_count, ...}), "article_indexes": set()]}
    :param articles: articles (See get_articles's doc for more info)
    :param wti: (See dictionary above for more info)
    """
    for cluster in CLUSTERS:
        cluster["topics"] = Counter()
        cluster["articles_indexes"] = set()

    for t, article in enumerate(articles):
        i = max(wti[t], key=wti[t].get)
        CLUSTERS[i]["topics"].update(article["topics"])
        CLUSTERS[i]["articles_indexes"].add(t)


def build_confusion_matrix():
    """
    Build confusion matrix from CLUSTERS global variable
    :return: matrix in the form of [[], [], ...]
    """
    sorted_clusters = sorted(CLUSTERS, key=lambda x: len(x["articles_indexes"]), reverse=True)
    matrix_column_count = len(TOPICS) + 1
    matrix_row_count = len(sorted_clusters)
    matrix = [[0 for _ in range(matrix_column_count)] for _ in range(matrix_row_count)]

    for i, cluster in enumerate(sorted_clusters):
        for j, topic in enumerate(TOPICS):
            matrix[i][j] = cluster["topics"][topic]
        matrix[i][-1] = len(cluster["articles_indexes"])

    return matrix


def compute_accuracy(articles):
    """
    :param articles: articles (See get_articles's doc for more info)
    :return: The accuracy of the model
    """
    correct_assignment_count = 0
    for cluster in CLUSTERS:
        cluster_label = cluster["topics"].most_common(1)[0][0]
        for t in cluster["articles_indexes"]:
            if cluster_label in articles[t]["topics"]:
                correct_assignment_count += 1

    return float(correct_assignment_count) / len(articles)


def compute_zti(articles, t_k_map, ntk, alpha, pik):
    """
    :param articles: articles (See get_articles's doc for more info)
    :param t_k_map: (See build_t_k_map's doc for more info)
    :param ntk: ntk (See dictionary above for more info)
    :param alpha: alphas (See dictionary above for more info)
    :param pik: pik (See dictionary above for more info)
    :return: zti (See dictionary above for more info)
    """
    zti = defaultdict(lambda: defaultdict(lambda: 0))
    for t in range(len(articles)):
        for i in range(len(CLUSTERS)):
            zti[t][i] = math.log(alpha[i]) + sum([math.log(pik[i][k]) * ntk[t][k] for k in t_k_map[t]])
    return zti


def compute_wti(articles, zti):
    """
    :param articles: articles (See get_articles's doc for more info)
    :param zti: zti (See dictionary above for more info)
    :return: wti (See dictionary above for more info)
    """
    wti = defaultdict(lambda: defaultdict(lambda: 0))
    for t in range(len(articles)):
        m = max(zti[t].values())
        for i in range(len(CLUSTERS)):
            if zti[t][i] - m >= -K:
                wti[t][i] = math.exp(zti[t][i] - m) / sum([math.exp(zti[t][j] - m)
                                                           for j in range(len(CLUSTERS)) if zti[t][j] - m >= -K])
    return wti


def compute_alpha(articles, wti):
    """
    :param articles: articles (See get_articles's doc for more info)
    :param wti: wti (See dictionary above for more info)
    :return: alphas (See dictionary above for more info)
    """
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
    """
    :param articles: articles (See get_articles's doc for more info)
    :param vocabulary: a vocabulary (See get_vocabulary's doc for more info)
    :param k_t_map: (See build_k_t_map's doc for more info)
    :param wti: wti (See dictionary above for more info)
    :param ntk: ntk (See dictionary above for more info)
    :return: pik (See dictionary above for more info)
    """
    pik = defaultdict(lambda: defaultdict(lambda: 0))
    for i in range(len(CLUSTERS)):
        denominator = LIDSTONE_LAMBDA * len(vocabulary) + \
                      sum(wti[t][i] * len(articles[t]["text"].split()) for t in range(len(articles)))
        for k in range(len(vocabulary)):
            numerator = LIDSTONE_LAMBDA + sum(wti[t][i] * ntk[t][k] for t in k_t_map[k])
            pik[i][k] = numerator / denominator
    return pik


def compute_log_likelihood(articles, t_k_map, ntk, alpha, pik):
    """
    :param articles: articles (See get_articles's doc for more info)
    :param t_k_map: (See build_t_k_map's doc for more info)
    :param ntk: ntk (See dictionary above for more info)
    :param alpha: alphas (See dictionary above for more info)
    :param pik: pik (See dictionary above for more info)
    :return: the log likelihood
    """
    log_likelihood = 0
    zti = compute_zti(articles, t_k_map, ntk, alpha, pik)
    for t in range(len(articles)):
        m = max(zti[t].values())
        sum_above_clusters = sum([math.exp(zti[t][i] - m) for i in range(len(CLUSTERS)) if zti[t][i] - m >= -K])
        log_likelihood += m + math.log(sum_above_clusters)
    return log_likelihood


def compute_perplexity(log_likelihood, word_count):
    """
    :param log_likelihood: (See compute_log_likelihood's doc for more info)
    :param word_count: total words count include duplicates(above threshold of the vocabulary)
    :return: the perplexity
    """
    return math.exp(-log_likelihood / word_count)


def run_em_initialization(articles, vocabulary):
    """
    Runs the expectation maximization initialization
    """
    wti = build_wti(articles)
    ntk = build_ntk(articles, vocabulary)
    return wti, ntk


def run_e_phase(articles, t_k_map, ntk, alpha, pik):
    """
    Runs the expectation phase
    """
    zti = compute_zti(articles, t_k_map, ntk, alpha, pik)
    wti = compute_wti(articles, zti)
    return wti


def run_m_phase(articles, vocabulary, k_t_map, wti, ntk):
    """
    Runs the maximization algorithm
    """
    alpha = compute_alpha(articles, wti)
    pik = compute_pik(articles, vocabulary, k_t_map, wti, ntk)
    return alpha, pik


def run_em(articles, vocabulary, word_count):
    """
    Runs the expectation maximization algorithm
    """
    k_t_map = build_k_t_map(articles, vocabulary)
    t_k_map = build_t_k_map(articles, vocabulary)
    logging.debug("EM start")
    wti, ntk = run_em_initialization(articles, vocabulary)
    alpha, pik = run_m_phase(articles, vocabulary, k_t_map, wti, ntk)
    log_likelihoods = [compute_log_likelihood(articles, t_k_map, ntk, alpha, pik)]
    perplexities = [compute_perplexity(log_likelihoods[-1], word_count)]
    for iteration_number in range(MAX_ITERATION_COUNT):
        logging.debug("%d - expectation start", iteration_number)
        wti = run_e_phase(articles, t_k_map, ntk, alpha, pik)
        logging.debug("%d - maximization start", iteration_number)
        alpha, pik = run_m_phase(articles, vocabulary, k_t_map, wti, ntk)
        log_likelihoods.append(compute_log_likelihood(articles, t_k_map, ntk, alpha, pik))
        logging.debug("%d - log likelihood = %f", iteration_number, log_likelihoods[-1])
        perplexities.append(compute_perplexity(log_likelihoods[-1], word_count))
        logging.debug("%d - perplexity = %f", iteration_number, perplexities[-1])
        if log_likelihoods[-2] > log_likelihoods[-1] - STOP_EM_LOG_LIKELIHOOD_THRESHOLD:
            break

    build_clusters(articles, wti)
    logging.debug("EM done")


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('develop_file_path', help='The path to the develop.txt file')
    return parser.parse_args()


def main(args):
    logging.debug("Start")
    vocabulary, word_count = get_vocabulary(args.develop_file_path)
    articles = get_articles(args.develop_file_path, vocabulary)
    run_em(articles, vocabulary, word_count)
    logging.info("Confusion matrix = %s", str(build_confusion_matrix()))
    logging.info("Accuracy = %f", compute_accuracy(articles))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main(get_arguments())
