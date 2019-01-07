from collections import Counter
import argparse
import re


SET_FILE_HEADER_LINE_REGEX = "^<\w+\s+\d+\s*(.*)>$"
WORD_IS_COUNT_THRESHOLD = 4


def get_vocabulary(articles_file_path):
    words_counter = Counter()
    with open(articles_file_path) as articles_file:
        for line in articles_file:
            line = line.strip()
            if not re.match(SET_FILE_HEADER_LINE_REGEX, line):
                words_counter.update(line.split())

    return list(filter(lambda x: words_counter[x] >= WORD_IS_COUNT_THRESHOLD, words_counter))


def get_articles(articles_file_path):
    vocabulary = get_vocabulary(articles_file_path)
    article_number = 1
    articles = {}
    with open(articles_file_path) as articles_file:
        for line in articles_file:
            line = line.strip()
            if line == "":
                continue
            header_line_match = re.match(SET_FILE_HEADER_LINE_REGEX, line)
            if header_line_match:
                articles[article_number] = {}
                articles[article_number]["topics"] = header_line_match.group(1).split()
            else:
                articles[article_number]["text"] = " ".join(filter(lambda x: x in vocabulary, line.split()))
                article_number += 1

    return articles


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('develop_file_path', help='The path to the develop.txt file')
    return parser.parse_args()


def main(args):
    print(get_articles(args.develop_file_path))


if __name__ == "__main__":
    main(get_arguments())
