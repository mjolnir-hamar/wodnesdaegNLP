import glob
import argparse
from random import sample
from typing import List, Dict
from collections import Counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--corpus-dir", dest="corpus_dir", required=True)
    parser.add_argument("-t", "--tag", required=True)
    parser.add_argument("-n", "--number-of-examples", dest="num_examples", default=5, type=int)
    parser.add_argument("--randomize-results", dest="randomize", action="store_true")
    args = parser.parse_args()

    # Place to put gather words
    tag_words: List[str] = []
    # Iterate over corpus files
    for file in glob.glob(f"{args.corpus_dir}/*.tsv"):
        with open(file, "r") as _f:
            # Iterate over lines
            for line in _f:
                # Split the line into its 3 components
                try:
                    text, pos_tags, lemmas = line.strip().split("\t")
                except ValueError:
                    continue
                # Split the tag string
                split_tags: List[str] = pos_tags.split(" ")

                # Place to put the current word
                curr_word: List[str] = []
                # Iterate over tokens in the text string
                for i, token in enumerate(text.split(" ")):
                    # Retrieve the tag for that token
                    tok_tag: str = split_tags[i]
                    # If the tag is not the one we're looking for, but we have tokens in curr_word
                    # (i.e. in a past iteration, the tag was the one we're looking for), add the word
                    # to tag_words, and reset curr_word
                    if tok_tag != args.tag and len(curr_word) > 0:
                        tag_words.append(" ".join(curr_word).lower())
                        curr_word = []

                    # If the tag is the one we're looking for, add the token to curr_word
                    elif tok_tag == args.tag:
                        curr_word.append(token)

                # Check if we have a curr_word after the last iteration
                if len(curr_word) > 0:
                    tag_words.append(" ".join(curr_word).lower())

    tag_words_counted: Dict[str, int] = Counter(tag_words)

    k: int = len(tag_words_counted.keys()) if args.num_examples > len(tag_words) else args.num_examples

    if args.randomize:
        for word in sample(list(tag_words_counted.keys()), k=k):
            print(f"{word}: {tag_words_counted[word]}")

    else:
        num_shown: int = 0
        for word, count in sorted(tag_words_counted.items(), key=lambda kv: kv[1], reverse=True):
            if num_shown == k:
                break
            print(f"{word}: {count}")
            num_shown += 1


if __name__ == "__main__":
    main()
