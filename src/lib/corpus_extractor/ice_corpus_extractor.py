import re
import glob
from tqdm import tqdm
from copy import deepcopy
from typing import List

from src.lib.data_types import (
    Corpus,
    NerTag,
    Lemma,
    Token,
    Sentence
)
import src.lib.consts.args as args_consts
import src.lib.consts.file_types as file_consts
from src.lib.corpus_extractor import CorpusExtractor

TAGGED_TOK_REGEX: re.Pattern = re.compile(r"\(([A-Z]+(?:-[A-Z]+)?)\s([\w$]+)-(\w+)\)")
SENT_SPLIT_REGEX: re.Pattern = re.compile(r"\([.;]\s[.:;]-[.:;]\)")

HEADING_END: str = "</heading>"


class IceCorpusExtractor(CorpusExtractor):

    @staticmethod
    def extract_sentences(grammar_file: str) -> List[Sentence]:
        sentences: List[Sentence] = []
        sentence_builder: List[Token] = []
        with open(grammar_file, "r") as _f:
            for line in _f:
                line: str = line.strip()
                if line == "":
                    continue
                elif HEADING_END in line:
                    sentence_builder = []

                sentence_sep_matches: List = SENT_SPLIT_REGEX.findall(line)

                remaining_line: str = deepcopy(line)
                sub_lines: List[str] = []
                if len(sentence_sep_matches) > 0:
                    for sent_sep_match in sentence_sep_matches:
                        sub_line, remaining_line = remaining_line.split(sent_sep_match, 1)
                        sub_lines.append(sub_line)
                else:
                    sub_lines.append(line)

                if sub_lines == [""]:
                    sentences.append(
                        Sentence(words=sentence_builder)
                    )
                    sentence_builder = []
                else:
                    for i, sub_line in enumerate(sub_lines):
                        for tag_match in TAGGED_TOK_REGEX.findall(sub_line):
                            sentence_builder.append(
                                Token(
                                    text=tag_match[1],
                                    lemma=Lemma(
                                        text=tag_match[2],
                                        ner_tag=NerTag(
                                            tag=tag_match[0]
                                        )
                                    ),
                                    ner_tag=NerTag(tag=tag_match[0])
                                )
                            )
                        if i != 0 and i != len(sub_lines) - 1:
                            sentences.append(
                                Sentence(words=sentence_builder)
                            )
                            sentence_builder = []

        return sentences

    def __call__(self, *args, **kwargs) -> List[Corpus]:
        corpora: List[Corpus] = []
        for grammar_file in tqdm(
                glob.glob(f"{kwargs[args_consts.CORPUS_DIR]}/*.{file_consts.PSD}"),
                desc=f"Loading corpora from {kwargs[args_consts.CORPUS_DIR]}"
        ):
            year, text_name, _, _ = grammar_file.split("/")[-1].split(".")
            corpora.append(
                Corpus(
                    name=text_name,
                    year=year,
                    sentences=self.extract_sentences(grammar_file=grammar_file)
                )
            )
        return corpora
