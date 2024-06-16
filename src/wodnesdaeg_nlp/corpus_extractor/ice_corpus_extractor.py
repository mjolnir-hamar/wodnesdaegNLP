import re
import glob
import logging
from tqdm import tqdm
from copy import deepcopy
from typing import (
    Any,
    List,
    Dict
)

from src.wodnesdaeg_nlp.data_types import (
    Token,
    Lemma,
    NerTag,
    Sentence,
    Corpus
)
import src.wodnesdaeg_nlp.consts.args as args_consts
import src.wodnesdaeg_nlp.consts.file_types as file_consts
from src.wodnesdaeg_nlp.corpus_extractor import CorpusExtractor


logger = logging.getLogger(__name__)

TAGGED_TOK_REGEX: re.Pattern = re.compile(r"\(([A-Z]+(?:-[A-Z]+)?)\s([\w$]+)-(\w+)\)")
SENT_SPLIT_REGEX: re.Pattern = re.compile(r"\([.;]\s[.:;]-[.:;]\)")

HEADING_END: str = "</heading>"


class IceCorpusExtractor(CorpusExtractor):

    @staticmethod
    def extract_sentences(grammar_file: str, keep_case_markings: bool = True) -> List[Sentence]:
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
                            if not keep_case_markings:
                                ner_tag = tag_match[0].split("-")[0]
                            else:
                                ner_tag = tag_match[0]
                            sentence_builder.append(
                                Token(
                                    text=tag_match[1],
                                    lemma=Lemma(
                                        text=tag_match[2],
                                        ner_tag=NerTag(
                                            tag=ner_tag
                                        )
                                    ),
                                    ner_tag=NerTag(tag=ner_tag)
                                )
                            )
                        if i != 0 and i != len(sub_lines) - 1:
                            sentences.append(
                                Sentence(words=sentence_builder)
                            )
                            sentence_builder = []

        return sentences

    def extract_corpora(self, **kwargs) -> List[Corpus]:
        corpora: List[Corpus] = []
        corpus_dir: str = kwargs[args_consts.CORPUS_DIR]
        sentence_extraction_kwargs: Dict[str, Any] = {
            kwarg: val for kwarg, val in kwargs.items() if kwarg in args_consts.CORPUS_EXTRACTOR_KWARGS
        }
        for grammar_file in tqdm(
                glob.glob(f"{corpus_dir}/*.{file_consts.PSD}"),
                desc=f"Loading corpora from {corpus_dir}"
        ):
            year, text_name, _, _ = grammar_file.split("/")[-1].split(".")
            corpora.append(
                Corpus(
                    name=text_name,
                    year=year,
                    sentences=self.extract_sentences(
                        grammar_file=grammar_file,
                        **sentence_extraction_kwargs
                    )
                )
            )
        logging.info(f"Extracted {len(corpora)} corpora from {corpus_dir}")

        return corpora
