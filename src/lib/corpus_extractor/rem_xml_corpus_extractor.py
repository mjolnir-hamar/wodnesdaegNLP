import re
import glob
import logging
from tqdm import tqdm
from lxml import etree
from typing import (
    Any,
    List,
    Dict
)

from src.lib.data_types import (
    Token,
    Lemma,
    NerTag,
    Sentence,
    Corpus
)
import src.lib.consts.args as args_consts
import src.lib.consts.file_types as file_consts
from src.lib.corpus_extractor import CorpusExtractor

"""
Based on https://github.com/cltk/middle_high_german_texts/blob/master/reader.py
"""

logger = logging.getLogger(__name__)

TOKEN_IDXS: str = "token_indices"
COLUMN_IDXS: str = "column_indices"
START: str = "start"
END: str = "end"

TOKEN: str = "token"
TOKENS: str = f"{TOKEN}s"
LINE: str = "line"
LINES: str = f"{LINE}s"
COLUMN: str = "column"
COLUMNS: str = f"{COLUMN}s"
TAG: str = "tag"
TOK_ANNO: str = "tok_anno"
ID: str = "id"
RANGE: str = "range"

NORM: str = "norm"
POS: str = "pos"
LEMMA: str = "lemma"
TOK_FEATS: List[str] = [NORM, POS, LEMMA]

LINE_RANGE_REGEX: re.Pattern = re.compile(r"^t(\d+)_d\d+..t(\d+)_d\d+$")
COLUMN_RANGE_REGEX: re.Pattern = re.compile(r"^l(\d+)..l(\d+)$")


class RemXmlCorpusExtractor(CorpusExtractor):

    @staticmethod
    def extract_columns(xml_file: str) -> Dict:
        parser: etree.XMLParser = etree.XMLParser(load_dtd=True, no_network=False)
        tree: etree.ElementTree() = etree.parse(xml_file, parser=parser)
        root = tree.getroot()

        tokens: Dict[int, List[Dict]] = {
            int(token.get(ID)[1:]): [
                {child.tag: child.get(TAG) for child in annot.getchildren()}
                for annot in token.findall(f".//{TOK_ANNO}")
            ] for token in root.findall(f".//{TOKEN}")
        }

        lines: Dict = {}
        for line in root.findall(f".//{LINE}"):
            line_range_match: re.Match[str] = LINE_RANGE_REGEX.match(line.get(RANGE))
            if line_range_match is None:
                continue
            start_idx: int = int(line_range_match.group(1))
            end_idx: int = int(line_range_match.group(2))
            line_toks: List[Dict[str, str]] = [
                {
                    tok_feat: sub_tok[tok_feat] for tok_feat in TOK_FEATS if tok_feat in sub_tok.keys()
                } for idx in range(start_idx, end_idx + 1) for sub_tok in tokens[idx]
            ]

            lines[int(line.get(ID)[1:])] = {
                TOKEN_IDXS: {
                    START: start_idx,
                    END: end_idx
                },
                TOKENS: line_toks
            }

        columns: Dict = {}
        for i, column in enumerate(root.findall(f".//{COLUMN}")):
            column_range_match: re.Match[str] = COLUMN_RANGE_REGEX.match(column.get(RANGE))
            if column_range_match is not None:
                start_idx: int = int(column_range_match.group(1))
                end_idx: int = int(column_range_match.group(2))
                column_lines: List[str] = [
                    lines[idx] for idx in range(start_idx, end_idx + 1) if idx in lines.keys()
                ]

                columns[int(column.get(ID)[1:])] = {
                    COLUMN_IDXS: {
                        START: start_idx,
                        END: end_idx
                    },
                    LINES: column_lines
                }

        return columns

    @staticmethod
    def columns_to_corpus(columns: Dict, keep_case_markings: bool = True) -> List[Sentence]:
        sentences: List[Sentence] = []
        for col_idx, col_data in sorted(columns.items(), key=lambda kv: kv[0]):
            for line in col_data[LINES]:
                tokens: List[Token] = []
                is_good_line: bool = True
                for tok in line[TOKENS]:
                    if NORM in tok.keys():
                        if "--" in tok[NORM]:
                            is_good_line = False
                            break
                        tokens.append(
                            Token(
                                text=tok[NORM],
                                lemma=Lemma(
                                    text=tok[LEMMA],
                                    ner_tag=NerTag(tag=tok[POS])
                                ),
                                ner_tag=NerTag(tag=tok[POS])
                            )
                        )
                if is_good_line:
                    sentences.append(
                        Sentence(words=tokens)
                    )

        return sentences

    def extract_corpora(self, **kwargs) -> List[Corpus]:
        corpora: List[Corpus] = []
        corpus_dir: str = kwargs[args_consts.CORPUS_DIR]
        sentence_extraction_kwargs: Dict[str, Any] = {
            kwarg: val for kwarg, val in kwargs.items() if kwarg in args_consts.CORPUS_EXTRACTOR_KWARGS
        }
        for xml_file in tqdm(
                glob.glob(f"{corpus_dir}/*.{file_consts.XML}"),
                desc=f"Loading corpora from {corpus_dir}"
        ):

            columns = self.extract_columns(xml_file)
            sentences = self.columns_to_corpus(columns, **sentence_extraction_kwargs)
            corpora.append(
                Corpus(
                    name=xml_file.strip("/").split("/")[-1],
                    sentences=sentences
                )
            )
        logging.info(f"Extracted {len(corpora)} corpora from {corpus_dir}")

        return corpora
