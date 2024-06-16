import re
import glob
import logging
from tqdm import tqdm
from lxml import etree
from typing import List

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

TREEBANK_FILE_NAME_REGEX = re.compile(r"\d{4}\.\d{2}\.\d{4}")

SENTENCE = "sentence"
WORD = "word"
FORM = "form"
LEMMA = "lemma"
POSTAG = "postag"

INVALID_POS_TAGS = {"-", "u"}


class LatinTreebankPerseusCorpusExtractor(CorpusExtractor):

    @staticmethod
    def extract_sentences(xml_file: str):
        parser: etree.XMLParser = etree.XMLParser(load_dtd=True, no_network=False)
        tree: etree.ElementTree = etree.parse(xml_file, parser=parser)
        root = tree.getroot()
        sentences = root.findall(SENTENCE)

        corpus_sentences: List[Sentence] = []
        for sentence in sentences:
            words = sentence.findall(WORD)
            tokens: List[Token] = []
            for word in words:
                token = word.get(FORM)
                lemma = word.get(LEMMA)
                pos_tag = word.get(POSTAG)[0]
                if pos_tag not in INVALID_POS_TAGS:
                    tokens.append(
                        Token(
                            text=token,
                            lemma=Lemma(
                                text=lemma,
                                ner_tag=NerTag(tag=pos_tag)
                            ),
                            ner_tag=NerTag(tag=pos_tag)
                        )
                    )
            corpus_sentences.append(Sentence(words=tokens))

        return corpus_sentences

    def extract_corpora(self, **kwargs) -> List[Corpus]:
        corpora: List[Corpus] = []
        corpus_dir: str = kwargs[args_consts.CORPUS_DIR]
        for xml_file in tqdm(
                glob.glob(f"{corpus_dir}/*.{file_consts.XML}"),
                desc=f"Loading corpora from {corpus_dir}"
        ):
            fname = xml_file.split("/")[-1].replace(".xml", "")
            if TREEBANK_FILE_NAME_REGEX.match(fname):
                corpora.append(
                    Corpus(
                        name=fname,
                        sentences=self.extract_sentences(xml_file)
                    )
                )
        logging.info(f"Extracted {len(corpora)} corpora from {corpus_dir}")

        return corpora
