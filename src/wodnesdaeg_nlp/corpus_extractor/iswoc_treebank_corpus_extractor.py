import logging
from lxml import etree
from typing import (
    Dict,
    List,
    Set
)

from wodnesdaeg_nlp.data_types import (
    Token,
    Lemma,
    NerTag,
    Sentence,
    Corpus
)
import wodnesdaeg_nlp.consts.args as args_consts
import wodnesdaeg_nlp.consts.languages as lang_consts
import wodnesdaeg_nlp.consts.file_types as file_consts
from wodnesdaeg_nlp.corpus_extractor import CorpusExtractor

"""
Used to extract data from https://github.com/iswoc/iswoc-treebank/
"""

logger = logging.getLogger(__name__)


class ISWOCTreebankCorpusExtractor(CorpusExtractor):

    LANG_TO_FILE_MAP: Dict[str, Set[str]] = {
        lang_consts.ANG: {
            "æls",
            "apt",
            "chrona",
            "or",
            "wscp"
        }
    }

    SOURCE: str = "source"
    TITLE: str = "title"
    DIV: str = "div"
    SENTENCE: str = "sentence"
    FORM: str = "form"
    LEMMA: str = "lemma"
    TOKEN: str = "token"
    PART_OF_SPEECH: str = "part-of-speech"

    def extract_sentences(self, xml_file: str) -> List[Corpus]:
        parser: etree.XMLParser = etree.XMLParser(load_dtd=True, no_network=False)
        tree: etree.ElementTree = etree.parse(xml_file, parser=parser)
        root = tree.getroot()
        texts = root.findall(self.SOURCE)
        corpora: List[Corpus] = []
        for text in texts:
            title: str = text.findall(self.TITLE)[0].text
            sentences: List[Sentence] = []
            for subtext in text.findall(self.DIV):
                for sentence in subtext.findall(self.SENTENCE):
                    tokens: List[Token] = []
                    for token in sentence.findall(self.TOKEN):
                        pos_tag: str = token.get(self.PART_OF_SPEECH)
                        if pos_tag is not None:
                            tokens.append(
                                Token(
                                    text=token.get(self.FORM),
                                    lemma=Lemma(
                                        text=token.get(self.LEMMA),
                                        ner_tag=NerTag(tag=pos_tag)
                                    ),
                                    ner_tag=NerTag(tag=pos_tag)
                                )
                            )
                    sentences.append(Sentence(words=tokens))
            corpora.append(
                Corpus(
                    name=title,
                    sentences=sentences
                )
            )
        return corpora

    def extract_corpora(self, **kwargs) -> List[Corpus]:
        corpora: List[Corpus] = []
        corpus_dir: str = kwargs[args_consts.CORPUS_DIR]
        for xml_file in self.LANG_TO_FILE_MAP[kwargs[args_consts.LANGUAGE]]:
            corpora += self.extract_sentences(f"{corpus_dir}/{xml_file}.{file_consts.XML}")
        return corpora
