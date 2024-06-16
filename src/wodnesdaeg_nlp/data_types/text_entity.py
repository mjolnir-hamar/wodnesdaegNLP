from pydantic import BaseModel
from typing import List

from src.wodnesdaeg_nlp.data_types.tag import NerTag


class TextEntity(BaseModel):

    text: str


class Lemma(TextEntity):

    ner_tag: NerTag


class Token(TextEntity):

    ner_tag: NerTag
    lemma: Lemma


class Sentence(BaseModel):

    words: List[Token]
