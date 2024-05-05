from pydantic import BaseModel
from src.lib.data_types.tag import NerTag
from typing import List


class TextEntity(BaseModel):

    text: str


class Lemma(TextEntity):

    ner_tag: NerTag


class Token(TextEntity):

    ner_tag: NerTag
    lemma: Lemma


class Sentence(BaseModel):

    words: List[Token]
