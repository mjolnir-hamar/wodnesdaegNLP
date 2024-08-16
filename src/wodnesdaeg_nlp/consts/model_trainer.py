import re


POS_TAGGING: str = "pos_tagging"
LEMMATIZATION: str = "lemmatization"
LEMMATIZATION_CAUSAL_LM: str = f"{LEMMATIZATION}_causal_lm"
NER: str = "ner"
TEST_GEN: str = "text-generation"
TEXT_2_TEXT_GEN: str = f"text2{TEST_GEN}"

TEXT: str = "text"
NER_TAGS: str = f"{NER}_tags"
TEXT_TAGS: str = "text_tags"
LEMMAS: str = "lemmas"

TRAIN: str = "train"
TEST: str = "test"
VAL: str = "val"

TOKENIZER: str = "tokenizer"
MAX_SEQ_LEN: str = "max_seq_len"

LEMMA_START_TAG = "<lemma>"
LEMMA_END_TAG = "</lemma>"
LEMMA_SPAN_REGEX = re.compile(rf"{LEMMA_START_TAG}[^<]+{LEMMA_END_TAG}")
