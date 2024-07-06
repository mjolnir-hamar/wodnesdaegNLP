import os
import glob
import json
import shutil
import logging
from tqdm import tqdm
from typing import (
    List,
    Dict
)

from wodnesdaeg_nlp.data_types import (
    Token,
    Lemma,
    NerTag,
    Sentence,
    Corpus
)


logger = logging.getLogger(__name__)


class CorpusExtractor:

    OUTDIR_SUFFIX: str = ".extracted_corpora"

    def extract_corpora(self, **kwargs):
        raise NotImplementedError(
            "Base \"CorpusExtractor\" cannot extract corpora, itself; use a dataset specific child class"
        )

    @staticmethod
    def load_precomputed_corpora(precomputed_corpus_dir: str) -> List[Corpus]:

        corpora: List[Corpus] = []
        for file in tqdm(
                glob.glob(f"{precomputed_corpus_dir}/*.tsv"),
                desc=f"Loading precomputed corpora from {precomputed_corpus_dir}"
        ):
            fname = file.split("/")[-1].replace(".tsv", "")
            if "." in fname:
                text_name, year = fname.split(".")
            else:
                text_name = fname
                year = ""
            sentences: List[Sentence] = []
            with open(file, "r") as _f:
                for line in _f:
                    line = line.strip().split("\t")
                    try:
                        tokens: List[str] = line[0].split(" ")
                        ner_tags: List[str] = line[1].split(" ")
                        lemmas: List[str] = line[2].split(" ")
                    except IndexError:
                        continue

                    sentences.append(
                        Sentence(words=[
                            Token(
                                text=token,
                                lemma=Lemma(
                                    text=lemmas[i],
                                    ner_tag=NerTag(
                                        tag=ner_tags[i]
                                    )
                                ),
                                ner_tag=NerTag(tag=ner_tags[i])
                            ) for i, token in enumerate(tokens)
                        ])
                    )
            corpora.append(
                Corpus(
                    name=text_name,
                    year=year,
                    sentences=sentences
                )
            )

        return corpora

    @staticmethod
    def consolidate_ner_tags(
            corpora: List[Corpus],
            ner_tag_config_file: str,
            fail_on_missing_tag_mapping: bool = False
    ) -> List[Corpus]:

        with open(ner_tag_config_file, "r") as _j:
            ner_tag_config: [Dict[str, str]] = json.load(_j)

        for corpus in tqdm(corpora, desc=f"Consolidating NER tags using {ner_tag_config_file}"):
            for sentence in corpus.sentences:
                for word in sentence.words:
                    orig_ner_tag: str = word.ner_tag.tag
                    try:
                        word.ner_tag.tag = ner_tag_config[orig_ner_tag]
                    except KeyError:
                        if not fail_on_missing_tag_mapping:
                            logger.warning(f"NER tag {orig_ner_tag} not found in {ner_tag_config_file}")
                        else:
                            raise KeyError(f"NER tag {orig_ner_tag} not found in {ner_tag_config_file}")

        return corpora

    def write_corpora_to_file(self, corpora: List[Corpus], outdir: str):

        outdir = f"{outdir}{self.OUTDIR_SUFFIX}"
        if os.path.isdir(outdir):
            shutil.rmtree(outdir)
        os.mkdir(outdir)

        for corpus in tqdm(corpora, desc=f"Writing corpora to {outdir}"):
            with open(f"{outdir}/{corpus.full_name}.tsv", "w") as _o:
                for sentence in corpus.sentences:
                    tokens: List[str] = []
                    ner_tags: List[str] = []
                    lemmas: List[str] = []
                    for word in sentence.words:
                        tokens.append(word.text)
                        ner_tags.append(word.ner_tag.tag)
                        lemmas.append(word.lemma.text)
                    _o.write(
                        f"{' '.join(tokens)}\t{' '.join(ner_tags)}\t{' '.join(lemmas)}\n"
                    )
