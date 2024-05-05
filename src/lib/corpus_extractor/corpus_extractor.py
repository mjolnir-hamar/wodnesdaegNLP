import os
import shutil
from tqdm import tqdm
from typing import List

from src.lib.data_types import Corpus

OUTDIR_SUFFIX = ".extracted_corpora"


class CorpusExtractor:

    def write_corpora_to_file(self, corpora: List[Corpus], outdir: str):

        outdir = f"{outdir}{OUTDIR_SUFFIX}"
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