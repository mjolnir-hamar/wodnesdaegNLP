from typing import (
    List,
    Dict
)
from random import (
    choices,
    sample
)

from wodnesdaeg_nlp.data_types import (
    Sentence,
    Corpus
)
from wodnesdaeg_nlp.corpus_extractor import CorpusExtractor


class CorpusJoiner(CorpusExtractor):

    @staticmethod
    def flatten_corpora(
            corpora_to_flatten: List[Corpus],
            flattened_corpus_name: str,
            sampling_target: int = None
    ) -> Corpus:
        flattened_corpus_sentences: List[Sentence] = []
        for corpus_to_flatten in corpora_to_flatten:
            flattened_corpus_sentences += corpus_to_flatten.sentences

        if sampling_target is not None:
            if sampling_target > len(flattened_corpus_sentences):
                flattened_corpus_sentences = choices(flattened_corpus_sentences, k=sampling_target)
            elif sampling_target == len(flattened_corpus_sentences):
                pass
            else:
                flattened_corpus_sentences = sample(flattened_corpus_sentences, k=sampling_target)

        return Corpus(
            name=flattened_corpus_name,
            sentences=flattened_corpus_sentences
        )

    def join_corpora(
            self,
            corpora_to_join: Dict[str, List[Corpus]],
            downsample_to_smallest: bool = False,
            upsample_to_largest: bool = False,
            sampling_target: int = None
    ) -> List[Corpus]:

        if sampling_target is not None and (downsample_to_smallest or upsample_to_largest):
            raise ValueError(
                "Cannot set \"sampling_target\" with either or both of \"downsample_to_smallest\" "
                "and \"upsample_to_largest\""
            )
        elif downsample_to_smallest and upsample_to_largest:
            raise ValueError(
                "Cannot set both \"downsample_to_smallest\" and \"upsample_to_largest\""
            )

        if downsample_to_smallest:
            sampling_target = min(
                [sum([len(corpus.sentences) for corpus in corpora_list]) for corpora_list in corpora_to_join.values()]
            )
        elif upsample_to_largest:
            sampling_target = max(
                [sum([len(corpus.sentences) for corpus in corpora_list]) for corpora_list in corpora_to_join.values()]
            )

        return [
            self.flatten_corpora(
                corpora_to_flatten=corpora_list,
                flattened_corpus_name=corpus_name,
                sampling_target=sampling_target
            ) for corpus_name, corpora_list in corpora_to_join.items()
        ]
