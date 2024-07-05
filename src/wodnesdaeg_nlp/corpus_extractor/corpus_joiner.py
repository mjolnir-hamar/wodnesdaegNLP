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
            sampling_target: int = None,
            sampling_portion_split: Dict[str, float] = {}
    ) -> List[Corpus]:

        if sampling_target is not None and (downsample_to_smallest or upsample_to_largest):
            raise ValueError(
                "Cannot set \"sampling_target\" with either or both of \"downsample_to_smallest\" "
                "and \"upsample_to_largest\""
            )
        elif sampling_portion_split != {} and (downsample_to_smallest or upsample_to_largest):
            raise ValueError(
                "Cannot set \"sampling_portion_split\" with either or both of \"downsample_to_smallest\" "
                "and \"upsample_to_largest\""
            )
        elif downsample_to_smallest and upsample_to_largest:
            raise ValueError(
                "Cannot set both \"downsample_to_smallest\" and \"upsample_to_largest\""
            )
        elif sampling_portion_split != {}:
            if sum(sampling_portion_split.values()) < 1.0:
                raise ValueError(
                    f"\"sampling_portion_split\" sizes must equal 1.0: {sampling_portion_split}"
                )
            elif corpora_to_join.keys() != sampling_portion_split.keys():
                raise ValueError(
                    f"Keys in \"corpora_to_join\" ({corpora_to_join.keys()}) must be the same as keys in "
                    f"sampling_portion_split\" ({sampling_portion_split.keys()})"
                )

        if downsample_to_smallest:
            joined_corpus_sampling_target = min(
                [sum([len(corpus.sentences) for corpus in corpora_list]) for corpora_list in corpora_to_join.values()]
            )
        elif upsample_to_largest:
            joined_corpus_sampling_target = max(
                [sum([len(corpus.sentences) for corpus in corpora_list]) for corpora_list in corpora_to_join.values()]
            )
        else:
            joined_corpus_sampling_target = sampling_target

        joined_corpora: List[Corpus] = []
        for corpus_name, corpora_list in corpora_to_join.items():
            if sampling_portion_split != {}:
                joined_corpus_sampling_target = int(sampling_target * sampling_portion_split[corpus_name])
            joined_corpora.append(
                self.flatten_corpora(
                    corpora_to_flatten=corpora_list,
                    flattened_corpus_name=corpus_name,
                    sampling_target=joined_corpus_sampling_target
                )
            )

        return joined_corpora
