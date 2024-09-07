import torch
import torch.nn.functional as F
from torch.nn import CosineSimilarity
from transformers import (
    PreTrainedTokenizerFast,
    AutoModel,
    AutoTokenizer
)
from typing import(
    Any,
    Tuple
)

from wodnesdaeg_nlp.util import TermColor


class SemanticSimilarityEvaluator:

    @staticmethod
    def color_format_similarity_calculation(similarity_score: float):
        """
        Color the semantic similarity calculation based on thresholds for printing to the terminal
        """
        if similarity_score > 0.7:
            return f"{TermColor.GREEN}{similarity_score}{TermColor.END}"
        elif similarity_score > 0.3:
            return f"{TermColor.YELLOW}{similarity_score}{TermColor.END}"
        elif similarity_score >= 0.0:
            return f"{TermColor.RED}{similarity_score}{TermColor.END}"
        else:
            return f"{TermColor.DIM}{similarity_score}{TermColor.END}"

    @staticmethod
    def load_pretrained_model_and_tokenizer(model_location: str) -> Tuple[PreTrainedTokenizerFast, Any]:
        """
        Load a model and its associated tokenizer using generic HF auto classes

        Unlike ModelTrainer and ModelPredictor, the auto class modeling task doesn't matter for semantic similarity
        computation. In fact, Hugging Face will complain that some weights aren't initialized. This isn't a problem
        because all we care about is the input (embedding) layer of the pretrained model, which should be intact.
        """
        return AutoTokenizer.from_pretrained(model_location), AutoModel.from_pretrained(model_location)


    @staticmethod
    def mean_pooling(model_output, attention_mask):
        """
        From sentence_transformer documentation here: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
        """
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_normalized_embedding_from_model_prediction(
            self, input_str: str, tokenizer: PreTrainedTokenizerFast, model: Any
    ) -> torch.Tensor:
        """
        Send an input string through the pretrained tokenizer and model, then extract the embeddings, pool, and
        normalize the result
        """
        encoded_input = tokenizer(input_str, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            model_output = model(**encoded_input)
        input_str_embeddings = self.mean_pooling(model_output, encoded_input["attention_mask"])
        return F.normalize(input_str_embeddings, p=2, dim=1)

    def compute_cosine_similarity(
            self,
            input_str_1: str,
            input_str_2: str,
            tokenizer: PreTrainedTokenizerFast,
            model: Any,
            print_result: bool = False
    ) -> float:
        """
        Computes cosine similarity between 2 strings using a pretrained model
        """
        input_str_1_embeddings = self.get_normalized_embedding_from_model_prediction(
            input_str_1, tokenizer, model
        )
        input_str_2_embeddings = self.get_normalized_embedding_from_model_prediction(
            input_str_2, tokenizer, model
        )
        cos = CosineSimilarity()
        cosine_sim = cos(input_str_1_embeddings, input_str_2_embeddings)[0]
        if print_result:
            print(
                f"Input strings:\n"
                f"\t{input_str_1}\n"
                f"\t{input_str_2}\n"
                f"Cosine Similarity: {self.color_format_similarity_calculation(cosine_sim)}"
            )
        return cosine_sim
