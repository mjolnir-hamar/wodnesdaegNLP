from typing import Dict, Any

from wodnesdaeg_nlp.file_reader import *
from wodnesdaeg_nlp.interactive_input_reader import *
from wodnesdaeg_nlp.corpus_extractor import *
from wodnesdaeg_nlp.model_trainer import *
from wodnesdaeg_nlp.model_predictor import *
from wodnesdaeg_nlp.semantic_similarity_evaluator import *


SRC_CLS_REGISTRY: Dict[str, Any] = {
    "FlatFileReader": FlatFileReader,
    "InteractiveInputReader": InteractiveInputReader,
    "CorpusExtractor": CorpusExtractor,
    "CorpusJoiner": CorpusJoiner,
    "IceCorpusExtractor": IceCorpusExtractor,
    "ISWOCTreebankCorpusExtractor": ISWOCTreebankCorpusExtractor,
    "LatinTreebankPerseusCorpusExtractor": LatinTreebankPerseusCorpusExtractor,
    "RemXmlCorpusExtractor": RemXmlCorpusExtractor,
    "HuggingFacePytorchModelFineTuner": HuggingFacePytorchModelFineTuner,
    "HuggingFacePytorchModelPredictor": HuggingFacePytorchModelPredictor,
    "SemanticSimilarityEvaluator": SemanticSimilarityEvaluator
}
