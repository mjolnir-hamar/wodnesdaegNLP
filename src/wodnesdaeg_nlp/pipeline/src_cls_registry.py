from typing import Dict, Any

from src.wodnesdaeg_nlp.file_reader import *
from src.wodnesdaeg_nlp.interactive_input_reader import *
from src.wodnesdaeg_nlp.corpus_extractor import *
from src.wodnesdaeg_nlp.model_trainer import *
from src.wodnesdaeg_nlp.model_predictor import *


SRC_CLS_REGISTRY: Dict[str, Any] = {
    "FlatFileReader": FlatFileReader,
    "InteractiveInputReader": InteractiveInputReader,
    "CorpusExtractor": CorpusExtractor,
    "IceCorpusExtractor": IceCorpusExtractor,
    "LatinTreebankPerseusCorpusExtractor": LatinTreebankPerseusCorpusExtractor,
    "RemXmlCorpusExtractor": RemXmlCorpusExtractor,
    "HuggingFacePytorchModelFineTuner": HuggingFacePytorchModelFineTuner,
    "HuggingFacePytorchModelPredictor": HuggingFacePytorchModelPredictor
}
