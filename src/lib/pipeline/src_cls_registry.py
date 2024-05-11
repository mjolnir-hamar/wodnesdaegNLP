from typing import Dict, Any

from src.lib.file_reader import *
from src.lib.corpus_extractor import *
from src.lib.model_trainer import *
from src.lib.model_predictor import *


SRC_CLS_REGISTRY: Dict[str, Any] = {
    "FlatFileReader": FlatFileReader,
    "CorpusExtractor": CorpusExtractor,
    "IceCorpusExtractor": IceCorpusExtractor,
    "RemXmlCorpusExtractor": RemXmlCorpusExtractor,
    "HuggingFacePytorchModelFineTuner": HuggingFacePytorchModelFineTuner,
    "HuggingFacePytorchModelPredictor": HuggingFacePytorchModelPredictor
}
