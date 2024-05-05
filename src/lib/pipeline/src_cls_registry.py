from typing import Dict, Any

from src.lib.corpus_extractor import *
from src.lib.model_trainer import *


SRC_CLS_REGISTRY: Dict[str, Any] = {
    "CorpusExtractor": CorpusExtractor,
    "IceCorpusExtractor": IceCorpusExtractor,
    "RemXmlCorpusExtractor": RemXmlCorpusExtractor,
    "HuggingFacePytorchModelFineTuner": HuggingFacePytorchModelFineTuner
}
