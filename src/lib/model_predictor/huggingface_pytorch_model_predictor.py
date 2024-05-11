from typing import (
    Any,
    List,
    Dict,
    Tuple
)
from transformers import (
    PreTrainedTokenizerFast,
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
    Pipeline
)

from src.lib.data_types import (
    File,
    TextEntity,
    ModelPrediction
)
from .model_predictor import ModelPredictor
import src.lib.consts.model_trainer as model_consts


class HuggingFacePytorchModelPredictor(ModelPredictor):

    def __init__(self, task: str):
        super().__init__(task)

    def load_pretrained_model_and_tokenizer(self, model_location: str) -> Tuple[PreTrainedTokenizerFast, Any]:
        tokenizer = AutoTokenizer.from_pretrained(model_location)
        if self.task == model_consts.POS_TAGGING:
            model = AutoModelForTokenClassification.from_pretrained(model_location)
        else:
            raise NotImplementedError
        return tokenizer, model

    def create_model_pipeline(self, model_location: str) -> Pipeline:
        tokenizer, model = self.load_pretrained_model_and_tokenizer(model_location=model_location)
        if self.task == model_consts.POS_TAGGING:
            cls_task = model_consts.NER
        else:
            raise NotImplementedError
        cls = pipeline(task=cls_task, model=model, tokenizer=tokenizer)
        return cls

    def run_inference(self, cls: Pipeline, file_lines: File) -> List[List[ModelPrediction]]:
        model_predictions: List[List[ModelPrediction]] = []
        for line in file_lines.lines:
            pred = cls(line.text)
            model_predictions.append(
                self.align_bert_output_with_input_tokens(
                    input_str=line.text, bert_output=pred
                )
            )
        return model_predictions

    @staticmethod
    def align_bert_output_with_input_tokens(input_str: str, bert_output: List[Dict]) -> List[ModelPrediction]:
        s_line: List[str] = input_str.split(" ")
        i: int = -1
        final_output: List[ModelPrediction] = []
        for pred in bert_output:
            if not pred["word"].startswith("#") and pred["word"] != "$":
                i += 1
            else:
                continue
            final_output.append(
                ModelPrediction(
                    text=TextEntity(text=s_line[i]),
                    label=TextEntity(text=pred["entity"]),
                    score=float(pred["score"])
                ))
        return final_output
