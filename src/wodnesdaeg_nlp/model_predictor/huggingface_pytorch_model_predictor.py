import json
from typing import (
    Any,
    List,
    Dict,
    Tuple
)
from tqdm import tqdm
from transformers import (
    PreTrainedTokenizerFast,
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSeq2SeqLM,
    pipeline,
    Pipeline
)

from src.wodnesdaeg_nlp.data_types import (
    File,
    TextEntity,
    ModelPrediction,
    POSModelPrediction,
    LemmatizerModelPrediction
)
from .model_predictor import ModelPredictor
import src.wodnesdaeg_nlp.consts.model_trainer as model_consts


class HuggingFacePytorchModelPredictor(ModelPredictor):

    def __init__(self, task: str):
        super().__init__(task)

    def load_pretrained_model_and_tokenizer(self, model_location: str) -> Tuple[PreTrainedTokenizerFast, Any]:
        """
        Loads a model and tokenizer for model inference
        """
        tokenizer = AutoTokenizer.from_pretrained(model_location)
        if self.task == model_consts.POS_TAGGING:
            model = AutoModelForTokenClassification.from_pretrained(model_location)
        elif self.task == model_consts.LEMMATIZATION:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_location)
        else:
            raise NotImplementedError
        return tokenizer, model

    def create_model_pipeline(self, model_location: str) -> Pipeline:
        """
        Creates a HuggingFace pipeline object for a specific model inference task
        """
        tokenizer, model = self.load_pretrained_model_and_tokenizer(model_location=model_location)
        if self.task == model_consts.POS_TAGGING:
            cls_task = model_consts.NER
        elif self.task == model_consts.LEMMATIZATION:
            cls_task = model_consts.TEXT_2_TEXT_GEN
        else:
            raise NotImplementedError(f"Pipeline creation for task \"{self.task}\" not yet implemented.")
        cls = pipeline(task=cls_task, model=model, tokenizer=tokenizer)
        return cls

    def run_pos_inference(self, cls: Pipeline, file_lines: File) -> List[List[POSModelPrediction]]:
        """
        Runs POS model inference on text loaded from a file
        """
        model_predictions: List[List[POSModelPrediction]] = []
        for line in file_lines.lines:
            pred = cls(line.text)
            model_predictions.append(
                self.align_bert_output_with_input_tokens(
                    input_str=line.text, bert_output=pred
                )
            )
        return model_predictions

    @staticmethod
    def run_lemmatizer_inference(
            cls: Pipeline, pos_model_predictions: List[List[POSModelPrediction]]
    ) -> List[List[LemmatizerModelPrediction]]:
        """
        Runs lemmatizer model inference on POS model predictions

        Lemmatizer model input requires POS tags
        """
        model_predictions: List[List[LemmatizerModelPrediction]] = []
        for model_preds in tqdm(pos_model_predictions, desc="Running lemmatizer inference..."):
            sentence_preds: List[LemmatizerModelPrediction] = []
            for model_pred in model_preds:
                lemma_input = f"{model_pred.token.text} {model_pred.pos_tag.text}"
                lemma_output = cls(lemma_input)[0]["generated_text"]
                sentence_preds.append(
                    LemmatizerModelPrediction(
                        **json.loads(model_pred.json()),
                        lemma=TextEntity(text=lemma_output)
                    )
                )
            model_predictions.append(sentence_preds)
        return model_predictions

    def run_inference(
            self,
            cls: Pipeline,
            file_lines: File = None,
            pos_model_predictions: List[List[POSModelPrediction]] = None
    ) -> List[List[ModelPrediction]]:
        """
        Orchestrates model inference for supported tasks
        """
        if self.task == model_consts.POS_TAGGING:
            if file_lines is None:
                raise ValueError("POS model inference requires the \"file_lines\" argument.")
            return self.run_pos_inference(cls=cls, file_lines=file_lines)
        else:
            if pos_model_predictions is None:
                raise ValueError("Lemmatizer model inference requires the \"pos_model_predictions\" argument.")
            return self.run_lemmatizer_inference(cls=cls, pos_model_predictions=pos_model_predictions)

    @staticmethod
    def align_bert_output_with_input_tokens(input_str: str, bert_output: List[Dict]) -> List[POSModelPrediction]:
        """
        Aligns BERT model tokenized output with original input (i.e. ignores subword units and takes the first
        output for each tokenized input
        """
        s_line: List[str] = input_str.split(" ")
        i: int = -1
        final_output: List[POSModelPrediction] = []
        for pred in bert_output:
            if not pred["word"].startswith("#") and pred["word"] != "$":
                i += 1
            else:
                continue
            final_output.append(
                POSModelPrediction(
                    token=TextEntity(text=s_line[i]),
                    pos_tag=TextEntity(text=pred["entity"]),
                    pos_score=float(pred["score"])
                ))
        return final_output

    @staticmethod
    def print_lemmatizer_model_predictions(model_predictions: List[List[LemmatizerModelPrediction]]):
        for sentence_model_predictions in model_predictions:
            sentence = " ".join(
                [token_model_prediction.token.text for token_model_prediction in sentence_model_predictions]
            )
            print(f"POS TAGGER AND LEMMATIZER PREDICTION FOR \"{sentence}\"")
            for token_model_prediction in sentence_model_predictions:
                print(
                    f"Token: {token_model_prediction.token.text}\t"
                    f"POS Tag: {token_model_prediction.pos_tag.text} ({token_model_prediction.pos_score})\t"
                    f"Lemma: {token_model_prediction.lemma.text}"
                )
            print()

    def print_model_predictions(self, model_predictions: List[List[ModelPrediction]]):
        if self.task == model_consts.POS_TAGGING:
            pass
        elif self.task == model_consts.LEMMATIZATION:
            self.print_lemmatizer_model_predictions(model_predictions=model_predictions)
        else:
            raise NotImplementedError(f"Printing model predictions for task \"{self.task}\" is not yet implemented.")
