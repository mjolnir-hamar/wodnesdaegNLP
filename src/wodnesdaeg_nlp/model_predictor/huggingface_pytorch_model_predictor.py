import json
import logging
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
    AutoModelForCausalLM,
    pipeline,
    Pipeline
)
from peft import PeftModel
from tabulate import tabulate

from wodnesdaeg_nlp.util import TermColor
from wodnesdaeg_nlp.data_types import (
    File,
    TextEntity,
    ModelPrediction,
    POSModelPrediction,
    LemmatizerModelPrediction
)
from .model_predictor import ModelPredictor
import wodnesdaeg_nlp.consts.model_trainer as model_consts


logger = logging.getLogger(__name__)


class HuggingFacePytorchModelPredictor(ModelPredictor):

    def __init__(self, task: str):
        super().__init__(task)

    def load_pretrained_model_and_tokenizer(self, model_location: str, is_lora: bool = False) -> Tuple[PreTrainedTokenizerFast, Any]:
        """
        Loads a model and tokenizer for model inference
        """
        tokenizer = AutoTokenizer.from_pretrained(model_location)
        if self.task == model_consts.POS_TAGGING:
            model = AutoModelForTokenClassification.from_pretrained(model_location)
        elif self.task == model_consts.LEMMATIZATION:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_location)
        elif self.task == model_consts.LEMMATIZATION_CAUSAL_LM:
            model = AutoModelForCausalLM.from_pretrained(model_location)
        else:
            raise NotImplementedError

        if is_lora:
            logger.info("Loading a LoRA model")
            model = PeftModel.from_pretrained(model, model_location)

        return tokenizer, model

    def create_model_pipeline(self, model_location: str, is_lora: bool = False) -> Pipeline:
        """
        Creates a HuggingFace pipeline object for a specific model inference task
        """
        tokenizer, model = self.load_pretrained_model_and_tokenizer(model_location=model_location, is_lora=is_lora)
        if self.task == model_consts.POS_TAGGING:
            cls_task = model_consts.NER
        elif self.task == model_consts.LEMMATIZATION:
            cls_task = model_consts.TEXT_2_TEXT_GEN
        elif self.task == model_consts.LEMMATIZATION_CAUSAL_LM:
            cls_task = model_consts.TEST_GEN
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
    def post_process_causal_lm_lemmatizer_output(pred_lemma: str) -> str:
        """
        Post process Causal LM lemmatizer output to extract the tagged lemma, remove the tags, and strip white space
        Input: hanom PRO <lemma> hann </lemma>
        Output: hann
        """
        return model_consts.LEMMA_SPAN_REGEX.findall(
            pred_lemma
        )[0].replace(
            model_consts.LEMMA_START_TAG, ""
        ).replace(
            model_consts.LEMMA_END_TAG, ""
        ).strip()

    def run_lemmatizer_inference(
            self, cls: Pipeline, pos_model_predictions: List[List[POSModelPrediction]] = None, file_lines: File = None
    ) -> List[List[LemmatizerModelPrediction]]:
        """
        Runs lemmatizer model inference on POS model predictions

        Lemmatizer model input requires POS tags
        """
        model_predictions: List[List[LemmatizerModelPrediction]] = []
        if pos_model_predictions is not None:
            for model_preds in tqdm(pos_model_predictions, desc="Running lemmatizer inference..."):
                sentence_preds: List[LemmatizerModelPrediction] = []
                for model_pred in model_preds:
                    lemma_input = f"{model_pred.token.text} {model_pred.pos_tag.text}"
                    lemma_output = cls(lemma_input)[0]["generated_text"]
                    if self.task == model_consts.LEMMATIZATION_CAUSAL_LM:
                        lemma_output = self.post_process_causal_lm_lemmatizer_output(lemma_output)
                    sentence_preds.append(
                        LemmatizerModelPrediction(
                            **json.loads(model_pred.json()),
                            lemma=TextEntity(text=lemma_output)
                        )
                    )
                model_predictions.append(sentence_preds)
        else:
            for line in file_lines.lines:
                token, pos_tag = line.text.strip().split(" ")
                lemma_output = cls(line.text)[0]["generated_text"]
                if self.task == model_consts.LEMMATIZATION_CAUSAL_LM:
                    lemma_output = self.post_process_causal_lm_lemmatizer_output(lemma_output)
                model_predictions.append([
                    LemmatizerModelPrediction(
                        token=TextEntity(text=token),
                        pos_score=-1.0,
                        pos_tag=TextEntity(text=pos_tag),
                        lemma=TextEntity(text=lemma_output)
                    )
                ])
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
            if pos_model_predictions:
                return self.run_lemmatizer_inference(cls=cls, pos_model_predictions=pos_model_predictions)
            elif file_lines:
                return self.run_lemmatizer_inference(cls=cls, file_lines=file_lines)
            else:
                raise ValueError(
                    "Lemmatizer model inference requires either the \"pos_model_predictions\" or \"file_lines\" "
                    "argument."
                )

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
            print(
                f"{TermColor.BOLD}{TermColor.UNDERLINE}POS TAGGER AND LEMMATIZER PREDICTION FOR "
                f"\"{sentence}\"{TermColor.END}"
            )
            output_table = [
                [
                    f"{TermColor.BOLD}Token{TermColor.END}",
                    f"{TermColor.BOLD}POS Tag (Conf. Score){TermColor.END}",
                    f"{TermColor.BOLD}Lemma{TermColor.END}"
                ]
            ]
            for token_model_prediction in sentence_model_predictions:
                pos_score = token_model_prediction.pos_score
                if pos_score >= 0.85:
                    pos_score = f"{TermColor.GREEN}{pos_score}{TermColor.END}"
                elif pos_score >= 0.5:
                    pos_score = f"{TermColor.YELLOW}{pos_score}{TermColor.END}"
                elif pos_score >= 0.0:
                    pos_score = f"{TermColor.RED}{pos_score}{TermColor.END}"
                else:
                    pos_score = f"{TermColor.DIM}{pos_score}{TermColor.END}"
                output_table.append([
                    token_model_prediction.token.text,
                    f"{token_model_prediction.pos_tag.text} ({pos_score})",
                    token_model_prediction.lemma.text
                ])
            print(tabulate(output_table, headers="firstrow"))
            print()

    def print_model_predictions(self, model_predictions: List[List[ModelPrediction]]):
        if self.task == model_consts.POS_TAGGING:
            pass
        elif self.task == model_consts.LEMMATIZATION or self.task == model_consts.LEMMATIZATION_CAUSAL_LM:
            self.print_lemmatizer_model_predictions(model_predictions=model_predictions)
        else:
            raise NotImplementedError(f"Printing model predictions for task \"{self.task}\" is not yet implemented.")

    @staticmethod
    def write_lemmatizer_model_predictions(model_predictions: List[List[LemmatizerModelPrediction]], output_fname: str):
        model_output_json = []
        for sentence_model_predictions in model_predictions:
            sentence = " ".join(
                [token_model_prediction.token.text for token_model_prediction in sentence_model_predictions]
            ).strip()
            sentence_output_json = {"sentence": sentence, "token_predictions": []}
            for token_model_prediction in sentence_model_predictions:
                sentence_output_json["token_predictions"].append({
                    "token": token_model_prediction.token.text.strip(),
                    "pos_tag": {
                        "tag": token_model_prediction.pos_tag.text,
                        "score":token_model_prediction.pos_score
                    },
                    "lemma": token_model_prediction.lemma.text
                })
            model_output_json.append(sentence_output_json)
        with open(output_fname, "w") as _o:
            json.dump(model_output_json, _o, indent=2)


    def write_model_predictions(self, model_predictions: List[List[ModelPrediction]], output_fname: str):
        if self.task == model_consts.POS_TAGGING:
            pass
        elif self.task == model_consts.LEMMATIZATION:
            self.write_lemmatizer_model_predictions(model_predictions=model_predictions, output_fname=output_fname)
        else:
            raise NotImplementedError(f"Writing model predictions for task \"{self.task}\" is not yet implemented.")
