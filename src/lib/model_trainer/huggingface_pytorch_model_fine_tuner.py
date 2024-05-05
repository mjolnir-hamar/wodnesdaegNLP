import logging
import numpy as np
import pandas as pd
from copy import deepcopy
from typing import (
    Any,
    List,
    Tuple,
    Set,
    Dict
)
import evaluate
from datasets import (
    ClassLabel,
    Dataset,
    DatasetDict,
    Sequence,
)
from transformers import (
    PreTrainedTokenizerFast,
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollator,
    DataCollatorForTokenClassification,
    BatchEncoding
)

from src.lib.data_types import Corpus
from .model_trainer import ModelTrainer
import src.lib.consts.model_trainer as model_consts


logger = logging.getLogger(__name__)

TASK_TO_HF_AUTO_MODEL: Dict[str, Any] = {
    model_consts.POS_TAGGING: AutoModelForTokenClassification
}


class HuggingFacePytorchModelFineTuner(ModelTrainer):

    def __init__(self, task: str):
        super().__init__(task)

    def convert_corpora_to_dataset(self, corpora: List[Corpus]) -> Tuple[Dataset, ClassLabel]:

        if self.task == model_consts.POS_TAGGING:
            return self.convert_corpora_to_pos_dataset(corpora=corpora)
        else:
            raise NotImplementedError

    @staticmethod
    def convert_corpora_to_pos_dataset(corpora: List[Corpus]) -> Tuple[Dataset, ClassLabel]:

        dataset_raw: Dict = {
            model_consts.TEXT: [],
            model_consts.NER_TAGS: []
        }

        all_ner_tags: Set = set()

        for corpus in corpora:
            for sentence in corpus.sentences:
                tokens: List[str] = []
                ner_tags: List[str] = []
                for word in sentence.words:
                    tokens.append(word.text.lower())
                    ner_tags.append(word.ner_tag.tag)
                if len(tokens) == len(ner_tags) and len(tokens) > 0:
                    dataset_raw[model_consts.TEXT].append(tokens)
                    dataset_raw[model_consts.NER_TAGS].append(ner_tags)
                    all_ner_tags |= set(ner_tags)

        logging.info(f"Loaded {len(dataset_raw[model_consts.TEXT])} texts")

        classmap: ClassLabel = ClassLabel(num_classes=len(all_ner_tags), names=list(all_ner_tags))
        dataset: Dataset = Dataset.from_pandas(pd.DataFrame(data=dataset_raw))
        dataset = dataset.cast_column(model_consts.NER_TAGS, Sequence(classmap))

        return dataset, classmap

    @staticmethod
    def train_test_val_split(dataset: Dataset, train_perc: float, test_perc: float) -> DatasetDict:
        test_val_size: float = 1.0 - train_perc
        val_size: float = 1.0 - (test_perc / test_val_size)
        train_test: DatasetDict = dataset.train_test_split(test_size=test_val_size)
        test_val: DatasetDict = train_test[model_consts.TEST].train_test_split(test_size=val_size)
        dataset_dict = DatasetDict({
            model_consts.TRAIN: train_test[model_consts.TRAIN],
            model_consts.TEST: test_val[model_consts.TRAIN],
            model_consts.VAL: test_val[model_consts.TEST]
        })
        logging.info(
            f"Train size: {len(dataset_dict[model_consts.TRAIN])}, "
            f"Test size: {len(dataset_dict[model_consts.TEST])}, "
            f"Val size: {len(dataset_dict[model_consts.VAL])}"
        )

        return dataset_dict

    def load_pretrained_model_and_tokenizer(
            self, model_location: str, classmap: ClassLabel
    ) -> Tuple[PreTrainedTokenizerFast, Any]:
        tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(model_location)
        label2id: Dict[str, int] = deepcopy(classmap._str2int)
        id2label: Dict[int, str] = {i: label for label, i in label2id.items()}
        if self.task == model_consts.POS_TAGGING:
            model: Any = AutoModelForTokenClassification.from_pretrained(
                model_location,
                num_labels=len(id2label.keys()),
                id2label=id2label,
                label2id=label2id
            )
        else:
            raise NotImplementedError

        return tokenizer, model

    def apply_tokenizer(
            self, tokenizer: PreTrainedTokenizerFast, dataset_dict: DatasetDict, max_seq_len: int
    ) -> DatasetDict:
        if self.task == model_consts.POS_TAGGING:
            return dataset_dict.map(
                self.apply_tokenizer_for_pos_tagging,
                batched=True,
                fn_kwargs={
                    model_consts.TOKENIZER: tokenizer,
                    model_consts.MAX_SEQ_LEN: max_seq_len
                }
            )
        else:
            raise NotImplementedError

    @staticmethod
    def apply_tokenizer_for_pos_tagging(
            examples: Dataset, tokenizer: PreTrainedTokenizerFast, max_seq_len: int
    ) -> BatchEncoding:
        """
        Based on align_labels_with_tokens from https://huggingface.co/learn/nlp-course/en/chapter7/2
        """
        tokenized_inputs: BatchEncoding = tokenizer(
            examples["text"],
            truncation=True,
            is_split_into_words=True,
            max_length=max_seq_len
        )

        labels: List[List[int]] = []
        for i, label in enumerate(examples[f"ner_tags"]):
            word_ids: List[int] = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx: int = None
            label_ids: List[int] = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def prepare_data_collator(self, tokenizer: PreTrainedTokenizerFast) -> DataCollator:
        if self.task == model_consts.POS_TAGGING:
            return DataCollatorForTokenClassification(tokenizer=tokenizer)
        else:
            raise NotImplementedError

    def prepare_compute_metrics(self, label_list: List[str]) -> Any:
        if self.task == model_consts.POS_TAGGING:
            seqeval = evaluate.load("seqeval")

            def compute_metrics(p):
                """
                Based on compute_metrics from https://huggingface.co/learn/nlp-course/en/chapter7/2
                """
                nonlocal label_list
                nonlocal seqeval

                predictions, labels = p
                predictions = np.argmax(predictions, axis=2)

                true_predictions = [
                    [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                    for prediction, label in zip(predictions, labels)
                ]
                true_labels = [
                    [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
                    for prediction, label in zip(predictions, labels)
                ]

                results = seqeval.compute(
                    predictions=true_predictions,
                    references=true_labels
                )
                return {
                    "precision": results["overall_precision"],
                    "recall": results["overall_recall"],
                    "f1": results["overall_f1"],
                    "accuracy": results["overall_accuracy"],
                }
        else:
            raise NotImplementedError

        return compute_metrics

    def train_model(
            self,
            dataset_dict: DatasetDict,
            model: Any,
            tokenizer: PreTrainedTokenizerFast,
            output_dir: str,
            learning_rate: float = 2e-5,
            weight_decay: float = 0.01,
            train_batch_size: int = 16,
            val_batch_size: int = 16,
            epochs: int = 4,
            val_strategy: str = "epoch",
            save_strategy: str = "epoch",
    ):

        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=val_batch_size,
            num_train_epochs=epochs,
            evaluation_strategy=val_strategy,
            save_strategy=save_strategy,
            push_to_hub=False
        )

        data_collator = self.prepare_data_collator(tokenizer=tokenizer)
        compute_metrics = self.prepare_compute_metrics(
            label_list=dataset_dict[model_consts.TRAIN].features[model_consts.NER_TAGS].feature.names
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset_dict[model_consts.TRAIN],
            eval_dataset=dataset_dict[model_consts.VAL],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )

        trainer.train()
