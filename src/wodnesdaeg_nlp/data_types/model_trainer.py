from pydantic import BaseModel
from typing import Any
from transformers.trainer_utils import TrainOutput


class ModelTrainerOutput(BaseModel):

    trainer: Any
    trainer_output: TrainOutput
