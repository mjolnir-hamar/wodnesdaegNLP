from pydantic import BaseModel

from .text_entity import TextEntity


class ModelPrediction(BaseModel):

    text: TextEntity
    score: float
    label: TextEntity
