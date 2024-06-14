from pydantic import BaseModel

from .text_entity import TextEntity


class ModelPrediction(BaseModel):
    pass


class POSModelPrediction(ModelPrediction):

    token: TextEntity
    pos_score: float
    pos_tag: TextEntity


class LemmatizerModelPrediction(POSModelPrediction):

    lemma: TextEntity
