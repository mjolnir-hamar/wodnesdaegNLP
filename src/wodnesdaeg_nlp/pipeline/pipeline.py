import logging
from pydantic import BaseModel
from typing import List

from src.wodnesdaeg_nlp.pipeline.pipe.pipe import Pipe


logger = logging.getLogger()


class Pipeline(BaseModel):

    pipes: List[Pipe]

