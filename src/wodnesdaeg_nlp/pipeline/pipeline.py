import logging
from pydantic import BaseModel
from typing import List

from wodnesdaeg_nlp.pipeline.pipe.pipe import Pipe


logger = logging.getLogger()


class Pipeline(BaseModel):

    pipes: List[Pipe]

