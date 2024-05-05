from typing import List, Optional
from pydantic import BaseModel, computed_field

from .text_entity import Sentence


class Corpus(BaseModel):

    name: str
    sentences: List[Sentence]
    year: Optional[str] = ""

    @computed_field
    @property
    def full_name(self) -> str:
        if self.year != "":
            return f"{self.name}.{self.year}"
        else:
            return self.name
