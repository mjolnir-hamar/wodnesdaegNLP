from pydantic import BaseModel
from typing import (
    Optional,
    List
)


class Tag(BaseModel):

    tag: str


class NerTag(Tag):

    extra_semantic_idents: Optional[List[str]] = []
