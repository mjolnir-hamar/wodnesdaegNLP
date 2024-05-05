from pydantic import BaseModel
from typing import List, Optional


class Tag(BaseModel):

    tag: str


class NerTag(Tag):

    extra_semantic_idents: Optional[List[str]] = []
