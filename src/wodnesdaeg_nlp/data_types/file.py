from pydantic import BaseModel
from typing import List


class FileLine(BaseModel):

    text: str


class File(BaseModel):

    name: str
    lines: List[FileLine]
