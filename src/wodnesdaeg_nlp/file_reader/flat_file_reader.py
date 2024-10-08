from typing import List

import wodnesdaeg_nlp.consts.file_types as file_consts
from wodnesdaeg_nlp.consts.string_consts import PUNCTS
from wodnesdaeg_nlp.data_types import (
    File,
    FileLine
)


class FlatFileReader:

    def read_file(self, file_path: str, column_number: int = 0, file_format: str = file_consts.TSV) -> File:
        lines: List[FileLine] = []
        with open(file_path, "r") as _f:
            for line in _f:
                lines.append(
                    FileLine(text=self.split_line(line=self.clean_line(line), file_format=file_format)[column_number])
                )
        return File(
            name=file_path.split("/")[-1],
            lines=lines
        )

    @staticmethod
    def split_line(line: str, file_format: str) -> List[str]:
        if file_format == file_consts.TSV:
            return line.split("\t")
        raise NotImplementedError

    @staticmethod
    def clean_line(line: str) -> str:
        for punct in PUNCTS:
            line = line.replace(punct, "")
        return line
