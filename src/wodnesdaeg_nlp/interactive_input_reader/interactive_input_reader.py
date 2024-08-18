import wodnesdaeg_nlp.consts.file_types as file_consts
from wodnesdaeg_nlp.consts.string_consts import PUNCTS
from wodnesdaeg_nlp.data_types import (
    File,
    FileLine
)


class InteractiveInputReader:

    def read_input_into_file(self) -> File:
        user_input = input("Sentence: ")
        return File(
            name="cli_user_input",
            lines=[
                FileLine(
                    text=self.clean_line(user_input),
                    file_format=file_consts.CLI_USER_INPUT
                )
            ]
        )

    @staticmethod
    def clean_line(line: str) -> str:
        for punct in PUNCTS:
            line = line.replace(punct, "")
        return line
