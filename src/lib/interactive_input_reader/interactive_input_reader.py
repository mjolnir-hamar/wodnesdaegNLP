import src.lib.consts.file_types as file_consts
from src.lib.data_types import (
    File,
    FileLine
)


class InteractiveInputReader:

    @staticmethod
    def read_input_into_file() -> File:
        user_input = input("Sentence: ")
        return File(
            name="cli_user_input",
            lines=[
                FileLine(
                    text=user_input,
                    file_format=file_consts.CLI_USER_INPUT
                )
            ]
        )
