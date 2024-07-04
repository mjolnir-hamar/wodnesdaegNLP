import re
import logging
from pydantic import BaseModel, computed_field
from typing import (
    Optional,
    Union,
    Any,
    List,
    Dict
)

from wodnesdaeg_nlp.pipeline.src_cls_registry import SRC_CLS_REGISTRY


logger = logging.getLogger()

EXECUTION_STEP_NAME_OUTPUT_REGEX: re.Pattern = re.compile(r"^[a-z_]+\.[a-z_]+$")


class ExecutionStep(BaseModel):

    name: str
    args: Dict[str, Union[str, int, float, bool]]
    expected_outputs: Optional[List[str]] = []
    outputs: Dict[str, Any] = {}


class Pipe(BaseModel):

    name: str
    input: Optional[str] = None
    output: Optional[str] = None
    src_cls_name: str
    args: Optional[Dict[str, Any]] = {}
    execution_steps: List[ExecutionStep]

    @computed_field
    @property
    def src_cls(self) -> Any:
        return SRC_CLS_REGISTRY[self.src_cls_name](**self.args)

    def retrieve_execution_step_output(self, step_name: str, output_name: str) -> Any:
        for execution_step in self.execution_steps:
            if execution_step.name == step_name:
                return execution_step.outputs[output_name]
        raise ValueError

    def execute(self, past_pipe_outputs: Dict[str, Any]):

        for execution_step in self.execution_steps:
            logging.info(f"Executing {execution_step.name} with {execution_step.args}")
            execution_func = getattr(self.src_cls, execution_step.name)
            args: Dict[str, Any] = {}
            for arg_name, arg_val in execution_step.args.items():
                if not isinstance(arg_val, str):
                    args[arg_name] = arg_val
                else:
                    if arg_val in past_pipe_outputs.keys():
                        args[arg_name] = past_pipe_outputs[arg_val]
                    elif not EXECUTION_STEP_NAME_OUTPUT_REGEX.match(arg_val):
                        args[arg_name] = arg_val
                    else:
                        step_name, output_name = arg_val.split(".")
                        args[arg_name] = self.retrieve_execution_step_output(
                            step_name=step_name,
                            output_name=output_name
                        )

            execution_step_outputs = execution_func(**args)

            if execution_step_outputs is not None:
                if len(execution_step.expected_outputs) == 1 and not isinstance(execution_step_outputs, tuple):
                    execution_step.outputs[execution_step.expected_outputs[0]] = execution_step_outputs
                elif len(execution_step.expected_outputs) > 1 and isinstance(execution_step_outputs, tuple):
                    for i, expected_output in enumerate(execution_step.expected_outputs):
                        execution_step.outputs[expected_output] = execution_step_outputs[i]
                else:
                    raise ValueError(
                        f"Execution step {execution_step.name} returned values but no expected outputs specified in "
                        "config"
                    )
