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
    args: Dict[str, Union[str, int, float, bool, Dict[str, Any]]]
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
        raise ValueError(f"Cannot retrieve execution step output for \"{step_name}\" from \"{output_name}\"")

    def resolve_arg_val(self, arg_val: Any, past_pipe_outputs: Dict[str, Any]):
        if isinstance(arg_val, bool) or isinstance(arg_val, int) or isinstance(arg_val, float):
            return arg_val
        elif isinstance(arg_val, str):
            if arg_val in past_pipe_outputs.keys():
                return past_pipe_outputs[arg_val]
            elif not EXECUTION_STEP_NAME_OUTPUT_REGEX.match(arg_val):
                return arg_val
            else:
                step_name, output_name = arg_val.split(".")
                return self.retrieve_execution_step_output(
                    step_name=step_name,
                    output_name=output_name
                )

    def prepare_args(self, prepared_args: Dict[str, Any], raw_args: Dict, past_pipe_outputs: Dict[str, Any]):

        for arg_name, arg_val in raw_args.items():
            if not isinstance(arg_val, list) and not isinstance(arg_val, dict):
                prepared_args[arg_name] = self.resolve_arg_val(arg_val, past_pipe_outputs)
            elif isinstance(arg_val, list):
                resolved_sub_arg_val_list = []
                for sub_arg_val in arg_val:
                    if isinstance(sub_arg_val, dict) or isinstance(sub_arg_val, list):
                        raise NotImplementedError(
                            f"Invalid nested argument value for \"{arg_name}\"; "
                            f"list entries cannot be lists or dictionaries: \"{sub_arg_val}\""
                        )
                    else:
                        resolved_sub_arg_val_list.append(self.resolve_arg_val(sub_arg_val, past_pipe_outputs))
                prepared_args[arg_name] = resolved_sub_arg_val_list
            elif isinstance(arg_val, dict):
                resolved_sub_arg_val_dict = {}
                for sub_arg_name, sub_arg_val in arg_val.items():
                    if isinstance(sub_arg_val, dict) or isinstance(sub_arg_val, list):
                        raise NotImplementedError(
                            f"Invalid nested argument value for \"{arg_name}.{sub_arg_name}\"; "
                            f"list entries cannot be lists or dictionaries: \"{sub_arg_val}\""
                        )
                    else:
                        resolved_sub_arg_val_dict[sub_arg_name] = self.resolve_arg_val(sub_arg_val, past_pipe_outputs)
                prepared_args[arg_name] = resolved_sub_arg_val_dict
            else:
                raise ValueError

        return prepared_args

    def execute(self, past_pipe_outputs: Dict[str, Any]):

        for execution_step in self.execution_steps:
            logging.info(f"Executing {execution_step.name} with {execution_step.args}")
            execution_func = getattr(self.src_cls, execution_step.name)
            args: Dict[str, Any] = self.prepare_args(
                prepared_args={},
                raw_args=execution_step.args,
                past_pipe_outputs=past_pipe_outputs
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
