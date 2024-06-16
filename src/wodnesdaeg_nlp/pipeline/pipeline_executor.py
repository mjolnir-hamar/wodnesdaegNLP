import yaml
from typing import (
    Any,
    Dict
)

from .pipeline import Pipeline
import src.wodnesdaeg_nlp.consts.pipeline as pipeline_consts


class PipelineExecutor:

    def __init__(self, yaml_config_file: str):
        self._read_config(yaml_config_file=yaml_config_file)
        self._build()

    def _read_config(self, yaml_config_file: str):
        with open(yaml_config_file, "r") as _y:
            self.pipeline_config = yaml.safe_load(_y)

    def _build(self):
        self.pipeline = Pipeline(
            **self.pipeline_config[pipeline_consts.PIPELINE]
        )

    def __call__(self):
        pipe_outputs: Dict[str, Any] = {}
        for pipe in self.pipeline.pipes:
            pipe.execute(past_pipe_outputs=pipe_outputs)
            if pipe.output is not None:
                step_name, output_name = pipe.output.split(".")
                pipe_outputs[f"{pipe.name}.{output_name}"] = pipe.retrieve_execution_step_output(
                    step_name=step_name,
                    output_name=output_name
                )

    def __str__(self):
        return self.pipeline.json()
