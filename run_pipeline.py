import argparse
import logging

from wodnesdaeg_nlp.pipeline import PipelineExecutor


logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run a wodnesdaegNLP pipeline config")
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()

    pipeline = PipelineExecutor(yaml_config_file=args.config)
    pipeline()
