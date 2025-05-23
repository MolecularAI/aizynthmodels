"""Module that evaluates a filter model"""

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from rxnutils.data.batch_utils import nlines

from aizynthmodels.quick_filter.tools.inference_score import main as run_inference
from aizynthmodels.template_based.utils import get_filename
from aizynthmodels.utils.configs.quick_filter import pipelines  # noqa: F401
from aizynthmodels.utils.configs.quick_filter.inference_score import InferenceScore
from aizynthmodels.utils.hydra import custom_config


@hydra.main(version_base=None, config_name="filter_pipeline")
@custom_config
def main(config: DictConfig) -> None:
    """Command-line interface for the featurization tool"""

    inference_config = OmegaConf.structured(InferenceScore)
    inference_config.file_prefix = config.file_prefix
    inference_config.datamodule = config.datamodule

    ndata_points = nlines(get_filename(config, "library", "testing"))
    inference_config.batch_size = int(ndata_points / 10) + 1

    latest = sorted((Path(config.output_directory) / "quick_filter").glob("version*"))[-1]
    inference_config.model_path = (latest / "checkpoints" / "last.ckpt").as_posix()

    inference_config.output_predictions = get_filename(config, "test_predictions")
    inference_config.output_score_data = get_filename(config, "test_metrics")

    run_inference(inference_config)


if __name__ == "__main__":
    main()
