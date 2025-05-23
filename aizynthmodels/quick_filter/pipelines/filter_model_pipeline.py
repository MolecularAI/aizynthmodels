"""Module containing a pipeline for training a quick-filter model"""

from pathlib import Path

from metaflow import FlowSpec, Parameter, step
from omegaconf import DictConfig, OmegaConf
from rxnutils.data.batch_utils import (
    combine_csv_batches,
    combine_numpy_array_batches,
    combine_sparse_matrix_batches,
    create_csv_batches,
)

from aizynthmodels.quick_filter.pipelines.scripts.create_filter_lib import main as create_filter_lib
from aizynthmodels.quick_filter.pipelines.scripts.eval_filter import main as eval_filter
from aizynthmodels.quick_filter.pipelines.scripts.featurize import main as featurize
from aizynthmodels.quick_filter.pipelines.scripts.split_data import main as split_data
from aizynthmodels.quick_filter.tools.generate_data import main as generate_data
from aizynthmodels.quick_filter.tools.train import main as training_runner
from aizynthmodels.template_based.utils import get_filename
from aizynthmodels.utils.configs.quick_filter.pipelines import FilterModelPipeline
from aizynthmodels.utils.hydra import load_config
from aizynthmodels.utils.pipelines.reporting import main as report_runner


class FilterModelFlow(FlowSpec):
    config_path = Parameter("config", required=True)

    @step
    def start(self):
        """Loading configuration"""
        self.config = load_config(self.config_path, FilterModelPipeline)
        print(OmegaConf.to_yaml(self.config))
        self.next(self.generation_setup)

    @step
    def generation_setup(self):
        """Preparing splits of the reaction data for negative data generation"""
        self.reaction_partitions = self._create_batches(
            get_filename(self.config, "template_library"), get_filename(self.config, "generated_library")
        )
        self.next(self.generation, foreach="reaction_partitions")

    @step
    def generation(self):
        """Generation of negative data reaction data"""
        idx, start, end = self.input
        if idx > -1:
            config = DictConfig(self.config)
            config.batch = [idx, start, end]
            generate_data(config)
        self.next(self.generation_join)

    @step
    def generation_join(self, inputs):
        """Joining negative data"""
        self.config = inputs[0].config
        combine_csv_batches(get_filename(self.config, "generated_library"), self.config.n_batches)
        self.next(self.create_filter_metadata)

    @step
    def create_filter_metadata(self):
        """Preprocess the filter library for model training"""
        if not Path(get_filename(self.config, "library")).exists():
            create_filter_lib(self.config)
        self.next(self.featurization_setup)

    @step
    def featurization_setup(self):
        """Preparing splits of the reaction data for feauturization"""
        self.reaction_partitions = self._create_batches(
            get_filename(self.config, "library"), get_filename(self.config, "model_labels")
        )
        self.next(self.featurization, foreach="reaction_partitions")

    @step
    def featurization(self):
        """Featurization of reaction data"""
        idx, start, end = self.input
        if idx > -1:
            config = DictConfig(self.config)
            config.batch = [idx, start, end]
            featurize(config)
        self.next(self.featurization_join)

    @step
    def featurization_join(self, inputs):
        """Joining featurized data"""
        self.config = inputs[0].config
        combine_numpy_array_batches(get_filename(self.config, "model_labels"), self.config.n_batches)
        combine_sparse_matrix_batches(get_filename(self.config, "model_inputs_prod"), self.config.n_batches)
        combine_sparse_matrix_batches(get_filename(self.config, "model_inputs_rxn"), self.config.n_batches)
        self.next(self.split_data)

    @step
    def split_data(self):
        """Split featurized data into training, validation and testing"""
        if not Path(get_filename(self.config, "model_labels", "training")).exists():
            split_data(self.config)
        self.next(self.model_training)

    @step
    def model_training(self):
        """Train the expansion model"""
        if not Path(get_filename(self.config, "onnx_model")).exists():
            training_runner(self.config)
        self.next(self.model_validation)

    @step
    def model_validation(self):
        """Validate the trained model"""
        if not Path(get_filename(self.config, "report")).exists():
            eval_filter(self.config)
            notebook_path = str(Path(__file__).parent / "notebooks" / "filter_model_val.py")
            report_runner(
                [
                    "--notebook",
                    notebook_path,
                    "--report_path",
                    get_filename(self.config, "report"),
                    "--python_kernel",
                    self.config.python_kernel,
                    "--training_path",
                    self.config.output_directory,
                    "--metrics_path",
                    get_filename(self.config, "test_metrics"),
                ]
            )
        self.next(self.end)

    @step
    def end(self):
        print(f"Report on trained model is located here: {get_filename(self.config, 'report')}")

    def _create_batches(self, input_filename, output_filename):
        return create_csv_batches(input_filename, self.config.n_batches, output_filename)


if __name__ == "__main__":
    FilterModelFlow()
