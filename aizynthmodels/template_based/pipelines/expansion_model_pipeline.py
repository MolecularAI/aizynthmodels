"""Module containing a pipeline for training an expansion model"""

from pathlib import Path

from metaflow import FlowSpec, Parameter, step
from omegaconf import DictConfig, OmegaConf
from rxnutils.data.batch_utils import combine_sparse_matrix_batches, create_csv_batches

from aizynthmodels.template_based.pipelines.scripts.create_template_lib import main as create_template_lib
from aizynthmodels.template_based.pipelines.scripts.eval_multi_step import main as eval_multi_step
from aizynthmodels.template_based.pipelines.scripts.eval_one_step import main as eval_one_step
from aizynthmodels.template_based.pipelines.scripts.featurize import main as featurize
from aizynthmodels.template_based.pipelines.scripts.split_data import main as split_data
from aizynthmodels.template_based.tools.train import main as training_runner
from aizynthmodels.template_based.utils import get_filename
from aizynthmodels.utils.configs.template_based.pipelines import ExpansionModelPipeline
from aizynthmodels.utils.hydra import load_config
from aizynthmodels.utils.pipelines.reporting import main as report_runner


class ExpansionModelFlow(FlowSpec):
    config_path = Parameter("config", required=True)

    @step
    def start(self):
        """Loading configuration"""
        self.config = load_config(self.config_path, ExpansionModelPipeline)
        print(OmegaConf.to_yaml(self.config))
        self.next(self.create_template_metadata)

    @step
    def create_template_metadata(self):
        """Preprocess the template library for model training"""
        if not Path(get_filename(self.config, "unique_templates")).exists():
            create_template_lib(self.config)
        self.next(self.featurization_setup)

    @step
    def featurization_setup(self):
        """Preparing splits of the reaction data for feauturization"""
        self.reaction_partitions = self._create_batches(
            get_filename(self.config, "template_code"), get_filename(self.config, "model_labels")
        )
        self.next(self.featurization, foreach="reaction_partitions")

    @step
    def featurization(self):
        """Featurization of reaction data"""
        idx, start, end = self.input
        if idx > -1:
            config = DictConfig(self.config)
            config.training_pipeline.batch = [idx, start, end]
            featurize(config)
        self.next(self.featurization_join)

    @step
    def featurization_join(self, inputs):
        """Joining featurized data"""
        self.config = inputs[0].config
        self._combine_batches(get_filename(self.config, "model_labels"))
        self._combine_batches(get_filename(self.config, "model_inputs"))
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
            eval_one_step(self.config)
            eval_multi_step(self.config)
            notebook_path = str(Path(__file__).parent / "notebooks" / "expansion_model_val.py")
            report_runner(
                [
                    "--notebook",
                    notebook_path,
                    "--report_path",
                    get_filename(self.config, "report"),
                    "--python_kernel",
                    self.config.training_pipeline.python_kernel,
                    "--training_path",
                    self.config.training_pipeline.training_output,
                    "--onestep_report",
                    get_filename(self.config, "onestep_report"),
                    "--multistep_report",
                    get_filename(self.config, "multistep_report"),
                ]
            )
        self.next(self.end)

    @step
    def end(self):
        print(f"Report on trained model is located here: {get_filename(self.config, 'report')}")

    def _combine_batches(self, filename):
        combine_sparse_matrix_batches(filename, self.config.training_pipeline.n_batches)

    def _create_batches(self, input_filename, output_filename):
        return create_csv_batches(input_filename, self.config.training_pipeline.n_batches, output_filename)


if __name__ == "__main__":
    ExpansionModelFlow()
