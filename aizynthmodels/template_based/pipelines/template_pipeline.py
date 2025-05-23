"""Module containing preparing a template library"""

import importlib
from pathlib import Path

from metaflow import FlowSpec, Parameter, step
from omegaconf import OmegaConf
from rxnutils.data.batch_utils import combine_csv_batches, create_csv_batches
from rxnutils.pipeline.runner import main as validation_runner

from aizynthmodels.template_based.tools.extract_templates import main as template_runner
from aizynthmodels.utils.configs.template_based.pipelines import TemplatePipeline
from aizynthmodels.utils.hydra import load_config
from aizynthmodels.utils.pipelines.files import prefix_filename
from aizynthmodels.utils.pipelines.reporting import main as selection_runner


class TemplatesExtractionFlow(FlowSpec):
    config_path = Parameter("config", required=True)

    @step
    def start(self):
        """Loading configuration"""
        self.config = load_config(self.config_path, TemplatePipeline)
        self.next(self.import_data)

    @step
    def import_data(self):
        """Importing and transforming reaction data"""
        if not Path(self.config.import_data_path).exists():
            module_name, cls_name = self.config.data_import_class.rsplit(".", maxsplit=1)
            loaded_module = importlib.import_module(module_name)
            importer = getattr(loaded_module, cls_name)(**self.config.data_import_config)
            importer.data.to_csv(self.config.import_data_path, sep="\t", index=False)
        self.next(self.reaction_validation_setup)

    @step
    def reaction_validation_setup(self):
        """Preparing splits of the reaction data for reaction validation"""
        self.reaction_partitions = self._create_batches(
            self.config.import_data_path, self.config.validated_reactions_path
        )
        self.next(self.reaction_validation, foreach="reaction_partitions")

    @step
    def reaction_validation(self):
        """Validating reaction data"""
        pipeline_path = str(Path(__file__).parent / "rules" / "reaction_validation_pipeline.yaml")
        idx, start, end = self.input
        if idx > -1:
            validation_runner(
                [
                    "--pipeline",
                    pipeline_path,
                    "--data",
                    self.config.import_data_path,
                    "--output",
                    f"{self.config.validated_reactions_path}.{idx}",
                    "--max-workers",
                    "1",
                    "--batch",
                    str(start),
                    str(end),
                    "--no-intermediates",
                ]
            )
        self.next(self.reaction_validation_join)

    @step
    def reaction_validation_join(self, inputs):
        """Joining validated reactions"""
        self.config = inputs[0].config
        self._combine_batches(self.config.validated_reactions_path)
        self.next(self.reaction_selection)

    @step
    def reaction_selection(self):
        """Selecting reactions and producing report"""
        notebook_path = str(Path(__file__).parent / "notebooks" / "reaction_selection.py")
        if not Path(self.config.selected_reactions_path).exists():
            selection_runner(
                [
                    "--notebook",
                    notebook_path,
                    "--report_path",
                    self.config.reaction_report_path,
                    "--python_kernel",
                    self.config.python_kernel,
                    "--input_filename",
                    self.config.validated_reactions_path,
                    "--output_filename",
                    self.config.selected_reactions_path,
                ]
            )
        self.next(self.template_extraction_setup)

    @step
    def template_extraction_setup(self):
        """Preparing splits of the reaction data for template extraction"""
        self.reaction_partitions = self._create_batches(
            self.config.selected_reactions_path, self.config.unvalidated_templates_path
        )
        self.next(self.template_extraction, foreach="reaction_partitions")

    @step
    def template_extraction(self):
        """Extracting RDChiral reaction templates"""
        idx, start, end = self.input
        batch_output = f"{self.config.unvalidated_templates_path}.{idx}"
        if idx > -1 and not Path(batch_output).exists():

            config = OmegaConf.create(
                {
                    "input_data": self.config.selected_reactions_path,
                    "output_data": batch_output,
                    "radius": 1,
                    "smiles_column": "RxnSmilesClean",
                    "ringbreaker_column": "RingBreaker",
                    "batch": (start, end),
                    "expand_ring": True,
                    "expand_hetero": True,
                }
            )

            template_runner(config)
        else:
            print(f"Skipping template extraction for idx {idx}. File exists = {Path(batch_output).exists()}")
        self.next(self.template_extraction_join)

    @step
    def template_extraction_join(self, inputs):
        """Joining extracted templates"""
        self.config = inputs[0].config
        self._combine_batches(self.config.unvalidated_templates_path)
        self.next(self.template_validation_setup)

    @step
    def template_validation_setup(self):
        """Preparing splits of the reaction data for template validation"""
        self.reaction_partitions = self._create_batches(
            self.config.unvalidated_templates_path, self.config.validated_templates_path
        )
        self.next(self.template_validation, foreach="reaction_partitions")

    @step
    def template_validation(self):
        """Validating extracted templates"""
        pipline_path = str(Path(__file__).parent / "rules" / "template_validation_pipeline.yaml")
        idx, start, end = self.input
        if idx > -1:
            validation_runner(
                [
                    "--pipeline",
                    pipline_path,
                    "--data",
                    self.config.unvalidated_templates_path,
                    "--output",
                    f"{self.config.validated_templates_path}.{idx}",
                    "--max-workers",
                    "1",
                    "--batch",
                    str(start),
                    str(end),
                    "--no-intermediates",
                ]
            )
        self.next(self.template_validation_join)

    @step
    def template_validation_join(self, inputs):
        """Joining validated templates"""
        self.config = inputs[0].config
        self._combine_batches(self.config.validated_templates_path)
        self.next(self.template_selection)

    @step
    def template_selection(self):
        """Selection templates and produce report"""
        notebook_path = str(Path(__file__).parent / "notebooks" / "template_selection.py")
        output_path = prefix_filename(
            self.config.selected_templates_prefix,
            self.config.selected_templates_postfix,
        )
        if not Path(output_path).exists():
            selection_runner(
                [
                    "--notebook",
                    notebook_path,
                    "--report_path",
                    self.config.templates_report_path,
                    "--python_kernel",
                    self.config.python_kernel,
                    "--input_filename",
                    self.config.validated_templates_path,
                    "--output_prefix",
                    self.config.selected_templates_prefix,
                    "--output_postfix",
                    self.config.selected_templates_postfix,
                    "--min_occurrence",
                    str(self.config.min_template_occurrence),
                    "--config_filename",
                    self.config_path,
                ]
            )
        self.next(self.end)

    @step
    def end(self):
        print(f"Report on extracted reaction is located here: {self.config.reaction_report_path}")
        print(f"Report on extracted templates is located here: {self.config.templates_report_path}")

    def _combine_batches(self, filename):
        combine_csv_batches(filename, self.config.n_batches)

    def _create_batches(self, input_filename, output_filename):
        return create_csv_batches(input_filename, self.config.n_batches, output_filename)


if __name__ == "__main__":
    TemplatesExtractionFlow()
