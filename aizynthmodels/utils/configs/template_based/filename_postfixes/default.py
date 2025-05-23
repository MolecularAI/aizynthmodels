from dataclasses import dataclass


@dataclass
class FilenamePostfixes:
    library: str = "template_library.csv"
    template_code: str = "template_code.csv"
    unique_templates: str = "unique_templates.csv.gz"
    template_lookup: str = "lookup.json"
    split_indices: str = "split_indices.npz"
    model_labels: str = "labels.npz"
    model_inputs: str = "inputs.npz"
    onnx_model: str = "expansion.onnx"
    report: str = "expansion_model_report.html"
    finder_output: str = "model_validation_finder_output.json.gz"
    multistep_report: str = "model_validation_multistep_report.json"
    expander_output: str = "model_validation_expander_output.json"
    onestep_report: str = "model_validation_onestep_report.json"
