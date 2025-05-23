from dataclasses import dataclass


@dataclass
class FilenamePostfixes:
    library: str = "filter_library.csv"
    template_library: str = "template_library.csv"
    generated_library: str = "generated_library.csv"
    split_indices: str = "split_indices.npz"
    model_labels: str = "labels.npz"
    model_inputs_rxn: str = "inputs_rxn.npz"
    model_inputs_prod: str = "inputs_prod.npz"
    onnx_model: str = "expansion.onnx"
    test_predictions: str = "predictions.json"
    test_metrics: str = "metrics_scores.csv"
    report: str = "filter_model_report.html"
