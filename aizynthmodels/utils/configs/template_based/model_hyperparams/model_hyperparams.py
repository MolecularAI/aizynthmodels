from dataclasses import dataclass


@dataclass
class ModelHyperparams:
    fingerprint_radius: int = 2
    fingerprint_size: int = 2048
    num_hidden_layers: int = 1
    num_hidden_nodes: int = 512
    dropout: float = 0.5
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    chirality: bool = False
