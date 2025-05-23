from dataclasses import dataclass


@dataclass
class ModelHyperparams:
    fp_size: int = 2048
    lstm_size: int = 1024
    dropout_prob: float = 0.5
    learning_rate: float = 0.001
    weight_decay: float = 0.001
