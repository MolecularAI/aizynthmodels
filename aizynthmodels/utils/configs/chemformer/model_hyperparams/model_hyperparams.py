from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Optimizer:
    """Optimizer parameters to use in training"""

    scheduler: str = "cycle"
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])


@dataclass
class InferenceModelHyperparams:
    """Hyperparameters needed to run inference tasks (mode: eval)"""

    max_seq_len: int = 512
    batch_first: bool = False


@dataclass
class ClientModelHyperparams(InferenceModelHyperparams):
    """Hyperparameters needed to run the ChemformerClient"""

    api_url: Optional[str] = None
    api_key: Optional[str] = None
    header: Optional[Dict] = None
    model_reagents: bool = False
    pad_token_idx: Optional[int] = None
    vocabulary_path: Optional[str] = None

    # These variables should not be updated
    optimizer: Dict = field(default_factory=lambda: {})


@dataclass
class TrainModelHyperparams:
    """Chemformer model hyperparameter config for training."""

    max_seq_len: int = 512
    batch_first: bool = False

    d_model: int = 512
    num_layers: int = 6
    num_heads: int = 8
    d_feedforward: int = 2048
    optimizer: Optimizer = Optimizer()
    activation: str = "gelu"
    warm_up_steps: int = 8000
    dropout: float = 0.1
    pad_token_idx: Optional[int] = None  # Placeholder arg -> is set by the tokenizer
    vocabulary_size: Optional[int] = None  # Placeholder arg -> is set by the tokenizer

    # Classifier arguments
    num_hidden_nodes: Optional[List[int]] = field(default_factory=lambda: [])
    num_classes: Optional[int] = None  # placeholder arg -> is set by the datamodule
