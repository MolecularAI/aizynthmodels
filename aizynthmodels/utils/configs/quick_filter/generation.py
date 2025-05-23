from dataclasses import dataclass, field
from typing import List


@dataclass
class NegativeReactionGenerator:
    """Config for aizynthmodels/quick_filter/generation.py"""

    type: List[str] = field(default_factory=lambda:["strict", "random"])
    random_samples: int = 1000
    random_state: int = 1984
