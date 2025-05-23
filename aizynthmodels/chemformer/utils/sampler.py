from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from typing import List

    from aizynthmodels.chemformer.utils.samplers.sampler_nodes import BaseSamplerNode


class Criterion:
    def __call__(self, node: BaseSamplerNode) -> bool:
        raise NotImplementedError("Not implemented")


class MaxLength(Criterion):
    def __init__(self, max_length: int) -> None:
        super(MaxLength, self).__init__()
        self.max_length = max_length

    def __call__(self, node: BaseSamplerNode) -> bool:
        return node.pos >= self.max_length - 1


class EOS(Criterion):
    def __init__(self) -> None:
        super(EOS, self).__init__()

    def __call__(self, node: BaseSamplerNode) -> bool:
        return torch.all(node.ll_mask).item()


class LogicalAnd(Criterion):
    def __init__(self, criteria: List[Criterion]) -> None:
        super(LogicalAnd, self).__init__()
        self.criteria = criteria

    def __call__(self, node: BaseSamplerNode) -> bool:
        return all([c(node) for c in self.criteria])


class LogicalOr(Criterion):
    def __init__(self, criteria: List[Criterion]) -> None:
        super(LogicalOr, self).__init__()
        self.criteria = criteria

    def __call__(self, node: BaseSamplerNode) -> bool:
        return any([c(node) for c in self.criteria])
