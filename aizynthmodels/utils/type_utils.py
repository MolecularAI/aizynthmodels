""" Module containing type definitions """

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch

StrDict = Dict[str, Any]
RouteList = List[StrDict]
RouteDistancesCalculator = Callable[[RouteList], np.ndarray]
PredictionType = Union[List[List[str]], torch.Tensor]
TargetType = Optional[Union[List[str], torch.Tensor]]
