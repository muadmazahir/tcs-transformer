"""TCS-Transformer Data Module"""

from .dataset import (
    TrainDataset,
    EvalRelpronDataset,
    EvalWeeds2014Dataset,
    TCSTransformerDataset,
)

from .collators import (
    PASTruthCollator,
    MyTruthCollator,
    EvalRelpronPASCollator,
    TCSTransformerCollator,
)

__all__ = [
    # Datasets
    "TrainDataset",
    "EvalRelpronDataset",
    "EvalWeeds2014Dataset",
    "TCSTransformerDataset",
    # Collators
    "PASTruthCollator",
    "MyTruthCollator",
    "EvalRelpronPASCollator",
    "TCSTransformerCollator",
]
