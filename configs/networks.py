import dataclasses
from typing import Union, Literal

from configs.activations import ActivationConfig, FTA, Relu


@dataclasses.dataclass(frozen=True)
class BaseNetworkConfig:
    type: str = "unknown"


@dataclasses.dataclass(frozen=True)
class QNetworkCounts(BaseNetworkConfig):
    type: Literal["q_network_counts"] = "q_network_counts"
    hidden_size: int = 64
    learnable_norm_params: bool = False
    count_layer: int = 1

    # Network Activation Configs
    activation1: ActivationConfig = FTA()
    activation2: ActivationConfig = Relu()


NetworkConfig = Union[QNetworkCounts, BaseNetworkConfig]
