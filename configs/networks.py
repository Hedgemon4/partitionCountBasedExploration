import dataclasses
from typing import Union, Literal

from configs.activations import ActivationConfig, FTA, Relu


@dataclasses.dataclass(frozen=True)
class BaseNetworkConfig:
    type: str = "unknown"


@dataclasses.dataclass(frozen=True)
class Block:
    hidden_size: int = 64
    learnable_norm_params: bool = True
    activation: ActivationConfig = Relu()


@dataclasses.dataclass(frozen=True)
class FTABlock(Block):
    activation: ActivationConfig = FTA()


@dataclasses.dataclass(frozen=True)
class QNetwork(BaseNetworkConfig):
    type: Literal["q_network"] = "q_network"
    blocks: list[Block] = dataclasses.field(default_factory=lambda: [Block(), Block()])


@dataclasses.dataclass(frozen=True)
class QNetworkCartpole(BaseNetworkConfig):
    type: Literal["q_network"] = "q_network"
    blocks: list[Block] = dataclasses.field(
        default_factory=lambda: [Block(hidden_size=256), Block(hidden_size=256)]
    )


@dataclasses.dataclass(frozen=True)
class QNetworkCounts(BaseNetworkConfig):
    type: Literal["q_network_counts"] = "q_network_counts"
    count_layer: int = 1
    blocks: list[Block] = dataclasses.field(
        default_factory=lambda: [FTABlock(), Block()]
    )


@dataclasses.dataclass(frozen=True)
class QNetworkCountsWithNextStatePrediction(BaseNetworkConfig):
    type: Literal["q_network_counts_with_next_state_prediction"] = (
        "q_network_counts_with_next_state_prediction"
    )
    count_layer: int = 1
    next_state_coef: float = 1.0
    blocks: list[Block] = dataclasses.field(
        default_factory=lambda: [FTABlock(), Block()]
    )


@dataclasses.dataclass(frozen=True)
class CNNNetworkConfig(BaseNetworkConfig):
    type: Literal["cnn_network"] = "cnn_network"
    padding: str = "VALID"


@dataclasses.dataclass(frozen=True)
class QNetworkCNNCountsConfig(CNNNetworkConfig):
    type: Literal["q_network_cnn_counts"] = "q_network_cnn_counts"
    count_layer: int = 1
    next_state_coef: float = 0.0
    blocks: list[Block] = dataclasses.field(
        default_factory=lambda: [FTABlock(hidden_size=512), Block(hidden_size=512)]
    )


@dataclasses.dataclass(frozen=True)
class QNetworkCNNConfig(CNNNetworkConfig):
    type: Literal["q_network_cnn"] = "q_network_cnn"
    blocks: list[Block] = dataclasses.field(
        default_factory=lambda: [Block(hidden_size=512)]
    )


NetworkConfig = Union[
    QNetwork,
    QNetworkCartpole,
    QNetworkCounts,
    QNetworkCountsWithNextStatePrediction,
    QNetworkCNNCountsConfig,
    QNetworkCNNConfig,
    BaseNetworkConfig,
]
