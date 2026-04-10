import dataclasses
from typing import Union, Literal


@dataclasses.dataclass(frozen=True)
class BaseActivationConfig:
    type: str = "unknown"


@dataclasses.dataclass(frozen=True)
class FTA(BaseActivationConfig):
    type: Literal["fta"] = "fta"
    bound: float = 1.0
    eta: float = 0.25
    static_centres: bool = True


@dataclasses.dataclass(frozen=True)
class FTAOriginal(FTA):
    bound: float = 20.0
    eta: float = 2.00
    static_centres: bool = True


@dataclasses.dataclass(frozen=True)
class Elephant(BaseActivationConfig):
    type: Literal["elephant"] = "elephant"
    a: float = 1.0
    h: float = 1.0
    d: float = 2.0


@dataclasses.dataclass(frozen=True)
class Relu(BaseActivationConfig):
    type: Literal["relu"] = "relu"


ActivationConfig = Union[FTA, FTAOriginal, Elephant, Relu]
