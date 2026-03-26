import dataclasses
from typing import Union, Literal
import jax.nn
import equinox as eqx


@dataclasses.dataclass(frozen=True)
class FTAConfig:
    type: Literal["fta"] = "fta"
    bound: float = 20.0
    eta: float = 2.0
    static_centres: bool = True


@dataclasses.dataclass(frozen=True)
class ElephantConfig:
    type: Literal["elephant"] = "elephant"
    a: float = 1.0
    h: float = 1.0
    d: float = 2.0


@dataclasses.dataclass(frozen=True)
class ReLUConfig:
    type: Literal["relu"] = "relu"


ActivationConfig = Union[FTAConfig, ElephantConfig, ReLUConfig]
