import dataclasses
from typing import Union, Literal
import jax.nn
import equinox as eqx


@dataclasses.dataclass(frozen=True)
class ActivationConfig:
    type: str = "unknown"

@dataclasses.dataclass(frozen=True)
class FTAConfig(ActivationConfig):
    type: Literal["fta"] = "fta"
    bound: float = 20.0
    eta: float = 2.0
    static_centres: bool = True

@dataclasses.dataclass(frozen=True)
class FTAMountainCarConfig(FTAConfig):
    bound: float = 1.0
    eta: float = 0.1
    static_centres: bool = True

@dataclasses.dataclass(frozen=True)
class ElephantConfig(ActivationConfig):
    type: Literal["elephant"] = "elephant"
    a: float = 1.0
    h: float = 1.0
    d: float = 2.0

@dataclasses.dataclass(frozen=True)
class ReLUConfig(ActivationConfig):
    type: Literal["relu"] = "relu"
