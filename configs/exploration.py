import chex

@chex.dataclass(frozen=True)
class ExplorationConfig:
    beta: float = 0.1
    count_layer: int = 1
