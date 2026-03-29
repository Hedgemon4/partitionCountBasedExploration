import chex

from configs.activations import ActivationConfig, FTAMountainCarConfig
from dataclasses import field
from configs.activations import FTAConfig, ReLUConfig


@chex.dataclass(frozen=True)
class PQNOriginalCartpoleConfig:
    seed: int = 0
    num_seeds: int = 30
    initial_learning_rate: float = 0.0001
    final_learning_rate: float = 1e-20
    environment: str = "CartPole-v1"
    num_environments: int = 32
    num_steps: int = 64
    total_time_steps: int = 5e5
    epsilon_start: float = 1.0
    epsilon_end: float = 0.2
    epsilon_decay: float = 0.2
    num_epochs: int = 4
    num_minibatches: int = 16
    hidden_size: int = 256
    gamma: float = 0.99
    lambda_returns: bool = True
    lam: float = 0.95
    max_grad_norm: float = 10
    reward_scale: float = 0.1
    num_episodes_for_average: int = 30
    learnable_norm_params: bool = True
    sarsa_returns: bool = False
    metrics_file_name: str = "pqn_original_cartpole_default_params.npz"


@chex.dataclass(frozen=True)
class PQNCartpoleConfig:
    seed: int = 0
    num_seeds: int = 30
    initial_learning_rate: float = 0.0001
    final_learning_rate: float = 1e-20
    environment: str = "CartPole-v1"
    num_environments: int = 32
    num_steps: int = 64
    total_time_steps: int = 5e5
    epsilon_start: float = 1.0
    epsilon_end: float = 0.2
    epsilon_decay: float = 0.2
    num_epochs: int = 4
    num_minibatches: int = 16
    hidden_size: int = 256
    gamma: float = 0.99
    lambda_returns: bool = True
    lam: float = 0.95
    max_grad_norm: float = 10
    reward_scale: float = 0.1
    num_episodes_for_average: int = 30
    learnable_norm_params: bool = True
    sarsa_returns: bool = True
    metrics_file_name: str = "pqn_cartpole_with_sarsa_default_params.npz"


@chex.dataclass(frozen=True)
class DefaultMountainCarConfig:
    seed: int = 0
    num_seeds: int = 30
    initial_learning_rate: float = 0.0001
    final_learning_rate: float = 1e-20
    environment: str = "MountainCar-v0"
    num_environments: int = 32
    num_steps: int = 64
    total_time_steps: int = 5e5
    epsilon_start: float = 1.0
    epsilon_end: float = 0.2
    epsilon_decay: float = 0.2
    num_epochs: int = 4
    num_minibatches: int = 16
    hidden_size: int = 256
    gamma: float = 0.99
    lambda_returns: bool = True
    lam: float = 0.95
    max_grad_norm: float = 10
    reward_scale: float = 0.1
    num_episodes_for_average: int = 30
    learnable_norm_params: bool = True
    sarsa_returns: bool = False
    bound: float = 20.0
    eta: float = 2.0
    metrics_folder_name: str = "pqn_mountaincar_with_fta"

    # Network Activation Configs
    act_1: ActivationConfig = field(default_factory=FTAMountainCarConfig)
    act_2: ActivationConfig = field(default_factory=ReLUConfig)

@chex.dataclass(frozen=True)
class CartPoleWithFTAConfig:
    seed: int = 0
    num_seeds: int = 30
    initial_learning_rate: float = 0.0001
    final_learning_rate: float = 1e-20
    environment: str = "CartPole-v1"
    num_environments: int = 32
    num_steps: int = 64
    total_time_steps: int = 5e5
    epsilon_start: float = 1.0
    epsilon_end: float = 0.2
    epsilon_decay: float = 0.2
    num_epochs: int = 4
    num_minibatches: int = 16
    hidden_size: int = 256
    gamma: float = 0.99
    lambda_returns: bool = True
    lam: float = 0.95
    max_grad_norm: float = 10
    reward_scale: float = 0.1
    num_episodes_for_average: int = 30
    learnable_norm_params: bool = False
    sarsa_returns: bool = True
    metrics_folder_name: str = "pqn_cartpole_with_fta"

    # Network Activation Configs
    act_1: ActivationConfig = field(default_factory=FTAMountainCarConfig)
    act_2: ActivationConfig = field(default_factory=ReLUConfig)

