import chex

from configs.activations import ActivationConfig
from dataclasses import field
from configs.activations import FTA, Relu
from configs.networks import (
    QNetworkCounts,
    NetworkConfig,
    QNetworkMountainCarCounts,
    QNetworkDoorKeyCounts,
    QNetwork,
)


@chex.dataclass(frozen=True)
class BaseConfig:
    # Experiment Configs
    seed: int = 0
    num_seeds: int = 30
    num_episodes_for_average: int = 30


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
class PQNCartpoleConfig(BaseConfig):
    # Experiment Configs
    output_folder_name: str = "pqn_cartpole"

    # Algorithm Configs
    initial_learning_rate: float = 0.0001
    final_learning_rate: float = 1e-20
    num_epochs: int = 4
    num_minibatches: int = 16
    gamma: float = 0.99
    lambda_returns: bool = True
    lam: float = 0.95
    max_grad_norm: float = 10.0
    reward_scale: float = 0.1
    sarsa_returns: bool = True

    # Env Configs
    environment: str = "CartPole-v1"
    num_environments: int = 32
    num_steps: int = 64
    total_time_steps: float = 5e5

    # Exploration Configs
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.2

    # Network Activation Configs
    network: NetworkConfig = QNetwork()


@chex.dataclass(frozen=True)
class PQNMountainCarConfig(BaseConfig):
    # Experiment Configs
    output_folder_name: str = "pqn_mountaincar"

    # Algorithm Configs
    initial_learning_rate: float = 0.004
    final_learning_rate: float = 0.004
    num_epochs: int = 8
    num_minibatches: int = 16
    gamma: float = 0.99
    lambda_returns: bool = True
    lam: float = 0.95
    max_grad_norm: float = 100.0
    reward_scale: float = 1.0
    sarsa_returns: bool = True

    # Env Configs
    environment: str = "MountainCar-v0"
    episode_length: int = 200
    num_environments: int = 32
    num_steps: int = 64
    total_time_steps: float = 5e5

    # Exploration Configs
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.2

    # Network Activation Configs
    network: NetworkConfig = QNetwork()


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
    act_1: ActivationConfig = field(default_factory=FTA)
    act_2: ActivationConfig = field(default_factory=Relu)


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
    hidden_size: int = 128
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
    act_1: ActivationConfig = FTA()
    act_2: ActivationConfig = Relu()


@chex.dataclass(frozen=True)
class CartPoleWithIntrinsicRewardsConfig(BaseConfig):
    # Experiment Configs
    output_folder_name: str = "pqn_cartpole_with_intrinsic_rewards"

    # Algorithm Configs
    initial_learning_rate: float = 0.0001
    final_learning_rate: float = 1e-20
    num_epochs: int = 4
    num_minibatches: int = 16
    gamma: float = 0.99
    lambda_returns: bool = True
    lam: float = 0.95
    max_grad_norm: float = 10
    reward_scale: float = 0.1
    sarsa_returns: bool = True

    # Env Configs
    environment: str = "CartPole-v1"
    num_environments: int = 32
    num_steps: int = 64
    total_time_steps: float = 5e5

    # Exploration Configs
    beta: float = 0.1
    epsilon_start: float = 1.0
    epsilon_end: float = 0.2
    epsilon_decay: float = 0.2

    # Network Activation Configs
    network: NetworkConfig = QNetworkCounts()


@chex.dataclass(frozen=True)
class MountainCarWithIntrinsicRewardsConfig(BaseConfig):
    # Experiment Configs
    output_folder_name: str = "pqn_same_epsilon_testing"

    # Algorithm Configs
    initial_learning_rate: float = 0.004
    final_learning_rate: float = 0.004
    num_epochs: int = 8
    num_minibatches: int = 16
    gamma: float = 0.99
    lambda_returns: bool = True
    lam: float = 0.95
    max_grad_norm: float = 100.0
    reward_scale: float = 1.0
    sarsa_returns: bool = True
    pessimistic: bool = False

    # Env Configs
    environment: str = "MountainCar-v0"
    episode_length: int = 200
    num_environments: int = 32
    num_steps: int = 64
    total_time_steps: float = 5e5

    # Exploration Configs
    beta: float = 0.1
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.2

    # Network Activation Configs
    network: NetworkConfig = QNetworkMountainCarCounts()


@chex.dataclass(frozen=True)
class DoorKeyWithIntrinsicRewardsConfig(BaseConfig):
    # Experiment Configs
    output_folder_name: str = "pqn_doorkey_with_intrinsic_rewards"

    # Algorithm Configs
    initial_learning_rate: float = 0.0001
    final_learning_rate: float = 1e-20
    num_epochs: int = 4
    num_minibatches: int = 16
    gamma: float = 0.99
    lambda_returns: bool = True
    lam: float = 0.95
    max_grad_norm: float = 10.0
    reward_scale: float = 1.0
    sarsa_returns: bool = True

    # Env Configs — change to Navix-DoorKey-6x6-v0 or 8x8 for harder variants
    environment: str = "Navix-DoorKey-8x8-v0"
    num_environments: int = 32
    num_steps: int = 128
    total_time_steps: float = 2e6

    # Exploration Configs — higher beta for sparse-reward exploration
    beta: float = 0.5
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.3

    # Network Activation Configs
    network: NetworkConfig = QNetworkDoorKeyCounts()
