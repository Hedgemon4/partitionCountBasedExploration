import equinox as eqx
import jax
from jax import Array
import jax.numpy as jnp
from jax.nn import one_hot

from activations import make_activation


class QNetwork(eqx.Module):
    layers: list

    def __init__(self, input_size, num_actions, key, network_config):
        key1, key2, key3 = jax.random.split(key, 3)
        hidden_size = network_config.hidden_size
        learnable_norm_params = network_config.learnable_norm_params

        self.layers = [
            eqx.nn.Linear(in_features=input_size, out_features=hidden_size, key=key1),
            eqx.nn.LayerNorm(
                hidden_size,
                use_weight=learnable_norm_params,
                use_bias=learnable_norm_params,
            ),
            jax.nn.relu,
            eqx.nn.Linear(in_features=hidden_size, out_features=hidden_size, key=key2),
            eqx.nn.LayerNorm(
                hidden_size,
                use_weight=learnable_norm_params,
                use_bias=learnable_norm_params,
            ),
            jax.nn.relu,
            eqx.nn.Linear(in_features=hidden_size, out_features=num_actions, key=key3),
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def loss(self, states, actions, targets):
        q_values = jax.vmap(self)(states)
        index = jnp.arange(q_values.shape[0])
        selected_q_values = q_values[index, actions]
        return 0.5 * jnp.mean((selected_q_values - targets) ** 2), selected_q_values


class QNetworkCounts(eqx.Module):
    block1: list
    block2: list
    value_head: list
    counts: Array
    count_layer: int

    def __init__(self, input_size, num_actions, key, network_config):
        key1, key2, key3 = jax.random.split(key, 3)

        self.count_layer = network_config.count_layer

        # Instantiate both activation layers
        activation_layer_1 = make_activation(network_config.activation1)
        activation_layer_2 = make_activation(network_config.activation2)

        # Initialize Counts
        num_bins_1 = getattr(activation_layer_1, "num_bins", 1)
        num_bins_2 = getattr(activation_layer_2, "num_bins", 1)

        if self.count_layer == 1:
            number_of_discrete_states = num_bins_1
        elif self.count_layer == 2:
            number_of_discrete_states = num_bins_2
        else:
            raise ValueError(
                "Count layer must be either 1 or 2, indicating which activation layer to use for the discrete representation"
            )

        if number_of_discrete_states < 2:
            raise ValueError(
                "Count layer must have at least two bins to have a discrete representation"
            )

        hidden_size = network_config.hidden_size
        learnable_norm_params = network_config.learnable_norm_params

        self.counts = jnp.ones((num_actions, hidden_size, number_of_discrete_states))

        # Determine the width of the second linear layer's input.
        second_linear_width = num_bins_1 * hidden_size
        final_linear_width = num_bins_2 * hidden_size

        self.block1 = [
            eqx.nn.Linear(in_features=input_size, out_features=hidden_size, key=key1),
            eqx.nn.LayerNorm(
                hidden_size,
                use_weight=learnable_norm_params,
                use_bias=learnable_norm_params,
            ),
            activation_layer_1,
        ]
        self.block2 = [
            eqx.nn.Lambda(jnp.ravel),
            eqx.nn.Linear(
                in_features=second_linear_width,
                out_features=hidden_size,
                key=key2,
            ),
            eqx.nn.LayerNorm(
                hidden_size,
                use_weight=learnable_norm_params,
                use_bias=learnable_norm_params,
            ),
            activation_layer_2,
        ]
        self.value_head = [
            eqx.nn.Lambda(jnp.ravel),
            eqx.nn.Linear(
                in_features=final_linear_width, out_features=num_actions, key=key3
            ),
        ]

    def update_counts(self, discrete_states, actions):
        updated_counts = self.counts.at[actions].add(discrete_states)
        return eqx.tree_at(lambda m: m.counts, self, updated_counts)

    def get_intrinsic_reward(self, discrete_state, action):
        counts = self.counts * discrete_state
        counts = jnp.sum(counts, axis=-1)
        counts = jnp.min(counts, axis=-1)
        reward = jnp.sqrt(2 * jnp.log(jnp.sum(counts, axis=-1)) / counts[action])
        return reward

    def __call__(self, x):
        # Explicitly indicate counts are not trainable
        jax.lax.stop_gradient(self.counts)

        for layer in self.block1:
            x = layer(x)

        first_activation = x

        for layer in self.block2:
            x = layer(x)

        second_activation = x

        for layer in self.value_head:
            x = layer(x)

        # Depending on which layer is being used for counts, select the appropriate activation for the discrete representation
        discrete_activation = (
            first_activation if self.count_layer == 1 else second_activation
        )

        # If the left linear tile is active, then it will be negative so won't be chosen by argmax, but should be used as the one hot
        left_linear_active = discrete_activation[:, 0] < 0.0
        argmax = jnp.argmax(discrete_activation, axis=-1)
        # Either the left linear tile if active, or the argmax of the rest of the tiles
        final_indices = jnp.where(left_linear_active, 0, argmax)
        discrete_representation = one_hot(final_indices, discrete_activation.shape[-1])

        return x, discrete_representation

    def loss(self, states, actions, targets):
        q_values, _ = jax.vmap(self)(states)
        index = jnp.arange(q_values.shape[0])
        selected_q_values = q_values[index, actions]
        return 0.5 * jnp.mean((selected_q_values - targets) ** 2), selected_q_values


class QNetworkWithIntrinsicValueHead(eqx.Module):
    block1: list
    block2: list
    extrinsic_value_head: list
    intrinsic_value_head: list
    counts: Array
    count_layer: int

    def __init__(self, input_size, num_actions, key, network_config):
        key1, key2, key3, key4 = jax.random.split(key, 4)

        self.count_layer = network_config.count_layer

        # Instantiate both activation layers
        activation_layer_1 = make_activation(network_config.activation1)
        activation_layer_2 = make_activation(network_config.activation2)

        # Initialize Counts
        num_bins_1 = getattr(activation_layer_1, "num_bins", 1)
        num_bins_2 = getattr(activation_layer_2, "num_bins", 1)

        if self.count_layer == 1:
            number_of_discrete_states = num_bins_1
        elif self.count_layer == 2:
            number_of_discrete_states = num_bins_2
        else:
            raise ValueError(
                "Count layer must be either 1 or 2, indicating which activation layer to use for the discrete representation"
            )

        if number_of_discrete_states < 2:
            raise ValueError(
                "Count layer must have at least two bins to have a discrete representation"
            )

        hidden_size = network_config.hidden_size
        learnable_norm_params = network_config.learnable_norm_params

        self.counts = jnp.ones((num_actions, hidden_size, number_of_discrete_states))

        # Determine the width of the second linear layer's input.
        second_linear_width = num_bins_1 * hidden_size
        final_linear_width = num_bins_2 * hidden_size

        self.block1 = [
            eqx.nn.Linear(in_features=input_size, out_features=hidden_size, key=key1),
            eqx.nn.LayerNorm(
                hidden_size,
                use_weight=learnable_norm_params,
                use_bias=learnable_norm_params,
            ),
            activation_layer_1,
        ]
        self.block2 = [
            eqx.nn.Lambda(jnp.ravel),
            eqx.nn.Linear(
                in_features=second_linear_width,
                out_features=hidden_size,
                key=key2,
            ),
            eqx.nn.LayerNorm(
                hidden_size,
                use_weight=learnable_norm_params,
                use_bias=learnable_norm_params,
            ),
            activation_layer_2,
        ]
        self.extrinsic_value_head = [
            eqx.nn.Lambda(jnp.ravel),
            eqx.nn.Linear(
                in_features=final_linear_width, out_features=num_actions, key=key3
            ),
        ]
        self.intrinsic_value_head = [
            eqx.nn.Lambda(jnp.ravel),
            eqx.nn.Linear(
                in_features=final_linear_width, out_features=num_actions, key=key4
            ),
        ]

    def update_counts(self, discrete_states, actions):
        updated_counts = self.counts.at[actions].add(discrete_states)
        return eqx.tree_at(lambda m: m.counts, self, updated_counts)

    def get_intrinsic_reward(self, discrete_state, action):
        counts = self.counts * discrete_state
        counts = jnp.sum(counts, axis=-1)
        counts = jnp.min(counts, axis=-1)
        reward = jnp.sqrt(2 * jnp.log(jnp.sum(counts, axis=-1)) / counts[action])
        return reward

    def __call__(self, x):
        # Explicitly indicate counts are not trainable
        jax.lax.stop_gradient(self.counts)

        for layer in self.block1:
            x = layer(x)

        first_activation = x

        for layer in self.block2:
            x = layer(x)

        second_activation = x

        for layer in self.extrinsic_value_head:
            extrinsic_output = layer(x)

        for layer in self.intrinsic_value_head:
            intrinsic_output = layer(x)

        # Depending on which layer is being used for counts, select the appropriate activation for the discrete representation
        discrete_activation = (
            first_activation if self.count_layer == 1 else second_activation
        )

        # If the left linear tile is active, then it will be negative so won't be chosen by argmax, but should be used as the one hot
        left_linear_active = discrete_activation[:, 0] < 0.0
        argmax = jnp.argmax(discrete_activation, axis=-1)
        # Either the left linear tile if active, or the argmax of the rest of the tiles
        final_indices = jnp.where(left_linear_active, 0, argmax)
        discrete_representation = one_hot(final_indices, discrete_activation.shape[-1])

        return extrinsic_output, intrinsic_output, discrete_representation

    def loss(self, states, actions, extrinsic_targets, intrinsic_targets):
        extrinsic_q_values, intrinsic_q_values, _ = jax.vmap(self)(states)

        # Compute extrinsic loss
        index = jnp.arange(extrinsic_q_values.shape[0])
        selected_q_values = extrinsic_q_values[index, actions]
        extrinsic_loss = 0.5 * jnp.mean((selected_q_values - extrinsic_targets) ** 2)

        # Compute intrinsic loss
        index = jnp.arange(intrinsic_q_values.shape[0])
        selected_intrinsic_q_values = intrinsic_q_values[index, actions]
        intrinsic_loss = 0.5 * jnp.mean(
            (selected_intrinsic_q_values - intrinsic_targets) ** 2
        )

        total_loss = extrinsic_loss + intrinsic_loss

        return total_loss, selected_q_values
