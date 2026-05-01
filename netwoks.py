import equinox as eqx
import jax
from jax import Array
import jax.numpy as jnp
from jax.nn import one_hot

from activations import make_activation
from configs.networks import (
    QNetwork as QNetworkConfig,
    QNetworkCartpole as QNetworkCartpoleConfig,
    QNetworkCounts as QNetworkCountsConfig,
    QNetworkCountsWithNextStatePrediction as QNetworkCountsWithNextStatePredictionConfig,
)


class QNetwork(eqx.Module):
    layers: list
    num_bins: int = 10

    def __init__(self, input_size, num_actions, key, network_config):
        self.layers = []
        blocks = network_config.blocks
        keys = jax.random.split(key, len(blocks) + 1)

        input_features = input_size
        previous_bins = 1
        for i, block in enumerate(blocks):
            hidden_size = block.hidden_size
            learnable_norm_params = block.learnable_norm_params

            # We need to flatten the output of the activation if there were multiple bins in the previous layer, since each bin will be treated as a separate feature for the next layer
            if previous_bins > 1:
                self.layers.append(eqx.nn.Lambda(jnp.ravel))

            self.layers.append(
                eqx.nn.Linear(
                    in_features=input_features, out_features=hidden_size, key=keys[i]
                )
            )

            self.layers.append(
                eqx.nn.LayerNorm(
                    hidden_size,
                    use_weight=learnable_norm_params,
                    use_bias=learnable_norm_params,
                )
            )
            activation = make_activation(block.activation)
            num_bins = getattr(activation, "num_bins", 1)

            self.layers.append(activation)

            # Compute the number of input features for the next layer, which will be the hidden size times the number of bins for the current activation
            input_features = hidden_size * num_bins
            previous_bins = num_bins

        self.layers.append(
            eqx.nn.Linear(
                in_features=blocks[-1].hidden_size,
                out_features=num_actions,
                key=keys[-1],
            )
        )

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def loss(self, mini_batch, targets):
        q_values = jax.vmap(self)(mini_batch.state)
        index = jnp.arange(q_values.shape[0])
        selected_q_values = q_values[index, mini_batch.action]

        q_loss = 0.5 * jnp.mean((selected_q_values - targets) ** 2)

        losses = {"total": q_loss, "q": q_loss}
        return q_loss, (selected_q_values, losses)


class QNetworkCounts(eqx.Module):
    blocks: list
    extrinsic_value_head: list
    counts: Array
    count_layer: int
    num_bins: int

    def __init__(self, input_size, num_actions, key, network_config):
        self.blocks = []
        self.count_layer = network_config.count_layer
        blocks = network_config.blocks
        keys = jax.random.split(key, len(blocks) + 1)
        number_of_discrete_states = 0

        input_features = input_size
        previous_bins = 1
        for i, block in enumerate(blocks):
            layer = []
            hidden_size = block.hidden_size
            learnable_norm_params = block.learnable_norm_params

            # We need to flatten the output of the activation if there were multiple bins in the previous layer, since each bin will be treated as a separate feature for the next layer
            if previous_bins > 1:
                layer.append(eqx.nn.Lambda(jnp.ravel))

            layer.append(
                eqx.nn.Linear(
                    in_features=input_features, out_features=hidden_size, key=keys[i]
                )
            )

            layer.append(
                eqx.nn.LayerNorm(
                    hidden_size,
                    use_weight=learnable_norm_params,
                    use_bias=learnable_norm_params,
                )
            )
            activation = make_activation(block.activation)
            num_bins = getattr(activation, "num_bins", 1)

            if self.count_layer == i + 1:
                # This will be the layer which outputs the discrete representation, so we need to get the bin size
                number_of_discrete_states = num_bins
                self.num_bins = num_bins
                if number_of_discrete_states < 2:
                    raise ValueError(
                        "Count layer must have at least two bins to have a discrete representation"
                    )
            layer.append(activation)

            self.blocks.append(layer)

            # Compute the number of input features for the next layer, which will be the hidden size times the number of bins for the current activation
            input_features = hidden_size * num_bins
            previous_bins = num_bins

        if number_of_discrete_states == 0:
            raise ValueError(
                "Count layer must be set to a valid block number to have a discrete representation for counts"
            )

        self.counts = jnp.ones(
            (
                num_actions,
                blocks[self.count_layer - 1].hidden_size,
                number_of_discrete_states,
            )
        )

        self.extrinsic_value_head = [
            eqx.nn.Linear(
                in_features=blocks[-1].hidden_size,
                out_features=num_actions,
                key=keys[-1],
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

        for i, block in enumerate(self.blocks):
            for layer in block:
                x = layer(x)
            # Depending on which layer is being used for counts, select the appropriate activation for the discrete representation
            if i + 1 == self.count_layer:
                discrete_activation = x

        for layer in self.extrinsic_value_head:
            x = layer(x)

        discrete_representation = self._discrete_representation(discrete_activation)

        return x, discrete_representation

    def _discrete_representation(self, discrete_activation):
        # If the left linear tile is active, then it will be negative so won't be chosen by argmax, but should be used as the one hot
        left_linear_active = discrete_activation[:, 0] < 0.0
        argmax = jnp.argmax(discrete_activation, axis=-1)
        # Either the left linear tile if active, or the argmax of the rest of the tiles
        final_indices = jnp.where(left_linear_active, 0, argmax)
        discrete_representation = one_hot(final_indices, discrete_activation.shape[-1])
        return discrete_representation

    def get_discrete_representation(self, states):
        x = states
        for i, block in enumerate(self.blocks):
            for layer in block:
                x = layer(x)
            # Depending on which layer is being used for counts, select the appropriate activation for the discrete representation
            if i + 1 == self.count_layer:
                discrete_activation = x
                break

        return self._discrete_representation(jax.lax.stop_gradient(discrete_activation))

    def loss(self, mini_batch, targets):
        q_values, _ = jax.vmap(self)(mini_batch.state)
        index = jnp.arange(q_values.shape[0])
        selected_q_values = q_values[index, mini_batch.action]

        q_loss = 0.5 * jnp.mean((selected_q_values - targets) ** 2)

        losses = {"total": q_loss, "q": q_loss}
        return q_loss, (selected_q_values, losses)


class QNetworkCountsSeparateValueHeads(QNetworkCounts):
    intrinsic_value_head: list

    def __init__(self, input_size, num_actions, key, network_config):
        super_key, key = jax.random.split(key)
        super().__init__(input_size, num_actions, super_key, network_config)




class QNetworkCountsWithNextStatePrediction(QNetworkCounts):
    next_state_head: list
    next_state_coef: float

    def __init__(self, input_size, num_actions, key, network_config):
        super_key, key = jax.random.split(key)
        super().__init__(input_size, num_actions, super_key, network_config)
        blocks = network_config.blocks

        # Predicts the next state (same dimensionality as the input observation)
        # from the shared trunk representation.
        self.next_state_head = [
            eqx.nn.Linear(
                in_features=blocks[-1].hidden_size,
                out_features=input_size,
                key=key,
            ),
        ]

    def __call__(self, x):
        # Explicitly indicate counts are not trainable
        jax.lax.stop_gradient(self.counts)

        for i, block in enumerate(self.blocks):
            for layer in block:
                x = layer(x)
            # Depending on which layer is being used for counts, select the appropriate activation for the discrete representation
            if i + 1 == self.count_layer:
                discrete_activation = x

        shared_output = x

        for layer in self.value_head:
            x = layer(x)

        predicted_next_state = shared_output
        for layer in self.next_state_head:
            predicted_next_state = layer(predicted_next_state)

        discrete_representation = self._discrete_representation(discrete_activation)

        return x, discrete_representation, predicted_next_state

    def loss(self, mini_batch, targets):
        q_values, _, predicted_next_states = jax.vmap(self)(mini_batch.state)
        index = jnp.arange(q_values.shape[0])
        selected_q_values = q_values[index, mini_batch.action]

        q_loss = 0.5 * jnp.mean((selected_q_values - targets) ** 2)

        next_state_loss = 0.5 * jnp.mean(
            (predicted_next_states - jax.lax.stop_gradient(mini_batch.next_state)) ** 2
        )

        total_loss = q_loss + self.next_state_coef * next_state_loss

        losses = {
            "total": total_loss,
            "q": q_loss,
            "next_state": next_state_loss,
        }
        return total_loss, (selected_q_values, losses)


def make_network(input_size, num_actions, key, network_config):
    """Build the network corresponding to `network_config`.

    Mirrors `make_activation` — dispatches on the config dataclass type so the
    caller doesn't need to hardcode a class.
    """
    if isinstance(network_config, (QNetworkConfig, QNetworkCartpoleConfig)):
        return QNetwork(
            input_size=input_size,
            num_actions=num_actions,
            key=key,
            network_config=network_config,
        )
    elif isinstance(network_config, QNetworkCountsWithNextStatePredictionConfig):
        return QNetworkCountsWithNextStatePrediction(
            input_size=input_size,
            num_actions=num_actions,
            key=key,
            network_config=network_config,
        )
    elif isinstance(network_config, QNetworkCountsConfig):
        return QNetworkCounts(
            input_size=input_size,
            num_actions=num_actions,
            key=key,
            network_config=network_config,
        )
    raise ValueError(f"Unknown network config: {type(network_config)}")
