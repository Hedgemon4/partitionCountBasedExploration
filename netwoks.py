import equinox as eqx
import jax
from jax import Array
import jax.numpy as jnp
from jax.nn import one_hot

from layers import make_activation, ChannelsLayerNorm
from configs.networks import (
    QNetwork as QNetworkConfig,
    QNetworkCartpole as QNetworkCartpoleConfig,
    QNetworkCounts as QNetworkCountsConfig,
    QNetworkCountsWithNextStatePrediction as QNetworkCountsWithNextStatePredictionConfig,
    QNetworkCNNCountsConfig,
)


class QNetwork(eqx.Module):
    blocks: list
    value_head: list
    num_bins: int

    def __init__(self, input_size, num_actions, key, network_config):
        self.blocks = []
        self.num_bins = 10
        blocks = network_config.blocks
        keys = jax.random.split(key, len(blocks) + 1)

        input_features = input_size
        previous_bins = 1
        for i, block in enumerate(blocks):
            hidden_size = block.hidden_size
            learnable_norm_params = block.learnable_norm_params

            # We need to flatten the output of the activation if there were multiple bins in the previous layer, since each bin will be treated as a separate feature for the next layer
            if previous_bins > 1:
                self.blocks.append(eqx.nn.Lambda(jnp.ravel))

            self.blocks.append(
                eqx.nn.Linear(
                    in_features=input_features, out_features=hidden_size, key=keys[i]
                )
            )

            self.blocks.append(
                eqx.nn.LayerNorm(
                    hidden_size,
                    use_weight=learnable_norm_params,
                    use_bias=learnable_norm_params,
                )
            )
            activation = make_activation(block.activation)
            num_bins = getattr(activation, "num_bins", 1)
            self.blocks.append(activation)

            # Compute the number of input features for the next layer, which will be the hidden size times the number of bins for the current activation
            input_features = hidden_size * num_bins
            previous_bins = num_bins

        value_head_input_size = blocks[-1].hidden_size
        if previous_bins > 1:
            self.blocks.append(eqx.nn.Lambda(jnp.ravel))
            value_head_input_size *= previous_bins

        self.value_head = [
            eqx.nn.Linear(
                in_features=value_head_input_size,
                out_features=num_actions,
                key=keys[-1],
            )
        ]

    def __call__(self, x):
        for layer in self.blocks:
            x = layer(x)
        for layer in self.value_head:
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
    value_head: list
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

        self.value_head = [
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

        for layer in self.value_head:
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


class QNetworkCountsWithNextStatePrediction(QNetworkCounts):
    next_state_head: list
    next_state_coef: float

    def __init__(self, input_size, num_actions, key, network_config):
        self.blocks = []
        self.count_layer = network_config.count_layer
        self.next_state_coef = network_config.next_state_coef
        blocks = network_config.blocks
        # +2 keys: one for value_head, one for next_state_head
        keys = jax.random.split(key, len(blocks) + 2)
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

        self.value_head = [
            eqx.nn.Linear(
                in_features=blocks[-1].hidden_size,
                out_features=num_actions,
                key=keys[-2],
            ),
        ]

        # Predicts the next state (same dimensionality as the input observation)
        # from the shared trunk representation.
        self.next_state_head = [
            eqx.nn.Linear(
                in_features=blocks[-1].hidden_size,
                out_features=input_size,
                key=keys[-1],
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


class QNetworkCNNCounts(QNetworkCounts):
    cnn: list
    next_state_head: list
    next_state_coef: float

    def __init__(self, input_size, num_actions, key, network_config):
        # Need to change input size
        keys = jax.random.split(key, 5)
        if network_config.padding == "VALID":
            network_input = 3136
        elif network_config.padding == "SAME":
            network_input = 7744
        else:
            raise ValueError("Unknown padding type")

        super().__init__(network_input, num_actions, keys[0], network_config)
        self.next_state_coef = network_config.next_state_coef
        print(f"Input size: {input_size}")

        self.cnn = [
            eqx.nn.Conv2d(
                in_channels=input_size,
                out_channels=32,
                kernel_size=(8, 8),
                stride=(4, 4),
                padding=network_config.padding,
                key=keys[1],
            ),
            ChannelsLayerNorm(32),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=network_config.padding,
                key=keys[2],
            ),
            ChannelsLayerNorm(64),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=network_config.padding,
                key=keys[3],
            ),
            ChannelsLayerNorm(64),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Lambda(jnp.ravel),
        ]
        blocks = network_config.blocks

        discrete_representation_block = blocks[network_config.count_layer - 1]
        activation = make_activation(discrete_representation_block.activation)
        count_hidden = blocks[self.count_layer - 1].hidden_size

        # The head predicts the next state's continuous FTA features. It mirrors the
        # encoder's count block (Linear -> LayerNorm -> FTA) so the predicted features
        # and the target features live in the same regime for the MSE.
        self.next_state_head = [
            eqx.nn.Linear(
                in_features=blocks[-1].hidden_size,
                out_features=count_hidden,
                key=keys[4],
            ),
            eqx.nn.LayerNorm(
                count_hidden,
                use_weight=discrete_representation_block.learnable_norm_params,
                use_bias=discrete_representation_block.learnable_norm_params,
            ),
            activation,
        ]

    def __call__(self, x):
        # Explicitly indicate counts are not trainable
        jax.lax.stop_gradient(self.counts)
        # Change from (batch, channels, height, width) to (batch, height, width, channels) for eqx.nn.Conv2d
        x = x / 255.0

        for layer in self.cnn:
            x = layer(x)

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

        # Return order: q-values, one-hot bins (for counts), continuous count-layer FTA
        # features (auxiliary target for the next-state forward model), and the head's
        # prediction of the next state's continuous features.
        return x, discrete_representation, discrete_activation, predicted_next_state

    def get_discrete_representation(self, states):
        x = states / 255.0
        for layer in self.cnn:
            x = layer(x)

        for i, block in enumerate(self.blocks):
            for layer in block:
                x = layer(x)
            # Depending on which layer is being used for counts, select the appropriate activation for the discrete representation
            if i + 1 == self.count_layer:
                discrete_activation = x
                break

        return self._discrete_representation(jax.lax.stop_gradient(discrete_activation))

    def loss(self, mini_batch, targets):
        ### Auxiliary task: predict the next state's continuous count-layer FTA features
        ### from the current state's trunk (a one-step forward model in feature space).
        q_values, _, _, predicted_next_features = jax.vmap(self)(mini_batch.state)
        index = jnp.arange(q_values.shape[0])
        selected_q_values = q_values[index, mini_batch.action]

        q_loss = 0.5 * jnp.mean((selected_q_values - targets) ** 2)

        # Target: continuous FTA features of the next state, captured during the rollout.
        # stop_gradient keeps it a fixed (rollout-time) target, like a lightweight target
        # network; only the prediction side carries gradient into the encoder.
        target = jax.lax.stop_gradient(mini_batch.next_continuous_state)
        per_example = 0.5 * jnp.mean(
            (predicted_next_features - target) ** 2, axis=(-1, -2)
        )
        # Mask transitions whose successor is a fresh episode reset, not a real next state.
        mask = 1.0 - mini_batch.done
        next_state_loss = jnp.sum(per_example * mask) / jnp.clip(
            jnp.sum(mask), a_min=1.0
        )

        total_loss = q_loss + self.next_state_coef * next_state_loss

        losses = {
            "total": total_loss,
            "q": q_loss,
            "next_state": next_state_loss,
        }
        return total_loss, (selected_q_values, losses)


class QNetworkCNN(QNetwork):
    cnn: list

    def __init__(self, input_size, num_actions, key, network_config):
        # Need to change input size
        keys = jax.random.split(key, 4)
        if network_config.padding == "VALID":
            network_input = 3136
        elif network_config.padding == "SAME":
            network_input = 7744
        else:
            raise ValueError("Unknown padding type")

        super().__init__(network_input, num_actions, keys[0], network_config)

        self.cnn = [
            eqx.nn.Conv2d(
                in_channels=input_size,
                out_channels=32,
                kernel_size=(8, 8),
                stride=(4, 4),
                padding=network_config.padding,
                key=keys[1],
            ),
            ChannelsLayerNorm(32),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=network_config.padding,
                key=keys[2],
            ),
            ChannelsLayerNorm(64),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=network_config.padding,
                key=keys[3],
            ),
            ChannelsLayerNorm(64),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Lambda(jnp.ravel),
        ]

    def __call__(self, x):
        x = x / 255.0

        for layer in self.cnn:
            x = layer(x)

        for layer in self.blocks:
            x = layer(x)

        for layer in self.value_head:
            x = layer(x)

        return x

    def loss(self, mini_batch, targets):
        q_values = jax.vmap(self)(mini_batch.state)
        index = jnp.arange(q_values.shape[0])
        selected_q_values = q_values[index, mini_batch.action]

        q_loss = 0.5 * jnp.mean((selected_q_values - targets) ** 2)

        total_loss = q_loss

        losses = {
            "total": total_loss,
            "q": q_loss,
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
    elif isinstance(network_config, QNetworkCNNCountsConfig):
        return QNetworkCNNCounts(
            input_size=input_size,
            num_actions=num_actions,
            key=key,
            network_config=network_config,
        )
    raise ValueError(f"Unknown network config: {type(network_config)}")
