import numpy as np
import jax.numpy as jnp


class Counts:
    def __init__(self, num_actions, hidden_size=1, num_bins=1, beta=1, init_count=1, mode='state_action'):
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.num_bins = num_bins
        self.beta = beta
        self.mode = mode
        self.init_count = init_count
        if mode == 'state_action':
            self._checkpoint_counts = np.zeros((self.num_actions, self.hidden_size, self.num_bins),
                                               dtype=np.int32) + init_count
        elif mode == 'state':
            self._checkpoint_counts = np.zeros((self.hidden_size, self.num_bins), dtype=np.int32) + init_count
        else:
            raise ValueError(f"Unknown mode {mode} for Counts module.")

    def reset_counts(self):
        if self.mode == 'state_action':
            self._checkpoint_counts = np.zeros((self.num_actions, self.hidden_size, self.num_bins),
                                               dtype=np.int32) + self.init_count
        elif self.mode == 'state':
            self._checkpoint_counts = np.zeros((self.hidden_size, self.num_bins), dtype=np.int32) + self.init_count

    def initial(self):
        if self.mode == 'state_action':
            return jnp.zeros((self.num_actions, self.hidden_size, self.num_bins), dtype=jnp.int32) + self.init_count
        elif self.mode == 'state':
            return jnp.zeros((self.hidden_size, self.num_bins), dtype=jnp.int32) + self.init_count

    def counts_add(self, step, worker=0):
        step = {k: v for k, v in step.items() if not k.startswith('log/')}
        action_id = step['action']
        stoch_state = step['dyn/stoch']

        if self.mode == 'state_action':
            np.add.at(self._checkpoint_counts, action_id, stoch_state.astype(np.int32))

        elif self.mode == 'state':
            self._checkpoint_counts += stoch_state.astype(np.int32)

    def add_counts(self, additional_counts):
        self._checkpoint_counts += additional_counts.astype(np.int32)

    def get_counts(self):
        return np.array(self._checkpoint_counts, copy=True, order='C')

    def set_counts(self, counts):
        self._checkpoint_counts = counts

    def counts_add_jit(self, state, action, counts):
        state = (state > 0.5).astype(jnp.int32)
        if self.mode == 'state_action':
            reshaped_state = state.reshape(-1, *state.shape[2:])
            counts = counts.at[action.ravel()].add(
                reshaped_state
            )
            return counts

        elif self.mode == 'state':
            counts = counts + state.sum(axis=(0, 1), dtype=jnp.int32)
            return counts

        return counts

    def get_intrinsic_reward(self, action, stoch_state, counts):
        if self.mode == 'state_action':
            counts = counts * stoch_state[..., None, :, :,]
            counts = jnp.sum(counts, axis=-1)
            counts = jnp.min(counts, axis=-1)
            # actions_expanded = jnp.expand_dims(action, -1)
            denominator = jnp.take_along_axis(counts, action[..., None], axis=-1).squeeze(-1)
            rewards = jnp.sqrt(2 * jnp.log(jnp.sum(counts, axis=-1)) / denominator)

        elif self.mode == 'state':
            # TODO: Have to fix stoch_state for cases where we have multiple environments
            counts = counts * stoch_state
            counts = jnp.sum(counts, axis=-1)
            counts = jnp.min(counts, axis=-1)
            rewards = jnp.sqrt(1 / counts)

        return rewards, self.beta

    def get_intrinsic_reward_numpy(self, action, stoch_state):
        if self.mode == 'state_action':
            stoch_state = np.repeat(stoch_state, self.num_actions, axis=0)
            counts = self._checkpoint_counts * stoch_state
            counts = np.sum(counts, axis=-1)
            counts = np.min(counts, axis=-1)
            rewards = np.sqrt(2 * np.log(np.sum(counts, axis=-1)) / counts[action])

        elif self.mode == 'state':
            # TODO: Have to fix stoch_state for cases where we have multiple environments
            counts = self._checkpoint_counts * stoch_state[0]
            counts = np.sum(counts, axis=-1)
            counts = np.min(counts, axis=-1)
            rewards = np.array([np.sqrt(1 / counts)])

        return rewards, self.beta

    @elements.timer.section('counts_save')
    def save(self):
        ### TODO: need to fix checkpointing
        data = {}
        data['counts'] = self._checkpoint_counts
        return data

    @elements.timer.section('counts_load')
    def load(self, data):
        self._checkpoint_counts = data['counts']