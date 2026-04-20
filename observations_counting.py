import chex
import jax.numpy as jnp


@chex.dataclass(frozen=True)
class ObservationCounts:
    # Going to be bins by actions
    observation_counts: chex.Array
    num_bins: chex.Scalar
    low: chex.Array
    high: chex.Array

    @classmethod
    def create(cls, num_bins, low, high, num_actions):
        observation_counts = jnp.zeros((num_actions, num_bins, num_bins), dtype=jnp.int32)
        return cls(
            observation_counts=observation_counts,
            num_bins=num_bins,
            low=low,
            high=high,
        )

    def update_counts(self, observations, actions):
        bin_index = jnp.floor(
            (observations - self.low) / (self.high - self.low) * self.num_bins
        ).astype(jnp.int32)
        bin_index = jnp.clip(bin_index, 0, self.num_bins - 1)

        obs_dim = self.low.shape[0]
        flat_index = jnp.ravel_multi_index(
            bin_index.T,  # (obs_dim, batch)
            (self.num_bins,) * obs_dim,
            mode='clip'
        )

        updated_counts = self.observation_counts.at[
            actions,
            flat_index,
        ].add(1)
        return self.replace(observation_counts=updated_counts)
