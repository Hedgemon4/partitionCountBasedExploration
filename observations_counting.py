import chex
import jax.numpy as jnp


@chex.dataclass(frozen=True)
class ObservationCounts:
    # Going to be bins by actions
    observation_counts: chex.Array
    num_bins: chex.Scalar
    low: chex.Array
    high: chex.Array

    def update_counts(self, observations, actions):
        bin_index = jnp.floor(
            (observations - self.low) / (self.high - self.low) * self.num_bins
        ).astype(jnp.int32)
        N = actions.shape[0]
        obs_dim = self.low.shape[0]
        updated_counts = self.observation_counts.at[
            jnp.repeat(actions, obs_dim),
            jnp.tile(jnp.arange(obs_dim), N),
            bin_index.ravel(),
        ].add(1)
        return self.replace(observation_counts=updated_counts)
