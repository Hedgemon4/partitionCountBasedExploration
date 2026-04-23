import chex
import jax.numpy as jnp
import equinox as eqx


class ObservationCounts(eqx.Module):
    # Going to be bins by actions
    observation_counts: chex.Array
    low: chex.Array
    high: chex.Array
    num_bins: int = eqx.field(static=True)

    @classmethod
    def create(cls, num_bins, low, high, num_actions):
        observation_counts = jnp.zeros(
            (num_actions, num_bins, num_bins), dtype=jnp.int32
        )
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

        flat_index = jnp.ravel_multi_index(
            bin_index.T,  # (obs_dim, batch)
            self.observation_counts.shape[1:],
            mode="clip",
        )

        updated_counts = self.observation_counts.at[
            actions,
            flat_index,
        ].add(1)

        return eqx.tree_at(lambda m: m.observation_counts, self, updated_counts)
