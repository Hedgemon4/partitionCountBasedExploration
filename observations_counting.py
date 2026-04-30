import chex
import jax.numpy as jnp
import equinox as eqx


class ObservationCounts(eqx.Module):
    """Per-action counts on a regular grid over the observation space.

    Storage shape is (num_actions, num_bins, num_bins, ..., num_bins) with
    one per-dim axis of size `num_bins` for each observation dimension.
    """

    observation_counts: chex.Array
    low: chex.Array
    high: chex.Array
    num_bins: int = eqx.field(static=True)

    @classmethod
    def create(cls, num_bins, low, high, num_actions):
        # Infer obs_dim from the bounds so this works for any obs-space
        # dimensionality rather than hard-coding the 2D MountainCar case.
        obs_dim = int(low.shape[0])
        observation_counts = jnp.zeros(
            (num_actions,) + (num_bins,) * obs_dim, dtype=jnp.int32
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

        # Historical bug: this used `ravel_multi_index` and then only passed
        # two indexers (`actions`, `flat_index`) into a (num_actions, B, B)
        # array. That only indexed the first two dims — the third was
        # broadcast (so every obs silently updated every bin along the last
        # axis) and `flat_index` values >= num_bins fell out of bounds and
        # were dropped by jax's scatter-add default, hiding data entirely.
        #
        # Fix: provide one indexer per obs dim so each (action, bin0, bin1,
        # ..., binD-1) cell is updated exactly once per observation.
        obs_dim = bin_index.shape[1]
        per_dim_indexers = tuple(bin_index[:, d] for d in range(obs_dim))
        updated_counts = self.observation_counts.at[(actions,) + per_dim_indexers].add(
            1
        )

        return eqx.tree_at(lambda m: m.observation_counts, self, updated_counts)
