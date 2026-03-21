#
# last_q = network.apply(
#     {
#         "params": train_state.params,
#         "batch_stats": train_state.batch_stats,
#     },
#     transitions.next_obs[-1],
#     train=False,
# )
# last_q = jnp.max(last_q, axis=-1)
#
# def _get_target(lambda_returns_and_next_q, transition):
#     lambda_returns, next_q = lambda_returns_and_next_q
#     target_bootstrap = (
#         transition.reward + config["GAMMA"] * (1 - transition.done) * next_q
#     )
#     delta = lambda_returns - next_q
#     lambda_returns = (
#         target_bootstrap + config["GAMMA"] * config["LAMBDA"] * delta
#     )
#     lambda_returns = (
#         1 - transition.done
#     ) * lambda_returns + transition.done * transition.reward
#     next_q = jnp.max(transition.q_val, axis=-1)
#     return (lambda_returns, next_q), lambda_returns
#
# last_q = last_q * (1 - transitions.done[-1])
# lambda_returns = transitions.reward[-1] + config["GAMMA"] * last_q
# _, targets = jax.lax.scan(
#     _get_target,
#     (lambda_returns, last_q),
#     jax.tree_util.tree_map(lambda x: x[:-1], transitions),
#     reverse=True,
# )
# lambda_targets = jnp.concatenate((targets, lambda_returns[np.newaxis]))
