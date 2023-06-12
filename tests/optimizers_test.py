from mbse.optimizers.cross_entropy_optimizer import CrossEntropyOptimizer
import jax.numpy as jnp
import jax
EPS = 5e-2

from jax.config import config
config.update("jax_log_compiles", 1)
def loss_function(x, bias):
    return -jnp.sum(jnp.square(x-bias))

@jax.jit
def optimize_for_bias(bias):
    func = lambda x: loss_function(x, bias)
    sequence, value = optimizer.optimize(func)
    return sequence, value
opt_cls = CrossEntropyOptimizer
num_steps = 20
action_dim = (10, 2)
for bias in jnp.linspace(-5, 5, 10):
    optimizer = opt_cls(
        action_dim=action_dim,
        num_steps=num_steps,
        upper_bound=jnp.inf,
        lr=0.1,
    )
    sequence, value = optimize_for_bias(bias)
    if jnp.max(jnp.abs(sequence-bias)) > EPS:
        print("Optimizer needs to be fixed")
        print("bias: ", bias)
        print("sequence: ", sequence)
    else:
        print("Optimizer works")
        print("bias: ", bias)
        print("sequence: ", sequence)





