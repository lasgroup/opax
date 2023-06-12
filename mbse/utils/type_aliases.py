from typing import Union
import jax
import flax.struct as struct


@struct.dataclass
class ModelProperties:
    alpha: Union[jax.Array, float] = 1.0
    bias_obs: Union[jax.Array, float] = 0.0
    bias_act: Union[jax.Array, float] = 0.0
    bias_out: Union[jax.Array, float] = 0.0
    scale_obs: Union[jax.Array, float] = 1.0
    scale_act: Union[jax.Array, float] = 1.0
    scale_out: Union[jax.Array, float] = 1.0


@struct.dataclass
class PolicyProperties:
    policy_bias_obs: Union[jax.Array, float] = 0.0
    policy_scale_obs: Union[jax.Array, float] = 0.0
