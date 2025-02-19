import chex
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Sequence
from jaxtyping import Array, Float, jaxtyped

from jaxvmas.simulator.core.jax_vectorized_object import (
    JaxVectorizedObject,
    action_size_dim,
    batch_dim,
    comm_dim,
)
from jaxvmas.simulator.utils import (
    JaxUtils,
)


@jaxtyped(typechecker=beartype)
class Action(JaxVectorizedObject):
    u: Float[Array, f"{batch_dim} {action_size_dim}"]
    c: Float[Array, f"{batch_dim} {comm_dim}"]

    u_range: float | Sequence[float]
    u_multiplier: float | Sequence[float]
    u_noise: float | Sequence[float]
    action_size: int

    u_range_jax_array: Float[Array, f"{action_size_dim}"]
    u_multiplier_jax_array: Float[Array, f"{action_size_dim}"]
    u_noise_jax_array: Float[Array, f"{action_size_dim}"]

    @classmethod
    def create(
        cls,
        batch_dim: int,
        action_size: int,
        comm_dim: int,
        u_range: float | Sequence[float],
        u_multiplier: float | Sequence[float],
        u_noise: float | Sequence[float],
    ):
        chex.assert_scalar_non_negative(action_size)
        chex.assert_scalar_non_negative(comm_dim)
        chex.assert_scalar_positive(batch_dim)

        u = jnp.full((batch_dim, action_size), jnp.nan)

        # This is okay since filter jit would make comm_dim as static jitted variable
        if comm_dim > 0:
            c = jnp.zeros((batch_dim, comm_dim))
        else:
            c = jnp.full((batch_dim, comm_dim), jnp.nan)

        # control range
        _u_range = u_range
        # agent action is a force multiplied by this amount
        _u_multiplier = u_multiplier
        # physical motor noise amount
        _u_noise = u_noise
        # Number of actions
        action_size = action_size

        u_range_jax_array = jnp.asarray(
            _u_range if isinstance(_u_range, Sequence) else [_u_range] * action_size,
        )
        u_multiplier_jax_array = jnp.asarray(
            (
                _u_multiplier
                if isinstance(_u_multiplier, Sequence)
                else [_u_multiplier] * action_size
            ),
        )
        u_noise_jax_array = jnp.asarray(
            _u_noise if isinstance(_u_noise, Sequence) else [_u_noise] * action_size,
        )

        action = cls(
            batch_dim,
            u,
            c,
            u_range,
            u_multiplier,
            u_noise,
            action_size,
            u_range_jax_array,
            u_multiplier_jax_array,
            u_noise_jax_array,
        )
        return action

    def __post_init__(self):
        for attr in (self.u_multiplier, self.u_range, self.u_noise):
            if isinstance(attr, list):
                assert len(attr) == self.action_size, (
                    "Action attributes u_... must be either a float or a list of floats"
                    " (one per action) all with same length"
                )

    @jaxtyped(typechecker=beartype)
    def _reset(self, env_index: int | float = jnp.nan) -> "Action":
        u = self.u
        u_reset = jnp.where(
            jnp.isnan(env_index),
            jnp.zeros_like(u),
            JaxUtils.where_from_index(env_index, jnp.zeros_like(u), u),
        )
        self = self.replace(u=u_reset)

        c = self.c
        c_reset = jnp.where(
            jnp.isnan(env_index),
            jnp.zeros_like(c),
            JaxUtils.where_from_index(env_index, jnp.zeros_like(c), c),
        )
        self = self.replace(c=c_reset)

        return self

    def replace(self, **kwargs):
        if "u" in kwargs:
            u = kwargs["u"]
            chex.assert_shape(u, self.u.shape)
        elif "c" in kwargs:
            c = kwargs["c"]
            chex.assert_shape(c, self.c.shape)

        return super().replace(**kwargs)
