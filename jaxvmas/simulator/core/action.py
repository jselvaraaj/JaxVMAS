import chex
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Sequence
from jaxtyping import Array, Float, Int, jaxtyped

from jaxvmas.simulator.core.jax_vectorized_object import (
    JaxVectorizedObject,
    action_size_dim,
    batch_axis_dim,
    comm_dim,
)
from jaxvmas.simulator.utils import (
    JaxUtils,
)


@jaxtyped(typechecker=beartype)
class Action(JaxVectorizedObject):
    _u: Float[Array, f"{batch_axis_dim} {action_size_dim}"] | None
    _c: Float[Array, f"{batch_axis_dim} {comm_dim}"] | None

    u_range: float | Sequence[float]
    u_multiplier: float | Sequence[float]
    u_noise: float | Sequence[float]
    action_size: int

    u_range_jax_array: Float[Array, f"{action_size_dim}"]
    u_multiplier_jax_array: Float[Array, f"{action_size_dim}"]
    u_noise_jax_array: Float[Array, f"{action_size_dim}"]

    @classmethod
    @chex.assert_max_traces(0)
    def create(
        cls,
        u_range: float | Sequence[float],
        u_multiplier: float | Sequence[float],
        u_noise: float | Sequence[float],
        action_size: int,
    ):
        chex.assert_scalar_non_negative(action_size)

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
            None,
            None,
            None,
            u_range,
            u_multiplier,
            u_noise,
            action_size,
            u_range_jax_array,
            u_multiplier_jax_array,
            u_noise_jax_array,
        )
        return action

    def assert_is_spwaned(self):
        super().assert_is_spwaned()
        assert self._u is not None, "Action must be spawned first"
        assert self._c is not None, "Action must be spawned first"

    @property
    def u(self):
        self.assert_is_spwaned()
        return self._u

    @chex.assert_max_traces(0)
    @jaxtyped(typechecker=beartype)
    def _spawn(self, batch_dim: int, comm_dim: int):
        chex.assert_scalar_non_negative(comm_dim)
        chex.assert_scalar_positive(batch_dim)

        u = jnp.zeros((batch_dim, self.action_size))

        # This is okay since filter jit would make comm_dim as static jitted variable
        if comm_dim > 0:
            c = jnp.zeros((batch_dim, comm_dim))
        else:
            c = jnp.full((batch_dim, comm_dim), jnp.nan)
        return self.replace(batch_dim=batch_dim, u=u, c=c)

    @property
    def c(self):
        self.assert_is_spwaned()
        return self._c

    def __post_init__(self):
        for attr in (self.u_multiplier, self.u_range, self.u_noise):
            if isinstance(attr, list):
                assert len(attr) == self.action_size, (
                    "Action attributes u_... must be either a float or a list of floats"
                    " (one per action) all with same length"
                )

    @jaxtyped(typechecker=beartype)
    def _reset(
        self,
        env_index: Int[Array, f"{batch_axis_dim}"] | Int[Array, ""] = jnp.asarray(-1),
    ) -> "Action":
        self.assert_is_spwaned()
        u = self.u
        u_reset = jnp.where(
            env_index == -1,
            jnp.zeros_like(u),
            JaxUtils.where_from_index(env_index, jnp.zeros_like(u), u),
        )

        c = self.c
        c_reset = jnp.where(
            env_index == -1,
            jnp.zeros_like(c),
            JaxUtils.where_from_index(env_index, jnp.zeros_like(c), c),
        )
        self = self.replace(c=c_reset, u=u_reset)

        return self

    def replace(self, **kwargs):
        if "u" in kwargs:
            u = kwargs.pop("u")
            kwargs["_u"] = u
            if self._u is not None:
                chex.assert_shape(u, self._u.shape)
        if "c" in kwargs:
            c = kwargs.pop("c")
            kwargs["_c"] = c
            if self._c is not None:
                chex.assert_shape(c, self._c.shape)

        return super().replace(**kwargs)
