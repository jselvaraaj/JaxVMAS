import chex
import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, Int, jaxtyped

from jaxvmas.simulator.core.jax_vectorized_object import (
    JaxVectorizedObject,
    batch_axis_dim,
    comm_dim,
    env_index_dim,
    pos_dim,
)
from jaxvmas.simulator.utils import (
    JaxUtils,
)


@jaxtyped(typechecker=beartype)
class EntityState(JaxVectorizedObject):
    pos: Float[Array, f"{batch_axis_dim} {pos_dim}"] | None
    vel: Float[Array, f"{batch_axis_dim} {pos_dim}"] | None
    rot: Float[Array, f"{batch_axis_dim} 1"] | None
    ang_vel: Float[Array, f"{batch_axis_dim} 1"] | None

    def assert_is_spwaned(self):
        msg = "_spwan first"
        assert self.pos is not None, msg
        assert self.vel is not None, msg
        assert self.rot is not None, msg
        assert self.ang_vel is not None, msg
        super().assert_is_spwaned()

    @jaxtyped(typechecker=beartype)
    def _reset(
        self,
        env_index: Int[Array, f"{env_index_dim}"] | None = None,
    ) -> "EntityState":
        self.assert_is_spwaned()
        return self.replace(
            pos=JaxUtils.where_from_index(
                env_index, jnp.zeros_like(self.pos), self.pos
            ),
            vel=JaxUtils.where_from_index(
                env_index, jnp.zeros_like(self.vel), self.vel
            ),
            rot=JaxUtils.where_from_index(
                env_index, jnp.zeros_like(self.rot), self.rot
            ),
            ang_vel=JaxUtils.where_from_index(
                env_index, jnp.zeros_like(self.ang_vel), self.ang_vel
            ),
        )

    # Resets state for all entities
    @chex.assert_max_traces(0)
    @jaxtyped(typechecker=beartype)
    def _spawn(self, batch_dim: int, dim_p: int) -> "EntityState":
        chex.assert_scalar_positive(batch_dim)
        chex.assert_scalar_positive(dim_p)

        return self.replace(
            batch_dim=batch_dim,
            pos=jnp.zeros((batch_dim, dim_p)),
            vel=jnp.zeros((batch_dim, dim_p)),
            rot=jnp.zeros((batch_dim, 1)),
            ang_vel=jnp.zeros((batch_dim, 1)),
        )

    def replace(self, **kwargs):
        if "pos" in kwargs:
            pos = kwargs["pos"]
            if self.pos is not None:
                chex.assert_shape(pos, self.pos.shape)
            if self.vel is not None:
                chex.assert_equal_shape([pos, self.vel])

        if "vel" in kwargs:
            vel = kwargs["vel"]
            if self.vel is not None:
                chex.assert_shape(vel, self.vel.shape)
            if self.pos is not None:
                chex.assert_equal_shape([vel, self.pos])

        if "ang_vel" in kwargs:
            ang_vel = kwargs["ang_vel"]
            if self.ang_vel is not None:
                chex.assert_shape(ang_vel, self.ang_vel.shape)
            if self.rot is not None:
                chex.assert_equal_shape([ang_vel, self.rot])

        if "rot" in kwargs:
            rot = kwargs["rot"]
            if self.rot is not None:
                chex.assert_shape(rot, self.rot.shape)
            if self.ang_vel is not None:
                chex.assert_equal_shape([rot, self.ang_vel])

        return super().replace(**kwargs)


@jaxtyped(typechecker=beartype)
class AgentState(EntityState):
    c: Float[Array, f"{batch_axis_dim} {comm_dim}"] | None
    force: Float[Array, f"{batch_axis_dim} {pos_dim}"] | None
    torque: Float[Array, f"{batch_axis_dim} 1"] | None

    def assert_is_spwaned(self):
        assert self.batch_dim is not None
        assert self.c is not None
        assert self.force is not None
        assert self.torque is not None
        super().assert_is_spwaned()

    @jaxtyped(typechecker=beartype)
    def _reset(
        self,
        env_index: Int[Array, f"{env_index_dim}"] | None = None,
    ) -> "AgentState":
        self.assert_is_spwaned()

        def env_index_is_nan(
            env_index: Int[Array, f"{env_index_dim}"] | None,
            c: Float[Array, f"{batch_axis_dim} {comm_dim}"],
            force: Float[Array, f"{batch_axis_dim} {pos_dim}"],
            torque: Float[Array, f"{batch_axis_dim} 1"],
        ):
            return jax.lax.cond(
                ~jnp.any(jnp.isnan(c)),
                lambda c, force, torque: (
                    jnp.zeros_like(c),
                    jnp.zeros_like(force),
                    jnp.zeros_like(torque),
                ),
                lambda c, force, torque: (
                    c,
                    jnp.zeros_like(force),
                    jnp.zeros_like(torque),
                ),
                c,
                force,
                torque,
            )

        def env_index_is_not_nan(
            env_index: Int[Array, f"{env_index_dim}"] | None,
            c: Float[Array, f"{batch_axis_dim} {comm_dim}"],
            force: Float[Array, f"{batch_axis_dim} {pos_dim}"],
            torque: Float[Array, f"{batch_axis_dim} 1"],
        ):
            return jax.lax.cond(
                ~jnp.any(jnp.isnan(c)),
                lambda c, force, torque: (
                    JaxUtils.where_from_index(env_index, jnp.zeros_like(c), c),
                    JaxUtils.where_from_index(env_index, jnp.zeros_like(force), force),
                    JaxUtils.where_from_index(
                        env_index, jnp.zeros_like(torque), torque
                    ),
                ),
                lambda c, force, torque: (
                    c,
                    JaxUtils.where_from_index(env_index, jnp.zeros_like(force), force),
                    JaxUtils.where_from_index(
                        env_index, jnp.zeros_like(torque), torque
                    ),
                ),
                c,
                force,
                torque,
            )

        c, force, torque = jax.lax.cond(
            env_index is None,
            env_index_is_nan,
            env_index_is_not_nan,
            env_index,
            self.c,
            self.force,
            self.torque,
        )
        self = self.replace(c=c, force=force, torque=torque)

        return EntityState._reset(self, env_index)

    @chex.assert_max_traces(0)
    @jaxtyped(typechecker=beartype)
    def _spawn(self, batch_dim: int, dim_c: int, dim_p: int) -> "AgentState":

        def dim_c_is_greater_than_0():
            return jnp.zeros((batch_dim, dim_c))

        def dim_c_is_not_greater_than_0():
            return jnp.full((batch_dim, dim_c), jnp.nan)

        c = jax.lax.cond(
            dim_c > 0,
            dim_c_is_greater_than_0,
            dim_c_is_not_greater_than_0,
        )

        self = self.replace(
            c=c,
            force=jnp.zeros((batch_dim, dim_p)),
            torque=jnp.zeros((batch_dim, 1)),
        )
        return EntityState._spawn(self, batch_dim, dim_p)

    def replace(self, **kwargs):

        if "c" in kwargs:
            c = kwargs["c"]
            if self.batch_dim is not None:
                chex.assert_shape(c, (self.batch_dim, None))
            if self.c is not None:
                chex.assert_shape(c, self.c.shape)

        if "force" in kwargs:
            force = kwargs["force"]
            if self.batch_dim is not None:
                chex.assert_shape(force, (self.batch_dim, None))
            if self.force is not None:
                chex.assert_shape(force, self.force.shape)

        if "torque" in kwargs:
            torque = kwargs["torque"]
            if self.batch_dim is not None:
                chex.assert_shape(torque, (self.batch_dim, None))
            if self.torque is not None:
                chex.assert_shape(torque, self.torque.shape)

        return super().replace(**kwargs)
