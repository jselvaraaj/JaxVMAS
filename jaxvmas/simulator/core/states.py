import chex
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from jaxvmas.equinox_utils import (
    equinox_filter_cond_return_pytree_node,
)
from jaxvmas.simulator.core.jax_vectorized_object import (
    JaxVectorizedObject,
    batch_dim,
    comm_dim,
    pos_dim,
)
from jaxvmas.simulator.utils import (
    JaxUtils,
)


@jaxtyped(typechecker=beartype)
class EntityState(JaxVectorizedObject):
    pos: Float[Array, f"{batch_dim} {pos_dim}"]
    vel: Float[Array, f"{batch_dim} {pos_dim}"]
    rot: Float[Array, f"{batch_dim} 1"]
    ang_vel: Float[Array, f"{batch_dim} 1"]

    @classmethod
    def create(cls, batch_dim: int, dim_p: int):
        chex.assert_scalar_positive(dim_p)

        # physical position
        pos = jnp.zeros((batch_dim, dim_p))
        # physical velocity
        vel = jnp.zeros((batch_dim, dim_p))
        # physical rotation -- from -pi to pi
        rot = jnp.zeros((batch_dim, 1))
        # angular velocity
        ang_vel = jnp.zeros((batch_dim, 1))
        return cls(batch_dim, pos, vel, rot, ang_vel)

    def _reset(self, env_index: int | float = jnp.nan) -> "EntityState":
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
    def _spawn(self, dim_p: int) -> "EntityState":
        return self.replace(
            pos=jnp.zeros((self.batch_dim, dim_p)),
            vel=jnp.zeros((self.batch_dim, dim_p)),
            rot=jnp.zeros((self.batch_dim, 1)),
            ang_vel=jnp.zeros((self.batch_dim, 1)),
        )

    def replace(self, **kwargs):
        if "pos" in kwargs:
            pos = kwargs["pos"]
            chex.assert_shape(pos, (self.batch_dim, None))
            chex.assert_equal_shape([pos, self.vel])

        elif "vel" in kwargs:
            vel = kwargs["vel"]
            chex.assert_shape(vel, (self.batch_dim, None))
            chex.assert_equal_shape([vel, self.pos])

        elif "ang_vel" in kwargs:
            ang_vel = kwargs["ang_vel"]
            chex.assert_shape(ang_vel, (self.batch_dim, None))
            chex.assert_equal_shape([ang_vel, self.rot])

        elif "rot" in kwargs:
            rot = kwargs["rot"]
            chex.assert_shape(rot, (self.batch_dim, None))
            chex.assert_equal_shape([rot, self.ang_vel])

        return super().replace(**kwargs)


@jaxtyped(typechecker=beartype)
class AgentState(EntityState):
    c: Float[Array, f"{batch_dim} {comm_dim}"]
    force: Float[Array, f"{batch_dim} {pos_dim}"]
    torque: Float[Array, f"{batch_dim} 1"]

    @classmethod
    def create(cls, batch_dim: int, dim_c: int, dim_p: int):
        chex.assert_scalar_positive(dim_c)

        entity_state = EntityState.create(batch_dim, dim_p)

        # This is okay since filter jit would make dim_c as static jitted variable
        if dim_c > 0:
            c = jnp.zeros((batch_dim, dim_c))
        else:
            c = jnp.full((batch_dim, dim_c), jnp.nan)

        # Agent force from actions
        force = jnp.zeros((batch_dim, dim_p))
        # Agent torque from actions
        torque = jnp.zeros((batch_dim, 1))
        return cls(
            batch_dim,
            entity_state.pos,
            entity_state.vel,
            entity_state.rot,
            entity_state.ang_vel,
            c,
            force,
            torque,
        )

    def _reset(self, env_index: int | float = jnp.nan) -> "AgentState":

        def env_index_is_nan(self, env_index: float):
            return equinox_filter_cond_return_pytree_node(
                ~jnp.any(jnp.isnan(self.c)),
                lambda _self: _self.replace(
                    c=jnp.zeros_like(_self.c),
                    force=jnp.zeros_like(_self.force),
                    torque=jnp.zeros_like(_self.torque),
                ),
                lambda _self: _self.replace(
                    force=jnp.zeros_like(_self.force),
                    torque=jnp.zeros_like(_self.torque),
                ),
                self,
            )

        def env_index_is_not_nan(self, env_index: int):
            return equinox_filter_cond_return_pytree_node(
                ~jnp.any(jnp.isnan(self.c)),
                lambda _self: _self.replace(
                    c=JaxUtils.where_from_index(
                        env_index, jnp.zeros_like(_self.c), _self.c
                    ),
                    force=JaxUtils.where_from_index(
                        env_index, jnp.zeros_like(_self.force), _self.force
                    ),
                    torque=JaxUtils.where_from_index(
                        env_index, jnp.zeros_like(_self.torque), _self.torque
                    ),
                ),
                lambda _self: _self.replace(
                    force=JaxUtils.where_from_index(
                        env_index, jnp.zeros_like(_self.force), _self.force
                    ),
                    torque=JaxUtils.where_from_index(
                        env_index, jnp.zeros_like(_self.torque), _self.torque
                    ),
                ),
                self,
            )

        self: "AgentState" = equinox_filter_cond_return_pytree_node(
            jnp.isnan(env_index),
            env_index_is_nan,
            env_index_is_not_nan,
            self,
            env_index,
        )

        self = EntityState._reset(self, env_index)

        return self

    def _spawn(self, dim_c: int, dim_p: int) -> "AgentState":

        def dim_c_is_greater_than_0(self: "AgentState", dim_c: int):
            return self.replace(c=jnp.zeros((self.batch_dim, dim_c)))

        def dim_c_is_not_greater_than_0(self: "AgentState", dim_c: int):
            return self.replace(c=jnp.full((self.batch_dim, dim_c), jnp.nan))

        self: "AgentState" = equinox_filter_cond_return_pytree_node(
            dim_c > 0,
            dim_c_is_greater_than_0,
            dim_c_is_not_greater_than_0,
            self,
            dim_c,
        )
        self = self.replace(
            force=jnp.zeros((self.batch_dim, dim_p)),
            torque=jnp.zeros((self.batch_dim, 1)),
        )
        return EntityState._spawn(self, dim_p)

    def replace(self, **kwargs):
        if "c" in kwargs:
            c = kwargs["c"]
            chex.assert_shape(c, (self.batch_dim, None))

        elif "force" in kwargs:
            force = kwargs["force"]
            chex.assert_shape(force, (self.batch_dim, None))
        elif "torque" in kwargs:
            torque = kwargs["torque"]
            chex.assert_shape(torque, (self.batch_dim, None))
        return super().replace(**kwargs)
