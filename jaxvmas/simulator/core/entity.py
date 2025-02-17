from enum import Enum

import chex
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Callable, Sequence
from jaxtyping import Array, Bool, Float, jaxtyped

from jaxvmas.equinox_utils import (
    equinox_filter_cond_return_dynamic,
    equinox_filter_cond_return_pytree_node,
)
from jaxvmas.simulator.core.jax_vectorized_object import (
    JaxVectorizedObject,
    batch_dim,
)
from jaxvmas.simulator.core.shapes import Shape, Sphere
from jaxvmas.simulator.core.states import EntityState
from jaxvmas.simulator.rendering import Geom
from jaxvmas.simulator.utils import (
    Color,
)


@jaxtyped(typechecker=beartype)
class Entity(JaxVectorizedObject):
    state: EntityState

    gravity: Float[Array, f"{batch_dim} 1"]

    name: str
    movable: bool
    rotatable: bool
    collide: bool
    density: float
    mass: float
    shape: Shape
    v_range: float
    max_speed: float
    color: Color
    is_joint: bool
    drag: float
    linear_friction: float
    angular_friction: float
    collision_filter: Callable[["Entity"], Bool[Array, "1"]]

    goal: str
    _render: Bool[Array, f"{batch_dim}"]

    @classmethod
    def create(
        cls,
        batch_dim: int,
        name: str,
        movable: bool = False,
        rotatable: bool = False,
        collide: bool = True,
        density: float = 25.0,  # Unused for now
        mass: float = 1.0,
        shape: Shape | None = None,
        v_range: float = jnp.nan,
        max_speed: float = jnp.nan,
        color=Color.GRAY,
        is_joint: bool = False,
        drag: float = jnp.nan,
        linear_friction: float = jnp.nan,
        angular_friction: float = jnp.nan,
        gravity: float | Sequence[float] | None = None,
        collision_filter: Callable[["Entity"], Bool[Array, "1"]] = lambda _: True,
        dim_p: int = 2,
    ):
        if shape is None:
            shape = Sphere()
        # name
        name = name
        # entity can move / be pushed
        movable = movable
        # entity can rotate
        rotatable = rotatable
        # entity collides with others
        collide = collide
        # material density (affects mass)
        density = density
        # mass
        mass = mass
        # max speed
        max_speed = max_speed
        v_range = v_range
        # color
        assert isinstance(color, Enum), f"Color must be a Enum, got {type(color)}"
        color = color
        # shape
        shape = shape
        # is joint
        is_joint = is_joint
        # collision filter
        collision_filter = collision_filter
        # drag
        drag = drag
        # friction
        linear_friction = linear_friction
        angular_friction = angular_friction
        # gravity
        if gravity is None:
            gravity = jnp.zeros((batch_dim, 1))
        else:
            gravity = jnp.asarray(gravity)
            chex.assert_shape(gravity, (batch_dim, 1))
        # entity goal
        goal = ""
        # Render the entity
        _render = jnp.full((batch_dim,), True)

        state = EntityState.create(batch_dim, dim_p)

        return cls(
            batch_dim,
            state,
            gravity,
            name,
            movable,
            rotatable,
            collide,
            density,
            mass,
            shape,
            v_range,
            max_speed,
            color,
            is_joint,
            drag,
            linear_friction,
            angular_friction,
            collision_filter,
            goal,
            _render,
        )

    @property
    def is_rendering(self):
        return self._render

    @property
    def moment_of_inertia(self):
        return self.shape.moment_of_inertia(self.mass)

    def reset_render(self):
        return self.replace(_render=jnp.full((self.batch_dim,), True))

    def collides(self, entity: "Entity"):
        # Here collision_filter is a static variable but entity is not.
        # So we need to use equinox_filter_cond to handle this.
        return equinox_filter_cond_return_dynamic(
            self.collide,
            lambda entity: jnp.asarray(self.collision_filter(entity)),
            lambda entity: jnp.asarray(False),
            entity,
        )

    def _spawn(self, **kwargs) -> "Entity":
        return self.replace(state=self.state._spawn(**kwargs))

    def _reset(self, env_index: int | float = jnp.nan):
        return self.replace(state=self.state._reset(env_index))

    def set_pos(self, pos: Array, batch_index: int | float = jnp.nan):
        return self._set_state_property("pos", pos, batch_index)

    def set_vel(self, vel: Array, batch_index: int | float = jnp.nan):
        return self._set_state_property("vel", vel, batch_index)

    def set_rot(self, rot: Array, batch_index: int | float = jnp.nan):
        return self._set_state_property("rot", rot, batch_index)

    def set_ang_vel(self, ang_vel: Array, batch_index: int | float = jnp.nan):
        return self._set_state_property("ang_vel", ang_vel, batch_index)

    def _set_state_property(
        self, prop_name: str, new: Array, batch_index: int | float = jnp.nan
    ):
        chex.assert_scalar(self.batch_dim)

        def batch_index_is_nan(old_state: EntityState, batch_index: float, new: Array):
            if len(new.shape) > 1 and new.shape[0] == self.batch_dim:
                new_state = old_state.replace(**{prop_name: new})
            else:
                value = getattr(old_state, prop_name)
                if new.ndim == value.ndim - 1:
                    new = new[None]
                new_state = old_state.replace(
                    **{prop_name: new.repeat(self.batch_dim, 0)}
                )
            return new_state

        def batch_index_is_not_nan(
            old_state: EntityState, batch_index: float, new: Array
        ):
            # Replace nan with a safe dummy index (e.g. 0) for tracing. Note, at this point, env_index should never be nan during execution.
            safe_index = jnp.where(jnp.isnan(batch_index), 0, batch_index).astype(
                jnp.int32
            )
            value = getattr(old_state, prop_name)
            new_state = old_state.replace(**{prop_name: value.at[safe_index].set(new)})
            return new_state

        new_state = equinox_filter_cond_return_pytree_node(
            jnp.any(jnp.isnan(batch_index)),
            batch_index_is_nan,
            batch_index_is_not_nan,
            self.state,
            batch_index,
            new,
        )
        # there was a notify_observers call in the past, so we need to notify observers manually
        return self.replace(state=new_state)

    @chex.assert_max_traces(0)
    def render(self, env_index: int = 0) -> "list[Geom]":
        from jaxvmas.simulator import rendering

        if not self.is_rendering[env_index]:
            return []
        geom = self.shape.get_geometry()
        xform = rendering.Transform()
        geom.add_attr(xform)

        xform.set_translation(*self.state.pos[env_index])
        xform.set_rotation(self.state.rot[env_index].item())

        color = self.color
        if isinstance(color, Array) and len(color.shape) > 1:
            color = color[env_index]
        geom.set_color(*color.value)

        return [geom]
