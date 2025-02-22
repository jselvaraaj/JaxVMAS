import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Callable, Sequence
from jaxtyping import Array, Bool, jaxtyped

from jaxvmas.simulator.core.entity import Entity
from jaxvmas.simulator.core.shapes import Shape
from jaxvmas.simulator.utils import (
    Color,
)


# properties of landmark entities
@jaxtyped(typechecker=beartype)
class Landmark(Entity):
    @classmethod
    def create(
        cls,
        name: str,
        shape: Shape = None,
        movable: bool = False,
        rotatable: bool = False,
        collide: bool = True,
        density: float = 25.0,  # Unused for now
        mass: float = 1.0,
        v_range: float = jnp.nan,
        max_speed: float = jnp.nan,
        color=Color.GRAY,
        is_joint: bool = False,
        drag: float = jnp.nan,
        linear_friction: float = jnp.nan,
        angular_friction: float = jnp.nan,
        gravity: float | Sequence[float] | None = None,
        collision_filter: Callable[[Entity], Bool[Array, "1"]] = lambda _: True,
    ):
        return super(Landmark, cls).create(
            name,
            movable,
            rotatable,
            collide,
            density,  # Unused for now
            mass,
            shape,
            v_range,
            max_speed,
            color,
            is_joint,
            drag,
            linear_friction,
            angular_friction,
            gravity,
            collision_filter,
        )
