#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.


from typing import TYPE_CHECKING

import jax.numpy as jnp

from jaxvmas.equinox_utils import PyTreeNode
from jaxvmas.simulator import rendering

if TYPE_CHECKING:
    from jaxvmas.simulator.core.entity import Entity
from jaxvmas.simulator.rendering import Geom
from jaxvmas.simulator.utils import Color, JaxUtils, X, Y

# Type dimensions
batch_dim = "batch"
pos_dim = "pos"

UNCOLLIDABLE_JOINT_RENDERING_WIDTH = 1


class Joint(PyTreeNode):
    entity_a: "Entity"
    entity_b: "Entity"
    rotate_a: bool
    rotate_b: bool
    fixed_rotation_a: float | None
    fixed_rotation_b: float | None
    landmark: "Entity"
    joint_constraints: list["JointConstraint"]

    @classmethod
    def create(
        cls,
        batch_dim: int,
        entity_a: "Entity",
        entity_b: "Entity",
        anchor_a: tuple[float, float] = (0.0, 0.0),
        anchor_b: tuple[float, float] = (0.0, 0.0),
        rotate_a: bool = True,
        rotate_b: bool = True,
        dist: float = 0.0,
        collidable: bool = False,
        width: float = 0.0,
        mass: float = 1.0,
        fixed_rotation_a: float | None = None,
        fixed_rotation_b: float | None = None,
    ):
        assert entity_a != entity_b, "Cannot join same entity"
        for anchor in (anchor_a, anchor_b):
            assert (
                max(anchor) <= 1 and min(anchor) >= -1
            ), f"Joint anchor points should be between -1 and 1, got {anchor}"
        assert dist >= 0, f"Joint dist must be >= 0, got {dist}"
        if dist == 0:
            assert not collidable, "Cannot have collidable joint with dist 0"
            assert width == 0, "Cannot have width for joint with dist 0"
            assert (
                fixed_rotation_a == fixed_rotation_b
            ), "If dist is 0, fixed_rotation_a and fixed_rotation_b should be the same"
        if fixed_rotation_a is not None:
            assert (
                not rotate_a
            ), "If you provide a fixed rotation for a, rotate_a should be False"
        if fixed_rotation_b is not None:
            assert (
                not rotate_b
            ), "If you provide a fixed rotation for b, rotate_b should be False"

        if width > 0:
            assert collidable

        self = cls(
            entity_a=entity_a,
            entity_b=entity_b,
            rotate_a=rotate_a,
            rotate_b=rotate_b,
            fixed_rotation_a=fixed_rotation_a,
            fixed_rotation_b=fixed_rotation_b,
            landmark=None,
            joint_constraints=[],
        )
        joint_constraints = []
        if dist == 0:
            joint_constraints.append(
                JointConstraint.create(
                    entity_a,
                    entity_b,
                    anchor_a=anchor_a,
                    anchor_b=anchor_b,
                    dist=dist,
                    rotate=rotate_a and rotate_b,
                    fixed_rotation=fixed_rotation_a,  # or b, it is the same
                ),
            )
        else:
            from jaxvmas.simulator.core.landmark import Landmark
            from jaxvmas.simulator.core.shapes import Box, Line

            self = self.replace(
                landmark=Landmark.create(
                    batch_dim=batch_dim,
                    name=f"joint {entity_a.name} {entity_b.name}",
                    collide=collidable,
                    movable=True,
                    rotatable=True,
                    mass=mass,
                    shape=(
                        Box(length=dist, width=width)
                        if width != 0
                        else Line(length=dist)
                    ),
                    color=Color.BLACK,
                    is_joint=True,
                ),
            )
            joint_constraints += [
                JointConstraint.create(
                    self.landmark,
                    entity_a,
                    anchor_a=(-1, 0),
                    anchor_b=anchor_a,
                    dist=0.0,
                    rotate=rotate_a,
                    fixed_rotation=fixed_rotation_a,
                ),
                JointConstraint.create(
                    self.landmark,
                    entity_b,
                    anchor_a=(1, 0),
                    anchor_b=anchor_b,
                    dist=0.0,
                    rotate=rotate_b,
                    fixed_rotation=fixed_rotation_b,
                ),
            ]

        return self.replace(joint_constraints=joint_constraints)

    def update_joint_state(self, entity_a: "Entity", entity_b: "Entity") -> "Joint":
        """Pure function to update joint state based on entity states"""

        assert (
            entity_a.name == self.entity_a.name
        ), f"Entity a name mismatch: {entity_a.name} != {self.entity_a.name}"
        assert (
            entity_b.name == self.entity_b.name
        ), f"Entity b name mismatch: {entity_b.name} != {self.entity_b.name}"

        self = self.replace(entity_a=entity_a, entity_b=entity_b)

        # Get positions of joint points
        pos_a = self.joint_constraints[0].pos_point(entity_a)
        pos_b = self.joint_constraints[1].pos_point(entity_b)

        # Calculate new landmark position and rotation
        new_pos = (pos_a + pos_b) / 2
        angle = jnp.atan2(
            pos_b[:, Y] - pos_a[:, Y],
            pos_b[:, X] - pos_a[:, X],
        )[..., None]

        # Create new landmark with updated position and rotation
        new_landmark = self.landmark.set_pos(self.landmark, new_pos, batch_index=None)
        new_landmark = new_landmark.set_rot(new_landmark, angle, batch_index=None)

        # Create new joint constraints with updated fixed rotations if needed
        new_joint_constraints = list(self.joint_constraints)

        if not self.rotate_a and self.fixed_rotation_a is None:
            new_constraint = new_joint_constraints[0].replace(
                fixed_rotation=angle - self.entity_a.state.rot
            )
            new_joint_constraints[0] = new_constraint

        if not self.rotate_b and self.fixed_rotation_b is None:
            new_constraint = new_joint_constraints[1].replace(
                fixed_rotation=angle - self.entity_b.state.rot
            )
            new_joint_constraints[1] = new_constraint

        # Return new joint instance with updated state
        return self.replace(
            landmark=new_landmark, joint_constraints=new_joint_constraints
        )


# Private class: do not instantiate directly
class JointConstraint(PyTreeNode):
    """
    This is an uncollidable constraint that bounds two entities in the specified anchor points at the specified distance
    """

    entity_a: "Entity"
    entity_b: "Entity"
    anchor_a: tuple[float, float]
    anchor_b: tuple[float, float]
    dist: float
    rotate: bool
    fixed_rotation: float | None
    _delta_anchor_tensor_map: dict["Entity", jnp.ndarray]

    @classmethod
    def create(
        cls,
        entity_a: "Entity",
        entity_b: "Entity",
        anchor_a: tuple[float, float] = (0.0, 0.0),
        anchor_b: tuple[float, float] = (0.0, 0.0),
        dist: float = 0.0,
        rotate: bool = True,
        fixed_rotation: float | None = None,
    ):
        assert entity_a != entity_b, "Cannot join same entity"
        for anchor in (anchor_a, anchor_b):
            assert (
                max(anchor) <= 1 and min(anchor) >= -1
            ), f"Joint anchor points should be between -1 and 1, got {anchor}"
        assert dist >= 0, f"Joint dist must be >= 0, got {dist}"
        if fixed_rotation is not None:
            assert not rotate, "If fixed rotation is provided, rotate should be False"
        if rotate:
            assert (
                fixed_rotation is None
            ), "If you provide a fixed rotation, rotate should be False"
            fixed_rotation = 0.0

        return cls(
            entity_a=entity_a,
            entity_b=entity_b,
            anchor_a=anchor_a,
            anchor_b=anchor_b,
            dist=dist,
            rotate=rotate,
            fixed_rotation=fixed_rotation,
            _delta_anchor_tensor_map={},
        )

    def update_joint_state(self, entity_a: "Entity", entity_b: "Entity"):
        self = self.replace(entity_a=entity_a, entity_b=entity_b)

        return self

    def _delta_anchor_jax_array(self, entity: "Entity"):
        _delta_anchor_tensor_map = {**self._delta_anchor_tensor_map}

        if entity.name == self.entity_a.name:
            anchor = self.anchor_a
        elif entity.name == self.entity_b.name:
            anchor = self.anchor_b
        else:
            raise ValueError(
                f"Entity {entity.name} is not part of joint {self.entity_a.name} {self.entity_b.name}"
            )

        delta = jnp.broadcast_to(
            entity.shape.get_delta_from_anchor(anchor)[None],
            entity.state.pos.shape,
        )
        _delta_anchor_tensor_map[entity.name] = delta

        return _delta_anchor_tensor_map[entity.name]

    def get_delta_anchor(self, entity: "Entity"):
        delta_anchor = self._delta_anchor_jax_array(entity)
        return JaxUtils.rotate_vector(
            delta_anchor,
            entity.state.rot,
        )

    def pos_point(
        self,
        entity: "Entity",
    ) -> tuple[jnp.ndarray, "JointConstraint"]:
        pos_point = self.get_delta_anchor(entity)
        return entity.state.pos + pos_point

    def render(
        self,
        env_index: int = 0,
    ) -> list[Geom]:
        if self.dist == 0:
            return []

        geoms: list[Geom] = []
        joint_line = Geom(
            (-self.dist / 2, 0),
            (self.dist / 2, 0),
            width=UNCOLLIDABLE_JOINT_RENDERING_WIDTH,
        )
        pos_point_a = self.pos_point(self.entity_a)
        pos_point_b = self.pos_point(self.entity_b)

        pos_point_a = pos_point_a[env_index]
        pos_point_b = pos_point_b[env_index]

        angle = jnp.atan2(
            pos_point_b[Y] - pos_point_a[Y],
            pos_point_b[X] - pos_point_a[X],
        )

        xform = rendering.Transform()
        xform.set_translation(*((pos_point_a + pos_point_b) / 2))
        xform.set_rotation(angle)
        joint_line.add_attr(xform)

        geoms.append(joint_line)
        return geoms
