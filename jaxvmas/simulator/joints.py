#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from typing import Optional, Tuple

import jax.numpy as jnp
from flax import struct
from jaxtyping import Array, Float

import jaxvmas.simulator.core as core
import jaxvmas.simulator.utils as utils
from jaxvmas.simulator.rendering import Geom

# Type dimensions
batch_dim = "batch"
pos_dim = "pos"

UNCOLLIDABLE_JOINT_RENDERING_WIDTH = 1


@struct.dataclass(frozen=True)
class JointConstraint:
    """
    This is an uncollidable constraint that bounds two entities in the specified anchor points at the specified distance
    """

    entity_a: core.Entity
    entity_b: core.Entity
    anchor_a: Tuple[float, float] = (0.0, 0.0)
    anchor_b: Tuple[float, float] = (0.0, 0.0)
    dist: float = 0.0
    rotate: bool = True
    fixed_rotation: Optional[float] = None

    def __post_init__(self):
        assert self.entity_a != self.entity_b, "Cannot join same entity"
        for anchor in (self.anchor_a, self.anchor_b):
            assert (
                max(anchor) <= 1 and min(anchor) >= -1
            ), f"Joint anchor points should be between -1 and 1, got {anchor}"
        assert self.dist >= 0, f"Joint dist must be >= 0, got {self.dist}"
        if self.fixed_rotation is not None:
            assert (
                not self.rotate
            ), "If fixed rotation is provided, rotate should be False"
        if self.rotate:
            assert (
                self.fixed_rotation is None
            ), "If you provide a fixed rotation, rotate should be False"

    def get_delta_anchor(
        self, entity: core.Entity
    ) -> Float[Array, f"{batch_dim} {pos_dim}"]:
        """Get the anchor point delta in world coordinates"""
        if entity == self.entity_a:
            anchor = self.anchor_a
        elif entity == self.entity_b:
            anchor = self.anchor_b
        else:
            raise ValueError("Entity must be either entity_a or entity_b")

        delta = jnp.array(entity.shape.get_delta_from_anchor(anchor))
        return utils.rotate_vector(delta, entity.state.rot)

    def pos_point(self, entity: core.Entity) -> Float[Array, f"{batch_dim} {pos_dim}"]:
        """Get the anchor point position in world coordinates"""
        return entity.state.pos + self.get_delta_anchor(entity)

    def render(self, env_index: int = 0) -> list[Geom]:
        """Render the joint constraint"""
        if self.dist == 0:
            return []

        from jaxvmas.simulator import rendering

        geoms: list[Geom] = []
        joint_line = rendering.Line(
            (-self.dist / 2, 0),
            (self.dist / 2, 0),
            width=UNCOLLIDABLE_JOINT_RENDERING_WIDTH,
        )

        pos_point_a = self.pos_point(self.entity_a)[env_index]
        pos_point_b = self.pos_point(self.entity_b)[env_index]

        angle = jnp.arctan2(
            pos_point_b[utils.Y] - pos_point_a[utils.Y],
            pos_point_b[utils.X] - pos_point_a[utils.X],
        )

        xform = rendering.Transform()
        xform.set_translation(*((pos_point_a + pos_point_b) / 2))
        xform.set_rotation(float(angle))
        joint_line.add_attr(xform)

        geoms.append(joint_line)
        return geoms


class Joint(struct.PyTreeNode):
    """A joint that connects two entities with specified constraints"""

    entity_a: core.Entity
    entity_b: core.Entity
    landmark: Optional[core.Landmark] = None
    joint_constraints: list[JointConstraint] = struct.field(default_factory=list)
    rotate_a: bool = True
    rotate_b: bool = True
    fixed_rotation_a: Optional[float] = None
    fixed_rotation_b: Optional[float] = None

    @classmethod
    def create(
        cls,
        entity_a: core.Entity,
        entity_b: core.Entity,
        anchor_a: Tuple[float, float] = (0.0, 0.0),
        anchor_b: Tuple[float, float] = (0.0, 0.0),
        rotate_a: bool = True,
        rotate_b: bool = True,
        dist: float = 0.0,
        collidable: bool = False,
        width: float = 0.0,
        mass: float = 1.0,
        fixed_rotation_a: Optional[float] = None,
        fixed_rotation_b: Optional[float] = None,
    ) -> "Joint":
        """Create a new joint with specified properties"""

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

        joint_constraints = []
        landmark = None

        if dist == 0:
            joint_constraints.append(
                JointConstraint(
                    entity_a=entity_a,
                    entity_b=entity_b,
                    anchor_a=anchor_a,
                    anchor_b=anchor_b,
                    dist=dist,
                    rotate=rotate_a and rotate_b,
                    fixed_rotation=fixed_rotation_a,  # or b, they are the same
                ),
            )
        else:
            landmark = core.Landmark.create(
                batch_size=entity_a.batch_size,
                name=f"joint {entity_a.name} {entity_b.name}",
                collide=collidable,
                movable=True,
                rotatable=True,
                mass=mass,
                shape=(
                    core.Box(length=dist, width=width)
                    if width != 0
                    else core.Line(length=dist)
                ),
                color=utils.Color.BLACK,
                is_joint=True,
            )

            joint_constraints += [
                JointConstraint(
                    entity_a=landmark,
                    entity_b=entity_a,
                    anchor_a=(-1, 0),
                    anchor_b=anchor_a,
                    dist=0.0,
                    rotate=rotate_a,
                    fixed_rotation=fixed_rotation_a,
                ),
                JointConstraint(
                    entity_a=landmark,
                    entity_b=entity_b,
                    anchor_a=(1, 0),
                    anchor_b=anchor_b,
                    dist=0.0,
                    rotate=rotate_b,
                    fixed_rotation=fixed_rotation_b,
                ),
            ]

        return cls(
            entity_a=entity_a,
            entity_b=entity_b,
            landmark=landmark,
            joint_constraints=joint_constraints,
            rotate_a=rotate_a,
            rotate_b=rotate_b,
            fixed_rotation_a=fixed_rotation_a,
            fixed_rotation_b=fixed_rotation_b,
        )

    def update(self) -> "Joint":
        """Update joint state based on connected entities"""
        if self.landmark is None:
            return self

        pos_a = self.joint_constraints[0].pos_point(self.entity_a)
        pos_b = self.joint_constraints[1].pos_point(self.entity_b)

        # Update landmark position
        landmark = self.landmark.replace(
            state=self.landmark.state.replace(pos=(pos_a + pos_b) / 2)
        )

        # Calculate and update angle
        angle = jnp.arctan2(
            pos_b[:, utils.Y] - pos_a[:, utils.Y],
            pos_b[:, utils.X] - pos_a[:, utils.X],
        ).reshape(-1, 1)

        landmark = landmark.replace(state=landmark.state.replace(rot=angle))

        # Update fixed rotations if needed
        constraints = list(self.joint_constraints)
        if not self.rotate_a and self.fixed_rotation_a is None:
            constraints[0] = JointConstraint(
                **{
                    **vars(constraints[0]),
                    "fixed_rotation": angle - self.entity_a.state.rot,
                }
            )
        if not self.rotate_b and self.fixed_rotation_b is None:
            constraints[1] = JointConstraint(
                **{
                    **vars(constraints[1]),
                    "fixed_rotation": angle - self.entity_b.state.rot,
                }
            )

        return self.replace(landmark=landmark, joint_constraints=constraints)

    def render(self, env_index: int = 0) -> list[Geom]:
        """Render the joint"""
        geoms = []
        for constraint in self.joint_constraints:
            geoms.extend(constraint.render(env_index))
        return geoms
