#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING

import chex
import jax.numpy as jnp
from beartype.typing import Callable
from jaxtyping import Array, Float

from jaxvmas.simulator.core.entity import Entity
from jaxvmas.simulator.core.jax_vectorized_object import batch_axis_dim
from jaxvmas.simulator.rendering import Geom
from jaxvmas.simulator.utils import (
    Color,
)

if TYPE_CHECKING:
    from jaxvmas.simulator.core.agent import Agent
    from jaxvmas.simulator.core.world import World
from jaxvmas.equinox_utils import PyTreeNode


class Sensor(PyTreeNode):

    def measure(self, agent: "Agent", world: "World") -> tuple:
        raise NotImplementedError

    def render(self, agent: "Agent", world: "World", env_index: int = 0) -> list[Geom]:
        raise NotImplementedError


n_rays = "n_rays"


class Lidar(Sensor):
    angles: Float[Array, f"{batch_axis_dim} {n_rays}"]
    max_range: float
    last_measurement: Float[Array, f"{batch_axis_dim} {n_rays}"]
    render_color: Color | tuple[float, float, float]
    alpha: float
    entity_filter: Callable[[Entity], bool]
    render: bool

    @classmethod
    def create(
        cls,
        batch_dim: int,
        angle_start: float = 0.0,
        angle_end: float = 2 * jnp.pi,
        n_rays: int = 8,
        max_range: float = 1.0,
        render_color: Color | tuple[float, float, float] = Color.GRAY,
        alpha: float = 1.0,
        render: bool = True,
        entity_filter: Callable[[Entity], bool] = lambda _: True,
    ):
        if (angle_start - angle_end) % (jnp.pi * 2) < 1e-5:
            angles = jnp.linspace(angle_start, angle_end, n_rays + 1)[:n_rays]
        else:
            angles = jnp.linspace(angle_start, angle_end, n_rays)

        angles = jnp.tile(angles, (batch_dim, 1))
        max_range = max_range
        last_measurement = jnp.full((batch_dim, n_rays), jnp.nan)
        render = render
        entity_filter = entity_filter
        render_color = render_color
        alpha = alpha

        return cls(
            angles=angles,
            max_range=max_range,
            last_measurement=last_measurement,
            render_color=render_color,
            alpha=alpha,
            entity_filter=entity_filter,
            render=render,
        )

    def measure(
        self, agent: "Agent", world: "World", vectorized: bool = True
    ) -> tuple["Lidar", Float[Array, f"{batch_axis_dim} {n_rays}"]]:
        if not vectorized:
            dists = []
            for angle in tuple(jnp.moveaxis(self.angles, 1, 0)):
                dists.append(
                    world.cast_ray(
                        agent,
                        angle + agent.state.rot.squeeze(-1),
                        max_range=self.max_range,
                        entity_filter=self.entity_filter,
                    )
                )
            measurement = jnp.stack(dists, dim=1)

        else:
            measurement = world.cast_rays(
                agent,
                self.angles + agent.state.rot,
                max_range=self.max_range,
                entity_filter=self.entity_filter,
            )
        last_measurement = measurement
        self = self.replace(last_measurement=last_measurement)
        return self, measurement

    @chex.assert_max_traces(0)
    def render(self, agent: "Agent", world: "World", env_index: int = 0) -> list[Geom]:
        if not self.render or self.last_measurement is None:
            return []

        from jaxvmas.simulator import rendering

        geoms = []

        angles = self.angles[env_index] + agent.state.rot[env_index].squeeze()
        dists = self.last_measurement[env_index]

        for angle, dist in zip(angles, dists):
            # Ray line
            ray = rendering.Line(start=(0.0, 0.0), end=(dist, 0.0), width=0.05)
            xform = rendering.Transform()
            xform.set_translation(*agent.state.pos[env_index])
            xform.set_rotation(angle)
            ray.add_attr(xform)
            ray.set_color(0, 0, 0, self.alpha)

            # Ray endpoint
            ray_circ = rendering.make_circle(0.01)
            ray_circ.set_color(*self.render_color, alpha=self.alpha)
            circ_xform = rendering.Transform()
            rot = jnp.stack([jnp.cos(angle), jnp.sin(angle)])
            pos_circ = agent.state.pos[env_index] + rot * dist
            circ_xform.set_translation(*pos_circ)
            ray_circ.add_attr(circ_xform)

            geoms.extend([ray, ray_circ])

        return geoms
