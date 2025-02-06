#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, List, Tuple, Union

import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array, Float

from jaxvmas.equinox_utils import PyTreeNode

if TYPE_CHECKING:
    from jaxvmas.simulator.core import Agent, Entity, World
from jaxvmas.simulator.rendering import Geom
from jaxvmas.simulator.utils import Color


class Sensor(PyTreeNode, ABC):
    agent: "Agent" | None
    world: "World"

    def __init__(self, world: "World"):
        super().__init__()
        self.world = world
        self.agent = None

    @abstractmethod
    def measure(self):
        raise NotImplementedError

    @abstractmethod
    def render(self, env_index: int = 0) -> "List[Geom]":
        raise NotImplementedError


# Define dimension variables for type annotations
batch_dim = "batch_dim"
n_rays = "n_rays"
pos_dim = "pos_dim"


class Lidar(Sensor):
    def __init__(
        self,
        world: "World",
        angle_start: float = 0.0,
        angle_end: float = 2 * jnp.pi,
        n_rays: int = 8,
        max_range: float = 1.0,
        entity_filter: Callable[["Entity"], bool] = lambda _: True,
        render_color: Union[Color, Tuple[float, float, float]] = Color.GRAY,
        alpha: float = 1.0,
        render: bool = True,
    ):
        super().__init__(world)
        if (angle_start - angle_end) % (jnp.pi * 2) < 1e-5:
            angles = jnp.linspace(angle_start, angle_end, n_rays + 1)[:n_rays]
        else:
            angles = jnp.linspace(angle_start, angle_end, n_rays)

        self._angles = jnp.tile(angles, (self._world.batch_size, 1))
        self._max_range = max_range
        self._last_measurement = None
        self._render = render
        self._entity_filter = entity_filter
        self._render_color = render_color
        self._alpha = alpha

    @property
    def entity_filter(self):
        return self._entity_filter

    @entity_filter.setter
    def entity_filter(self, entity_filter: Callable[["Entity"], bool]):
        self._entity_filter = entity_filter

    @property
    def render_color(self):
        if isinstance(self._render_color, Color):
            return self._render_color.value
        return self._render_color

    @property
    def alpha(self):
        return self._alpha

    def measure(self, vectorized: bool = True) -> Float[Array, f"{batch_dim} {n_rays}"]:
        agent_rot = self.agent.state.rot.squeeze(-1)

        if vectorized:
            measurement = self._world.cast_rays(
                self.agent,
                self._angles + agent_rot[:, None],
                self._max_range,
                self.entity_filter,
            )
        else:

            def scan_fn(_, angle):
                return self._world.cast_ray(
                    self.agent, angle + agent_rot, self._max_range, self.entity_filter
                )

            measurement = vmap(scan_fn, in_axes=(None, 0))(None, self._angles)

        self._last_measurement = measurement
        return measurement

    def set_render(self, render: bool):
        self._render = render

    def render(self, env_index: int = 0) -> List[Geom]:
        if not self._render or self._last_measurement is None:
            return []

        from jaxvmas.simulator import rendering

        geoms = []

        angles = self._angles[env_index] + self.agent.state.rot[env_index].squeeze()
        dists = self._last_measurement[env_index]

        for angle, dist in zip(angles, dists):
            # Ray line
            ray = rendering.Line(start=(0.0, 0.0), end=(dist, 0.0), width=0.05)
            xform = rendering.Transform()
            xform.set_translation(*self.agent.state.pos[env_index])
            xform.set_rotation(angle)
            ray.add_attr(xform)
            ray.set_color(0, 0, 0, self.alpha)

            # Ray endpoint
            ray_circ = rendering.make_circle(0.01)
            ray_circ.set_color(*self.render_color, alpha=self.alpha)
            circ_xform = rendering.Transform()
            rot = jnp.stack([jnp.cos(angle), jnp.sin(angle)])
            pos_circ = self.agent.state.pos[env_index] + rot * dist
            circ_xform.set_translation(*pos_circ)
            ray_circ.add_attr(circ_xform)

            geoms.extend([ray, ray_circ])

        return geoms
