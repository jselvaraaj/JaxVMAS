from dataclasses import asdict
from typing import TYPE_CHECKING

import chex
import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Callable, Sequence
from jaxtyping import Array, Bool, jaxtyped

from jaxvmas.simulator.core.action import Action
from jaxvmas.simulator.core.entity import Entity
from jaxvmas.simulator.core.shapes import Shape
from jaxvmas.simulator.core.states import AgentState
from jaxvmas.simulator.dynamics.common import Dynamics
from jaxvmas.simulator.dynamics.holonomic import Holonomic
from jaxvmas.simulator.rendering import Geom
from jaxvmas.simulator.sensors import Sensor
from jaxvmas.simulator.utils import (
    Color,
)

if TYPE_CHECKING:
    from jaxvmas.simulator.core.world import World


@jaxtyped(typechecker=beartype)
class Agent(Entity):
    state: AgentState
    action: Action

    obs_range: float
    obs_noise: float
    f_range: float
    max_f: float
    t_range: float
    max_t: float
    action_script: Callable[[Array, "Agent", "World"], tuple["Agent", "World"]]
    sensors: list[Sensor]
    c_noise: float
    silent: bool
    render_action: bool
    adversary: bool
    alpha: float

    dynamics: Dynamics
    action_size: int
    discrete_action_nvec: list[int]

    @classmethod
    def create(
        cls,
        batch_dim: int,
        name: str,
        dim_c: int,
        dim_p: int,
        shape: Shape | None = None,
        movable: bool = True,
        rotatable: bool = True,
        collide: bool = True,
        density: float = 25.0,  # Unused for now
        mass: float = 1.0,
        f_range: float = jnp.nan,
        max_f: float = jnp.nan,
        t_range: float = jnp.nan,
        max_t: float = jnp.nan,
        v_range: float = jnp.nan,
        max_speed: float = jnp.nan,
        color=Color.BLUE,
        alpha: float = 0.5,
        obs_range: float = jnp.nan,
        obs_noise: float = jnp.nan,
        u_noise: float | Sequence[float] = 0.0,
        u_range: float | Sequence[float] = 1.0,
        u_multiplier: float | Sequence[float] = 1.0,
        action_script: Callable[[Array, "Agent", "World"], tuple["Agent", "World"]] = (
            lambda _PRNG_key, *_: _
        ),
        sensors: list[Sensor] | None = None,
        c_noise: float = 0.0,
        silent: bool = True,
        adversary: bool = False,
        drag: float = jnp.nan,
        linear_friction: float = jnp.nan,
        angular_friction: float = jnp.nan,
        gravity: float | Sequence[float] | None = None,
        collision_filter: Callable[[Entity], Bool[Array, "1"]] = lambda _: True,
        render_action: bool = False,
        dynamics: Dynamics = None,  # Defaults to holonomic
        action_size: int | None = None,  # Defaults to what required by the dynamics
        discrete_action_nvec: (
            list[int] | None
        ) = None,  # Defaults to 3-way discretization if discrete actions are chosen (stay, decrement, increment)
    ):
        entity = Entity.create(
            batch_dim,
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
            is_joint=False,
            drag=drag,
            linear_friction=linear_friction,
            angular_friction=angular_friction,
            gravity=gravity,
            collision_filter=collision_filter,
        )

        # agents sensors
        sensors = [] if sensors is None else sensors

        if action_size is not None and discrete_action_nvec is not None:
            if action_size != len(discrete_action_nvec):
                raise ValueError(
                    f"action_size {action_size} is inconsistent with discrete_action_nvec {discrete_action_nvec}"
                )
        if discrete_action_nvec is not None:
            if not all(n > 1 for n in discrete_action_nvec):
                raise ValueError(
                    f"All values in discrete_action_nvec must be greater than 1, got {discrete_action_nvec}"
                )
        state = AgentState.create(batch_dim, dim_c, dim_p)
        # cannot observe the world
        obs_range = obs_range
        # observation noise
        obs_noise = obs_noise
        # force constraints
        f_range = f_range
        max_f = max_f
        # torque constraints
        t_range = t_range
        max_t = max_t
        # script behavior to execute
        action_script = action_script
        # non differentiable communication noise
        c_noise = c_noise
        # cannot send communication signals
        silent = silent
        # render the agent action force
        render_action = render_action
        # is adversary
        adversary = adversary
        # Render alpha
        alpha = alpha

        # Dynamics
        dynamics = dynamics if dynamics is not None else Holonomic()
        # Action
        if action_size is not None:
            action_size = action_size
        elif discrete_action_nvec is not None:
            action_size = len(discrete_action_nvec)
        else:
            action_size = dynamics.needed_action_size
        if discrete_action_nvec is None:
            discrete_action_nvec = [3] * action_size
        else:
            discrete_action_nvec = discrete_action_nvec
        action = Action.create(
            batch_dim=batch_dim,
            u_range=u_range,
            u_multiplier=u_multiplier,
            u_noise=u_noise,
            action_size=action_size,
            comm_dim=dim_c,
        )

        agent = cls(
            **(
                asdict(entity)
                | {
                    "state": state,
                    "action": action,
                    "obs_range": obs_range,
                    "obs_noise": obs_noise,
                    "f_range": f_range,
                    "max_f": max_f,
                    "t_range": t_range,
                    "max_t": max_t,
                    "action_script": action_script,
                    "sensors": sensors,
                    "c_noise": c_noise,
                    "silent": silent,
                    "render_action": render_action,
                    "adversary": adversary,
                    "alpha": alpha,
                    "dynamics": dynamics,
                    "action_size": action_size,
                    "discrete_action_nvec": discrete_action_nvec,
                }
            )
        )

        return agent

    @property
    def u_range(self):
        return self.action.u_range

    @jaxtyped(typechecker=beartype)
    def action_callback(
        self, PRNG_key: Array, world: "World"
    ) -> tuple["Agent", "World"]:
        PRNG_key, sub_key = jax.random.split(PRNG_key)
        self, world = self.action_script(sub_key, self, world)
        if self.silent or world.dim_c == 0:
            chex.assert_tree_all_finite(self.action.c)

        chex.assert_tree_all_finite(self.action.u)
        chex.assert_shape(self.action.u, (self.batch_dim, self.action_size))

        condition = (
            jnp.abs(self.action.u / self.action.u_multiplier_jax_array)
            <= self.action.u_range_jax_array
        )
        chex.assert_trees_all_equal(
            condition, jnp.full_like(condition, True, dtype=jnp.bool)
        )
        return self, world

    @jaxtyped(typechecker=beartype)
    def _spawn(self, dim_c: int, dim_p: int) -> "Agent":
        if dim_c == 0:
            assert (
                self.silent
            ), f"Agent {self.name} must be silent when world has no communication"
        if self.silent:
            dim_c = 0
        return super(Agent, self)._spawn(dim_c=dim_c, dim_p=dim_p)

    @jaxtyped(typechecker=beartype)
    def _reset(self, env_index: int) -> "Agent":
        self = self.replace(action=self.action._reset(env_index))
        self = self.replace(dynamics=self.dynamics.reset(env_index))
        return super(Agent, self)._reset(env_index)

    @chex.assert_max_traces(0)
    @jaxtyped(typechecker=beartype)
    def render(self, env_index: int = 0) -> "list[Geom]":
        from jaxvmas.simulator import rendering

        geoms = super(Agent, self).render(env_index)
        if len(geoms) == 0:
            return geoms
        for geom in geoms:
            geom.set_color(*self.color.value, alpha=self.alpha)
        if self.sensors is not None:
            for sensor in self.sensors:
                geoms += sensor.render(env_index=env_index)
        if self.render_action and self.state.force is not None:
            velocity = rendering.Line(
                self.state.pos[env_index],
                self.state.pos[env_index]
                + self.state.force[env_index] * 10 * self.shape.circumscribed_radius(),
                width=2,
            )
            velocity.set_color(*self.color.value)
            geoms.append(velocity)

        return geoms
