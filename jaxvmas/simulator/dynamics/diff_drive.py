#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, jaxtyped

if TYPE_CHECKING:
    from jaxvmas.simulator.core.agent import Agent
    from jaxvmas.simulator.core.world import World
from jaxvmas.simulator.dynamics.common import Dynamics
from jaxvmas.simulator.utils import X, Y


class DiffDrive(Dynamics):
    dt: float
    integration: str

    @classmethod
    def create(
        cls,
        world: "World",
        integration: str = "rk4",  # one of "euler", "rk4"
    ):
        assert integration == "rk4" or integration == "euler"

        dt = world.dt
        integration = integration
        return cls(dt, integration)

    @jaxtyped(typechecker=beartype)
    def f(self, state: Array, u_command: Array, ang_vel_command: Array) -> Array:
        theta = state[:, 2]
        dx = u_command * jnp.cos(theta)
        dy = u_command * jnp.sin(theta)
        dtheta = ang_vel_command
        return jnp.stack((dx, dy, dtheta), axis=-1)  # [batch_size,3]

    @jaxtyped(typechecker=beartype)
    def euler(self, state: Array, u_command: Array, ang_vel_command: Array) -> Array:
        return self.dt * self.f(state, u_command, ang_vel_command)

    @jaxtyped(typechecker=beartype)
    def runge_kutta(
        self, state: Array, u_command: Array, ang_vel_command: Array
    ) -> Array:
        k1 = self.f(state, u_command, ang_vel_command)
        k2 = self.f(state + self.dt * k1 / 2, u_command, ang_vel_command)
        k3 = self.f(state + self.dt * k2 / 2, u_command, ang_vel_command)
        k4 = self.f(state + self.dt * k3, u_command, ang_vel_command)
        return (self.dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    @property
    def needed_action_size(self) -> int:
        return 2

    @jaxtyped(typechecker=beartype)
    def process_action(self, agent: "Agent") -> tuple["DiffDrive", "Agent"]:
        u_command = agent.action.u[:, 0]  # Forward velocity
        ang_vel_command = agent.action.u[:, 1]  # Angular velocity

        # Current state of the agent
        state = jnp.concatenate((agent.state.pos, agent.state.rot), axis=1)

        v_cur_x = agent.state.vel[:, 0]  # Current velocity in x-direction
        v_cur_y = agent.state.vel[:, 1]  # Current velocity in y-direction
        v_cur_angular = agent.state.ang_vel[:, 0]  # Current angular velocity

        # Select the integration method to calculate the change in state
        if self.integration == "euler":
            delta_state = self.euler(state, u_command, ang_vel_command)
        else:
            delta_state = self.runge_kutta(state, u_command, ang_vel_command)

        # Calculate the accelerations required to achieve the change in state
        acceleration_x = (delta_state[:, 0] - v_cur_x * self.dt) / self.dt**2
        acceleration_y = (delta_state[:, 1] - v_cur_y * self.dt) / self.dt**2
        acceleration_angular = (
            delta_state[:, 2] - v_cur_angular * self.dt
        ) / self.dt**2

        # Calculate the forces required for the linear accelerations
        force_x = agent.mass * acceleration_x
        force_y = agent.mass * acceleration_y

        # Calculate the torque required for the angular acceleration
        torque = agent.moment_of_inertia * acceleration_angular

        # Update the physical force and torque required for the user inputs
        force = agent.state.force.at[:, X].set(force_x)
        force = force.at[:, Y].set(force_y)
        torque = torque[..., None]

        agent = agent.replace(
            state=agent.state.replace(
                force=force,
                torque=torque,
            )
        )

        return self, agent
