#  Copyright (c) 2023-2025.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

if TYPE_CHECKING:
    from jaxvmas.simulator.core import Agent
from jaxvmas.simulator.dynamics.common import Dynamics

dim_batch = "batch"
dim_state = "state"


class KinematicBicycle(Dynamics):
    width: float
    l_f: float
    l_r: float
    max_steering_angle: float
    integration: str
    dt: float

    @classmethod
    def create(
        cls,
        width: float,
        l_f: float,
        l_r: float,
        max_steering_angle: float,
        dt: float,
        integration: str = "rk4",  # "euler" or "rk4"
    ):
        assert integration in ("rk4", "euler"), "Integration must be 'euler' or 'rk4'"
        return cls(width, l_f, l_r, max_steering_angle, integration, dt)

    @property
    def needed_action_size(self) -> int:
        return 2

    def process_action(self, agent: "Agent") -> tuple["KinematicBicycle", "Agent"]:
        v_command = agent.action.u[:, 0]
        steering_command = agent.action.u[:, 1]

        # Clip steering angle to physical limits
        steering_command = jnp.clip(
            steering_command, -self.max_steering_angle, self.max_steering_angle
        )

        # Current state [x, y, rot]
        state = jnp.concatenate([agent.state.pos, agent.state.rot], axis=-1)

        # Calculate state derivatives
        def f(
            state: Float[Array, f"{dim_batch} {dim_state}"]
        ) -> Float[Array, f"{dim_batch} {dim_state}"]:
            theta = state[:, 2]  # Yaw angle
            beta = jnp.arctan2(
                jnp.tan(steering_command) * self.l_r / (self.l_f + self.l_r), 1.0
            )
            dx = v_command * jnp.cos(theta + beta)
            dy = v_command * jnp.sin(theta + beta)
            dtheta = (
                v_command
                / (self.l_f + self.l_r)
                * jnp.cos(beta)
                * jnp.tan(steering_command)
            )
            return jnp.stack([dx, dy, dtheta], axis=-1)

        # Integration methods
        dt = self.dt

        def euler(_):
            return dt * f(state)

        def rk4(_):
            k1 = f(state)
            k2 = f(state + dt * k1 / 2)
            k3 = f(state + dt * k2 / 2)
            k4 = f(state + dt * k3)
            return (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        delta_state = jax.lax.cond(self.integration == "rk4", rk4, euler, operand=None)

        # Calculate required accelerations
        v_cur = agent.state.vel
        ang_vel_cur = agent.state.ang_vel.squeeze(-1)

        acceleration_linear = (delta_state[:, :2] - v_cur * dt) / dt**2
        acceleration_angular = (delta_state[:, 2] - ang_vel_cur * dt) / dt**2

        # Convert to forces
        force = agent.mass * acceleration_linear
        torque = agent.moment_of_inertia * acceleration_angular[..., None]

        # Update agent state
        agent = agent.replace(
            state=agent.state.replace(
                force=force,
                torque=torque,
            )
        )
        return self, agent
