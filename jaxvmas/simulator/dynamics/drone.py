#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING

import chex
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Int, jaxtyped

if TYPE_CHECKING:
    from jaxvmas.simulator.core.agent import Agent
    from jaxvmas.simulator.core.world import World
from jaxvmas.simulator.dynamics.common import Dynamics
from jaxvmas.simulator.utils import JaxUtils, X, Y

batch_axis_dim = "batch_axis_dim"


@jaxtyped(typechecker=beartype)
class Drone(Dynamics):
    integration: str
    I_xx: float
    I_yy: float
    I_zz: float
    g: float
    dt: float
    batch_dim: int
    drone_state: Array

    @classmethod
    @chex.assert_max_traces(0)
    def create(
        cls,
        world: "World",
        I_xx: float = 8.1e-3,
        I_yy: float = 8.1e-3,
        I_zz: float = 14.2e-3,
        integration: str = "rk4",
    ) -> "Drone":

        assert integration in (
            "rk4",
            "euler",
        )

        integration = integration
        I_xx = I_xx
        I_yy = I_yy
        I_zz = I_zz
        g = 9.81
        dt = world.dt
        batch_dim = world.batch_dim
        drone_state = jnp.zeros((batch_dim, 12))
        return cls(integration, I_xx, I_yy, I_zz, g, dt, batch_dim, drone_state)

    @jaxtyped(typechecker=beartype)
    def reset(
        self,
        index: Int[Array, f"{batch_axis_dim}"] | Int[Array, ""] = jnp.asarray(-1),
    ) -> "Drone":
        # Drone state: phi(roll), theta (pitch), psi (yaw),
        #              p (roll_rate), q (pitch_rate), r (yaw_rate),
        #              x_dot (vel_x), y_dot (vel_y), z_dot (vel_z),
        #              x (pos_x), y (pos_y), z (pos_z)
        drone_state = JaxUtils.where_from_index(
            index,
            jnp.zeros_like(self.drone_state),
            self.drone_state,
        )
        self = self.replace(drone_state=drone_state)
        return self

    @jaxtyped(typechecker=beartype)
    def f(
        self,
        agent: "Agent",
        state: Array,
        thrust_command: Array,
        torque_command: Array,
    ) -> Array:
        phi = state[:, 0]
        theta = state[:, 1]
        psi = state[:, 2]
        p = state[:, 3]
        q = state[:, 4]
        r = state[:, 5]
        x_dot = state[:, 6]
        y_dot = state[:, 7]
        z_dot = state[:, 8]

        c_phi = jnp.cos(phi)
        s_phi = jnp.sin(phi)
        c_theta = jnp.cos(theta)
        s_theta = jnp.sin(theta)
        c_psi = jnp.cos(psi)
        s_psi = jnp.sin(psi)

        # Postion Dynamics
        x_ddot = (c_phi * s_theta * c_psi + s_phi * s_psi) * thrust_command / agent.mass
        y_ddot = (c_phi * s_theta * s_psi - s_phi * c_psi) * thrust_command / agent.mass
        z_ddot = (c_phi * c_theta) * thrust_command / agent.mass - self.g
        # Angular velocity dynamics
        p_dot = (torque_command[:, 0] - (self.I_yy - self.I_zz) * q * r) / self.I_xx
        q_dot = (torque_command[:, 1] - (self.I_zz - self.I_xx) * p * r) / self.I_yy
        r_dot = (torque_command[:, 2] - (self.I_xx - self.I_yy) * p * q) / self.I_zz

        return jnp.stack(
            [
                p,
                q,
                r,
                p_dot,
                q_dot,
                r_dot,
                x_ddot,
                y_ddot,
                z_ddot,
                x_dot,
                y_dot,
                z_dot,
            ],
            axis=-1,
        )

    @jaxtyped(typechecker=beartype)
    def needs_reset(self) -> Array:
        # Constraint roll and pitch within +-30 degrees
        return jnp.any(jnp.abs(self.drone_state[:, :2]) > 30 * (jnp.pi / 180), axis=-1)

    @jaxtyped(typechecker=beartype)
    def euler(
        self,
        agent: "Agent",
        state: Array,
        thrust: Array,
        torque: Array,
    ) -> Array:
        return self.dt * self.f(agent, state, thrust, torque)

    @jaxtyped(typechecker=beartype)
    def runge_kutta(
        self,
        agent: "Agent",
        state: Array,
        thrust: Array,
        torque: Array,
    ) -> Array:
        k1 = self.f(agent, state, thrust, torque)
        k2 = self.f(agent, state + self.dt * k1 / 2, thrust, torque)
        k3 = self.f(agent, state + self.dt * k2 / 2, thrust, torque)
        k4 = self.f(agent, state + self.dt * k3, thrust, torque)
        return (self.dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    @property
    def needed_action_size(self) -> int:
        return 4

    @jaxtyped(typechecker=beartype)
    def process_action(self, agent: "Agent") -> tuple["Drone", "Agent"]:
        u = agent.action.u
        thrust = u[:, 0]  # Thrust, sum of all propeller thrusts
        torque = u[:, 1:4]  # Torque in x, y, z direction

        thrust += agent.mass * self.g  # Ensure the drone is not falling

        drone_state = self.drone_state.at[:, 9].set(agent.state.pos[:, 0])  # x
        drone_state = drone_state.at[:, 10].set(agent.state.pos[:, 1])  # y
        drone_state = drone_state.at[:, 2].set(agent.state.rot[:, 0])  # psi (yaw)
        self = self.replace(drone_state=drone_state)

        if self.integration == "euler":
            delta_state = self.euler(agent, self.drone_state, thrust, torque)
        else:
            delta_state = self.runge_kutta(agent, self.drone_state, thrust, torque)

        # Calculate the change in state
        drone_state = self.drone_state + delta_state
        self = self.replace(drone_state=drone_state)

        v_cur_x = agent.state.vel[:, 0]  # Current velocity in x-direction
        v_cur_y = agent.state.vel[:, 1]  # Current velocity in y-direction
        v_cur_angular = agent.state.ang_vel[:, 0]  # Current angular velocity

        # Calculate the accelerations required to achieve the change in state
        acceleration_x = (delta_state[:, 6] - v_cur_x * self.dt) / self.dt**2
        acceleration_y = (delta_state[:, 7] - v_cur_y * self.dt) / self.dt**2
        acceleration_angular = (
            delta_state[:, 5] - v_cur_angular * self.dt
        ) / self.dt**2

        # Calculate the forces required for the linear accelerations
        force_x = agent.mass * acceleration_x
        force_y = agent.mass * acceleration_y

        # Calculate the torque required for the angular acceleration
        torque_yaw = agent.moment_of_inertia * acceleration_angular

        # Update the physical force and torque required for the user inputs
        force = agent.state.force.at[:, X].set(force_x)
        force = force.at[:, Y].set(force_y)
        torque = torque_yaw[..., None]

        agent = agent.replace(
            state=agent.state.replace(
                force=force,
                torque=torque,
            )
        )
        return self, agent
