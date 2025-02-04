#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import jax.numpy as jnp
from flax import struct
from jaxtyping import Array, Float

from jaxvmas.simulator.core import Agent, Entity, World
from jaxvmas.simulator.dynamics.holonomic import Holonomic
from jaxvmas.simulator.scenario import BaseScenario

# Type dimensions
batch = "batch"
n_agents = "n_agents"


@struct.dataclass
class WheelState:
    """Dynamic state for Wheel scenario."""

    wheel_angle: Float[Array, f"{batch}"] = struct.field(
        default_factory=lambda: jnp.zeros((1,))
    )
    wheel_angular_velocity: Float[Array, f"{batch}"] = struct.field(
        default_factory=lambda: jnp.zeros((1,))
    )
    agent_wheel_distances: Float[Array, f"{batch} {n_agents}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1))
    )
    target_angle_reached: Float[Array, f"{batch}"] = struct.field(
        default_factory=lambda: jnp.zeros((1,))
    )
    rotation_time: Float[Array, f"{batch}"] = struct.field(
        default_factory=lambda: jnp.zeros((1,))
    )


class Wheel(BaseScenario):
    """
    Scenario where agents must coordinate to rotate a wheel to a target angle.
    The wheel has momentum and requires careful control from multiple agents.
    """

    def __init__(self):
        super().__init__()
        self.n_agents = 4
        self.agent_size = 0.05
        self.wheel_radius = 0.3
        self.wheel_thickness = 0.05
        self.collision_penalty = 1.0
        self.target_angle = jnp.pi  # Target rotation is 180 degrees
        self.angle_tolerance = 0.1  # Tolerance for target angle
        self.min_rotation_time = 50  # Steps wheel must stay at target angle
        self.angular_damping = 0.1  # Damping factor for wheel rotation
        self.state = None

    def make_world(self, batch_dim: int, **kwargs) -> World:
        world = World(batch_dim=batch_dim, dim_p=2)

        # Add agents
        for i in range(self.n_agents):
            agent = Agent(name=f"agent_{i}", dynamics=Holonomic())
            agent.color = jnp.array([0.35, 0.35, 0.85])
            agent.collision_penalty = True
            agent.size = self.agent_size
            world.add_agent(agent)

        # Add wheel (as entity)
        wheel = Entity(name="wheel")
        wheel.collide = True
        wheel.movable = True
        wheel.rotatable = True  # Wheel can rotate
        wheel.size = self.wheel_thickness
        wheel.radius = self.wheel_radius
        wheel.color = jnp.array([0.85, 0.85, 0.35])  # Yellow wheel
        world.add_entity(wheel)

        # Initialize scenario state
        self.state = WheelState(
            wheel_angle=jnp.zeros(batch_dim),
            wheel_angular_velocity=jnp.zeros(batch_dim),
            agent_wheel_distances=jnp.zeros((batch_dim, self.n_agents)),
            target_angle_reached=jnp.zeros(batch_dim),
            rotation_time=jnp.zeros(batch_dim),
        )

        return world

    def reset_world_at(self, env_index: int | None):
        batch_size = self.world.batch_dim if env_index is None else 1

        # Place wheel at center
        wheel = self.world.entities[0]
        wheel.state.pos = jnp.zeros_like(wheel.state.pos)
        wheel.state.vel = jnp.zeros_like(wheel.state.vel)

        # Place agents around the wheel
        for i, agent in enumerate(self.world.agents):
            angle = i * (2 * jnp.pi / self.n_agents)
            radius = self.wheel_radius + 0.1  # Slightly outside wheel

            x_pos = radius * jnp.cos(angle)
            y_pos = radius * jnp.sin(angle)

            agent.state.pos = jnp.where(
                env_index is None,
                jnp.tile(jnp.array([x_pos, y_pos]), (batch_size, 1)),
                jnp.array([x_pos, y_pos]),
            )
            agent.state.vel = jnp.zeros_like(agent.state.vel)

        # Reset state
        self.state = WheelState(
            wheel_angle=jnp.zeros(batch_size),
            wheel_angular_velocity=jnp.zeros(batch_size),
            agent_wheel_distances=jnp.zeros((batch_size, self.n_agents)),
            target_angle_reached=jnp.zeros(batch_size),
            rotation_time=jnp.zeros(batch_size),
        )

    def reward(self, agent: Agent) -> Float[Array, f"{batch}"]:
        reward = jnp.zeros(self.world.batch_dim)
        agent_idx = self.world.agents.index(agent)
        wheel = self.world.entities[0]

        # Calculate distance to wheel
        dist_to_wheel = jnp.linalg.norm(agent.state.pos - wheel.state.pos, axis=-1)
        self.state = self.state.replace(
            agent_wheel_distances=self.state.agent_wheel_distances.at[:, agent_idx].set(
                dist_to_wheel
            )
        )

        # Calculate wheel dynamics
        # Agents apply torque based on their position relative to wheel center
        rel_pos = agent.state.pos - wheel.state.pos
        rel_vel = agent.state.vel
        torque = jnp.cross(rel_pos, rel_vel)[:, -1]  # Only use z-component

        # Update wheel angular velocity and angle
        new_angular_velocity = (
            self.state.wheel_angular_velocity
            + 0.1 * torque
            - self.angular_damping * self.state.wheel_angular_velocity
        )
        new_angle = self.state.wheel_angle + new_angular_velocity

        # Update wheel state
        self.state = self.state.replace(
            wheel_angle=new_angle,
            wheel_angular_velocity=new_angular_velocity,
        )

        # Check if target angle is reached
        angle_diff = jnp.abs(new_angle - self.target_angle)
        target_reached = angle_diff < self.angle_tolerance

        # Update target reached status and rotation time
        new_rotation_time = jnp.where(
            target_reached,
            self.state.rotation_time + 1,
            jnp.zeros_like(self.state.rotation_time),
        )
        self.state = self.state.replace(
            target_angle_reached=target_reached,
            rotation_time=new_rotation_time,
        )

        # Reward for being at correct distance from wheel
        optimal_dist = self.wheel_radius + self.agent_size
        dist_error = jnp.abs(dist_to_wheel - optimal_dist)
        reward = jnp.where(dist_error < 0.1, reward + 0.1, reward)

        # Reward for contributing to wheel rotation
        if angle_diff > self.angle_tolerance:
            # Reward positive torque when wheel needs to rotate clockwise
            # and negative torque when wheel needs to rotate counterclockwise
            desired_direction = jnp.sign(self.target_angle - self.state.wheel_angle)
            reward += 0.1 * torque * desired_direction

        # Bonus for maintaining target angle
        reward = jnp.where(
            new_rotation_time >= self.min_rotation_time, reward + 1.0, reward
        )

        # Collision penalties
        if agent.collision_penalty:
            for other in self.world.agents:
                if other is agent:
                    continue
                collision_dist = agent.size + other.size
                dist = jnp.linalg.norm(agent.state.pos - other.state.pos, axis=-1)
                reward = jnp.where(
                    dist < collision_dist, reward - self.collision_penalty, reward
                )

        return reward

    def observation(self, agent: Agent) -> Float[Array, f"{batch} ..."]:
        agent_idx = self.world.agents.index(agent)
        wheel = self.world.entities[0]

        # Get wheel state relative to agent
        wheel_pos = wheel.state.pos - agent.state.pos

        # Get positions of other agents relative to this agent
        other_pos = []
        other_vel = []
        for other in self.world.agents:
            if other is agent:
                continue
            other_pos.append(other.state.pos - agent.state.pos)
            other_vel.append(other.state.vel)

        # Stack observations
        obs = jnp.concatenate(
            [agent.state.pos]  # Own position
            + [agent.state.vel]  # Own velocity
            + [wheel_pos]  # Wheel position
            + [jnp.array([self.state.wheel_angle])]  # Current wheel angle
            + [jnp.array([self.state.wheel_angular_velocity])]  # Wheel angular velocity
            + [jnp.array([self.target_angle])]  # Target angle
            + other_pos  # Other agents' positions
            + other_vel  # Other agents' velocities
            + [jnp.array([self.state.target_angle_reached])]  # Target reached status
            + [jnp.array([self.state.rotation_time])],  # Time at target angle
            axis=-1,
        )
        return obs

    def done(self) -> Float[Array, f"{batch}"]:
        # Episode ends when wheel has been at target angle for minimum time
        return self.state.rotation_time >= self.min_rotation_time
