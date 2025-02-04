#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import jax
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
class BalanceState:
    """Dynamic state for Balance scenario."""

    agent_distances: Float[Array, f"{batch} {n_agents} {n_agents}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1, 1))
    )
    platform_tilt: Float[Array, f"{batch}"] = struct.field(
        default_factory=lambda: jnp.zeros((1,))
    )
    platform_angular_velocity: Float[Array, f"{batch}"] = struct.field(
        default_factory=lambda: jnp.zeros((1,))
    )
    agent_platform_distances: Float[Array, f"{batch} {n_agents}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1))
    )
    balance_time: Float[Array, f"{batch}"] = struct.field(
        default_factory=lambda: jnp.zeros((1,))
    )
    max_tilt: Float[Array, f"{batch}"] = struct.field(
        default_factory=lambda: jnp.zeros((1,))
    )


class Balance(BaseScenario):
    """
    Scenario where agents must cooperatively balance a platform.
    The platform tilts based on the distribution of agents' positions,
    and agents must maintain it close to horizontal.
    """

    def __init__(self):
        super().__init__()
        self.n_agents = 4
        self.agent_size = 0.05
        self.platform_length = 1.0
        self.platform_thickness = 0.05
        self.collision_penalty = 1.0
        self.arena_size = 1.0
        self.max_tilt_angle = jnp.pi / 4  # Maximum allowed tilt (45 degrees)
        self.tilt_sensitivity = 2.0  # How sensitive platform is to agent positions
        self.angular_damping = 0.1  # Damping factor for platform rotation
        self.min_balance_time = 100  # Steps platform must stay balanced
        self.balance_threshold = 0.1  # Maximum tilt for "balanced" state
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

        # Add platform (as entity)
        platform = Entity(name="platform")
        platform.collide = True
        platform.movable = False
        platform.rotatable = True
        platform.size = self.platform_thickness
        platform.length = self.platform_length
        platform.color = jnp.array([0.75, 0.75, 0.35])  # Yellow platform
        world.add_entity(platform)

        # Initialize scenario state
        self.state = BalanceState(
            agent_distances=jnp.zeros((batch_dim, self.n_agents, self.n_agents)),
            platform_tilt=jnp.zeros(batch_dim),
            platform_angular_velocity=jnp.zeros(batch_dim),
            agent_platform_distances=jnp.zeros((batch_dim, self.n_agents)),
            balance_time=jnp.zeros(batch_dim),
            max_tilt=jnp.zeros(batch_dim),
        )

        return world

    def reset_world_at(self, env_index: int | None):
        batch_size = self.world.batch_dim if env_index is None else 1

        # Place platform in center
        platform = self.world.entities[0]
        platform.state.pos = jnp.zeros_like(platform.state.pos)

        # Place agents randomly above platform
        for i, agent in enumerate(self.world.agents):
            pos = jax.random.uniform(
                jax.random.PRNGKey(i),
                (batch_size if env_index is None else 1, 2),
                minval=jnp.array([-self.platform_length / 2, 0.2]),
                maxval=jnp.array([self.platform_length / 2, 0.4]),
            )
            agent.state.pos = jnp.where(
                env_index is None,
                pos,
                pos[0],
            )
            agent.state.vel = jnp.zeros_like(agent.state.vel)

        # Reset state
        self.state = BalanceState(
            agent_distances=jnp.zeros((batch_size, self.n_agents, self.n_agents)),
            platform_tilt=jnp.zeros(batch_size),
            platform_angular_velocity=jnp.zeros(batch_size),
            agent_platform_distances=jnp.zeros((batch_size, self.n_agents)),
            balance_time=jnp.zeros(batch_size),
            max_tilt=jnp.zeros(batch_size),
        )

    def update_platform_dynamics(self):
        """Update platform tilt based on agent positions."""
        platform = self.world.entities[0]

        # Calculate torque from agent positions
        total_torque = jnp.zeros(self.world.batch_dim)
        for agent in self.world.agents:
            # Horizontal distance from platform center
            lever_arm = agent.state.pos[:, 0] - platform.state.pos[:, 0]
            # Weight force (simplified as 1.0)
            force = 1.0
            # Torque = force * lever_arm
            torque = force * lever_arm
            total_torque += torque

        # Update angular velocity with damping
        new_angular_velocity = (
            self.state.platform_angular_velocity
            + self.tilt_sensitivity * total_torque
            - self.angular_damping * self.state.platform_angular_velocity
        )

        # Update tilt angle
        new_tilt = jnp.clip(
            self.state.platform_tilt + new_angular_velocity,
            -self.max_tilt_angle,
            self.max_tilt_angle,
        )

        # Update state
        self.state = self.state.replace(
            platform_tilt=new_tilt,
            platform_angular_velocity=new_angular_velocity,
            max_tilt=jnp.maximum(self.state.max_tilt, jnp.abs(new_tilt)),
        )

    def reward(self, agent: Agent) -> Float[Array, f"{batch}"]:
        reward = jnp.zeros(self.world.batch_dim)
        agent_idx = self.world.agents.index(agent)

        # Update platform dynamics
        self.update_platform_dynamics()

        # Update distances between agents
        for i, agent_i in enumerate(self.world.agents):
            for j, agent_j in enumerate(self.world.agents):
                if i != j:
                    dist = jnp.linalg.norm(
                        agent_i.state.pos - agent_j.state.pos, axis=-1
                    )
                    self.state = self.state.replace(
                        agent_distances=self.state.agent_distances.at[:, i, j].set(dist)
                    )

        # Update distances to platform
        platform = self.world.entities[0]
        dist_to_platform = jnp.linalg.norm(
            agent.state.pos - platform.state.pos, axis=-1
        )
        self.state = self.state.replace(
            agent_platform_distances=self.state.agent_platform_distances.at[
                :, agent_idx
            ].set(dist_to_platform)
        )

        # Check if platform is balanced
        is_balanced = jnp.abs(self.state.platform_tilt) < self.balance_threshold

        # Update balance time
        new_balance_time = jnp.where(
            is_balanced,
            self.state.balance_time + 1,
            jnp.zeros_like(self.state.balance_time),
        )
        self.state = self.state.replace(balance_time=new_balance_time)

        # Reward components
        # 1. Tilt penalty (quadratic)
        tilt_penalty = -(self.state.platform_tilt**2)
        reward += tilt_penalty

        # 2. Angular velocity penalty (encourage smooth motion)
        velocity_penalty = -(self.state.platform_angular_velocity**2)
        reward += 0.1 * velocity_penalty

        # 3. Bonus for maintaining balance
        reward = jnp.where(
            new_balance_time >= self.min_balance_time,
            reward + 1.0,
            reward,
        )

        # 4. Collision penalties
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

        # Get positions and velocities of other agents relative to this agent
        other_pos = []
        other_vel = []
        for other in self.world.agents:
            if other is agent:
                continue
            other_pos.append(other.state.pos - agent.state.pos)
            other_vel.append(other.state.vel)

        # Get platform position relative to agent
        platform = self.world.entities[0]
        platform_pos = platform.state.pos - agent.state.pos

        # Stack observations
        obs = jnp.concatenate(
            [agent.state.pos]  # Own position
            + [agent.state.vel]  # Own velocity
            + [platform_pos]  # Platform position
            + [jnp.array([self.state.platform_tilt])]  # Current tilt
            + [jnp.array([self.state.platform_angular_velocity])]  # Angular velocity
            + other_pos  # Other agents' positions
            + other_vel  # Other agents' velocities
            + [jnp.array([self.state.balance_time])],  # Time balanced
            axis=-1,
        )
        return obs

    def done(self) -> Float[Array, f"{batch}"]:
        # Episode ends when platform has been balanced for minimum time
        # or when tilt exceeds maximum allowed angle
        return (self.state.balance_time >= self.min_balance_time) | (
            self.state.max_tilt >= self.max_tilt_angle
        )
