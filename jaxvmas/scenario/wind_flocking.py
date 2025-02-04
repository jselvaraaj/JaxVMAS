#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import jax
import jax.numpy as jnp
from flax import struct
from jaxtyping import Array, Float

from jaxvmas.simulator.core import Agent, World
from jaxvmas.simulator.dynamics.holonomic import Holonomic
from jaxvmas.simulator.scenario import BaseScenario

# Type dimensions
batch = "batch"
n_agents = "n_agents"


@struct.dataclass
class WindFlockingState:
    """Dynamic state for WindFlocking scenario."""

    agent_distances: Float[Array, f"{batch} {n_agents} {n_agents}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1, 1))
    )
    relative_velocities: Float[Array, f"{batch} {n_agents} {n_agents}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1, 1))
    )
    wind_velocity: Float[Array, f"{batch} 2"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 2))
    )
    flock_center: Float[Array, f"{batch} 2"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 2))
    )
    flock_velocity: Float[Array, f"{batch} 2"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 2))
    )


class WindFlocking(BaseScenario):
    """
    Scenario where agents must maintain a cohesive flock while dealing with wind forces.
    Wind direction and strength change over time, requiring adaptive flocking behavior.
    """

    def __init__(self):
        super().__init__()
        self.n_agents = 6
        self.agent_size = 0.05
        self.collision_penalty = 1.0
        self.arena_size = 1.0
        self.desired_distance = 0.3  # Target distance between agents
        self.max_wind_speed = 0.5
        self.wind_change_rate = 0.01  # How quickly wind changes
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

        # Initialize scenario state
        self.state = WindFlockingState(
            agent_distances=jnp.zeros((batch_dim, self.n_agents, self.n_agents)),
            relative_velocities=jnp.zeros((batch_dim, self.n_agents, self.n_agents)),
            wind_velocity=jnp.zeros((batch_dim, 2)),
            flock_center=jnp.zeros((batch_dim, 2)),
            flock_velocity=jnp.zeros((batch_dim, 2)),
        )

        return world

    def reset_world_at(self, env_index: int | None):
        batch_size = self.world.batch_dim if env_index is None else 1

        # Place agents in random positions within a smaller area
        for i, agent in enumerate(self.world.agents):
            pos = jax.random.uniform(
                jax.random.PRNGKey(i),
                (batch_size if env_index is None else 1, 2),
                minval=-0.5,
                maxval=0.5,
            )
            agent.state.pos = jnp.where(
                env_index is None,
                pos,
                pos[0],
            )
            agent.state.vel = jnp.zeros_like(agent.state.vel)

        # Initialize wind velocity
        wind_angle = jax.random.uniform(
            jax.random.PRNGKey(42),
            (batch_size if env_index is None else 1,),
            minval=0,
            maxval=2 * jnp.pi,
        )
        wind_speed = jax.random.uniform(
            jax.random.PRNGKey(43),
            (batch_size if env_index is None else 1,),
            minval=0,
            maxval=self.max_wind_speed,
        )
        wind_velocity = jnp.stack(
            [wind_speed * jnp.cos(wind_angle), wind_speed * jnp.sin(wind_angle)],
            axis=-1,
        )

        # Reset state
        self.state = WindFlockingState(
            agent_distances=jnp.zeros((batch_size, self.n_agents, self.n_agents)),
            relative_velocities=jnp.zeros((batch_size, self.n_agents, self.n_agents)),
            wind_velocity=wind_velocity,
            flock_center=jnp.zeros((batch_size, 2)),
            flock_velocity=jnp.zeros((batch_size, 2)),
        )

    def update_wind(self):
        """Update wind velocity with smooth changes."""
        # Add small random perturbations to wind direction and magnitude
        wind_speed = jnp.linalg.norm(self.state.wind_velocity, axis=-1)
        wind_angle = jnp.arctan2(
            self.state.wind_velocity[:, 1], self.state.wind_velocity[:, 0]
        )

        # Update angle and speed with noise
        key1, key2 = jax.random.split(jax.random.PRNGKey(0))
        angle_noise = jax.random.normal(key1, wind_angle.shape) * self.wind_change_rate
        speed_noise = jax.random.normal(key2, wind_speed.shape) * self.wind_change_rate

        new_angle = wind_angle + angle_noise
        new_speed = jnp.clip(wind_speed + speed_noise, 0.0, self.max_wind_speed)

        # Convert back to velocity components
        new_wind_velocity = jnp.stack(
            [new_speed * jnp.cos(new_angle), new_speed * jnp.sin(new_angle)], axis=-1
        )
        self.state = self.state.replace(wind_velocity=new_wind_velocity)

    def reward(self, agent: Agent) -> Float[Array, f"{batch}"]:
        reward = jnp.zeros(self.world.batch_dim)

        # Update wind
        self.update_wind()

        # Calculate flock center and velocity
        positions = jnp.stack([a.state.pos for a in self.world.agents], axis=1)
        velocities = jnp.stack([a.state.vel for a in self.world.agents], axis=1)

        flock_center = jnp.mean(positions, axis=1)
        flock_velocity = jnp.mean(velocities, axis=1)

        self.state = self.state.replace(
            flock_center=flock_center,
            flock_velocity=flock_velocity,
        )

        # Update distances and relative velocities
        for i, agent_i in enumerate(self.world.agents):
            for j, agent_j in enumerate(self.world.agents):
                if i != j:
                    # Distance between agents
                    dist = jnp.linalg.norm(
                        agent_i.state.pos - agent_j.state.pos, axis=-1
                    )
                    self.state = self.state.replace(
                        agent_distances=self.state.agent_distances.at[:, i, j].set(dist)
                    )

                    # Relative velocity
                    rel_vel = jnp.linalg.norm(
                        agent_i.state.vel - agent_j.state.vel, axis=-1
                    )
                    self.state = self.state.replace(
                        relative_velocities=self.state.relative_velocities.at[
                            :, i, j
                        ].set(rel_vel)
                    )

        # Reward components
        # 1. Cohesion - stay close to flock center
        dist_to_center = jnp.linalg.norm(
            agent.state.pos - self.state.flock_center, axis=-1
        )
        reward -= 0.1 * dist_to_center

        # 2. Separation - maintain desired distance from neighbors
        for other in self.world.agents:
            if other is agent:
                continue
            dist = jnp.linalg.norm(agent.state.pos - other.state.pos, axis=-1)
            dist_error = jnp.abs(dist - self.desired_distance)
            reward -= 0.1 * dist_error

        # 3. Alignment - match flock velocity
        vel_diff = jnp.linalg.norm(agent.state.vel - self.state.flock_velocity, axis=-1)
        reward -= 0.1 * vel_diff

        # 4. Wind adaptation - reward for moving with the wind
        wind_alignment = jnp.sum(
            agent.state.vel * self.state.wind_velocity, axis=-1
        ) / (jnp.linalg.norm(agent.state.vel, axis=-1) + 1e-6)
        reward += 0.1 * wind_alignment

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

        # Get positions and velocities of other agents relative to this agent
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
            + other_pos  # Other agents' positions
            + other_vel  # Other agents' velocities
            + [self.state.wind_velocity]  # Current wind velocity
            + [self.state.flock_center - agent.state.pos]  # Relative flock center
            + [self.state.flock_velocity],  # Flock velocity
            axis=-1,
        )
        return obs

    def done(self) -> Float[Array, f"{batch}"]:
        # Episode never ends
        return jnp.zeros(self.world.batch_dim, dtype=bool)
