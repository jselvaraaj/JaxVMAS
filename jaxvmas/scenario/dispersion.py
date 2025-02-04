#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

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
class DispersionState:
    """Dynamic state for Dispersion scenario."""

    agent_distances: Float[Array, f"{batch} {n_agents} {n_agents}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1, 1))
    )
    min_distances: Float[Array, f"{batch} {n_agents}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1))
    )
    coverage_score: Float[Array, f"{batch}"] = struct.field(
        default_factory=lambda: jnp.zeros((1,))
    )
    dispersion_time: Float[Array, f"{batch}"] = struct.field(
        default_factory=lambda: jnp.zeros((1,))
    )


class Dispersion(BaseScenario):
    """
    Scenario where agents must spread out to cover an area efficiently.
    Agents are rewarded for maintaining minimum distances from each other
    while staying within the arena bounds.
    """

    def __init__(self):
        super().__init__()
        self.n_agents = 8
        self.agent_size = 0.05
        self.collision_penalty = 1.0
        self.arena_size = 1.0
        self.min_dispersion_dist = 0.3  # Minimum desired distance between agents
        self.dispersion_reward_scale = 0.1
        self.coverage_threshold = (
            0.8  # Fraction of min_dispersion_dist for good coverage
        )
        self.min_dispersion_time = 50  # Minimum time steps to maintain dispersion
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
        self.state = DispersionState(
            agent_distances=jnp.zeros((batch_dim, self.n_agents, self.n_agents)),
            min_distances=jnp.zeros((batch_dim, self.n_agents)),
            coverage_score=jnp.zeros(batch_dim),
            dispersion_time=jnp.zeros(batch_dim),
        )

        return world

    def reset_world_at(self, env_index: int | None):
        batch_size = self.world.batch_dim if env_index is None else 1

        # Place agents in a tight cluster near the center
        for i, agent in enumerate(self.world.agents):
            angle = 2 * jnp.pi * i / self.n_agents
            radius = 0.2  # Initial cluster radius
            pos = jnp.array(
                [
                    radius * jnp.cos(angle),
                    radius * jnp.sin(angle),
                ]
            )
            agent.state.pos = jnp.where(
                env_index is None,
                jnp.tile(pos[None, :], (batch_size, 1)),
                pos,
            )
            agent.state.vel = jnp.zeros_like(agent.state.vel)

        # Reset state
        self.state = DispersionState(
            agent_distances=jnp.zeros((batch_size, self.n_agents, self.n_agents)),
            min_distances=jnp.zeros((batch_size, self.n_agents)),
            coverage_score=jnp.zeros(batch_size),
            dispersion_time=jnp.zeros(batch_size),
        )

    def reward(self, agent: Agent) -> Float[Array, f"{batch}"]:
        reward = jnp.zeros(self.world.batch_dim)
        agent_idx = self.world.agents.index(agent)

        # Update distances between agents
        min_dist = jnp.inf * jnp.ones(self.world.batch_dim)
        for i, agent_i in enumerate(self.world.agents):
            for j, agent_j in enumerate(self.world.agents):
                if i != j:
                    dist = jnp.linalg.norm(
                        agent_i.state.pos - agent_j.state.pos, axis=-1
                    )
                    self.state = self.state.replace(
                        agent_distances=self.state.agent_distances.at[:, i, j].set(dist)
                    )
                    if i == agent_idx:
                        min_dist = jnp.minimum(min_dist, dist)

        # Update minimum distances for this agent
        self.state = self.state.replace(
            min_distances=self.state.min_distances.at[:, agent_idx].set(min_dist)
        )

        # Calculate coverage score (fraction of agents with good minimum distances)
        good_dispersion = self.state.min_distances > (
            self.coverage_threshold * self.min_dispersion_dist
        )
        coverage = jnp.mean(good_dispersion.astype(jnp.float32), axis=-1)
        self.state = self.state.replace(coverage_score=coverage)

        # Update dispersion time if coverage is good
        self.state = self.state.replace(
            dispersion_time=jnp.where(
                coverage >= self.coverage_threshold,
                self.state.dispersion_time + 1,
                jnp.zeros_like(self.state.dispersion_time),
            )
        )

        # Reward components
        # 1. Dispersion reward based on minimum distance to other agents
        target_dist_diff = min_dist - self.min_dispersion_dist
        dispersion_reward = jnp.where(
            min_dist < self.min_dispersion_dist,
            self.dispersion_reward_scale
            * target_dist_diff,  # Penalty for being too close
            self.dispersion_reward_scale
            * jnp.exp(-target_dist_diff),  # Small reward for good distance
        )
        reward += dispersion_reward

        # 2. Boundary penalty (keep agents inside arena)
        pos = agent.state.pos
        dist_to_boundary = jnp.maximum(
            jnp.abs(pos) - self.arena_size,
            0.0,
        )
        boundary_penalty = -self.collision_penalty * jnp.sum(dist_to_boundary, axis=-1)
        reward += boundary_penalty

        # 3. Collision penalties
        if agent.collision_penalty:
            for other in self.world.agents:
                if other is agent:
                    continue
                collision_dist = agent.size + other.size
                dist = jnp.linalg.norm(agent.state.pos - other.state.pos, axis=-1)
                reward = jnp.where(
                    dist < collision_dist,
                    reward - self.collision_penalty,
                    reward,
                )

        return reward

    def observation(self, agent: Agent) -> Float[Array, f"{batch} ..."]:
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
            + [self.state.min_distances],  # Minimum distances to other agents
            axis=-1,
        )
        return obs

    def done(self) -> Float[Array, f"{batch}"]:
        # Episode ends when good dispersion is maintained for minimum time
        return self.state.dispersion_time >= self.min_dispersion_time
