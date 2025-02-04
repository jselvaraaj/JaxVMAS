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
class LineState:
    """Dynamic state for Line scenario."""

    agent_distances: Float[Array, f"{batch} {n_agents} {n_agents}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1, 1))
    )
    line_quality: Float[Array, f"{batch}"] = struct.field(
        default_factory=lambda: jnp.zeros((1,))
    )
    spacing_quality: Float[Array, f"{batch}"] = struct.field(
        default_factory=lambda: jnp.zeros((1,))
    )
    formation_time: Float[Array, f"{batch}"] = struct.field(
        default_factory=lambda: jnp.zeros((1,))
    )


class Line(BaseScenario):
    """
    Scenario where agents must form a line formation with equal spacing.
    The line can be horizontal or vertical, and agents are rewarded for
    maintaining proper formation and spacing.
    """

    def __init__(self):
        super().__init__()
        self.n_agents = 5
        self.agent_size = 0.05
        self.collision_penalty = 1.0
        self.arena_size = 1.0
        self.target_spacing = 0.2
        self.spacing_threshold = 0.05
        self.line_threshold = 0.05
        self.formation_time_threshold = 50
        self.line_reward = 1.0
        self.spacing_reward = 1.0
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
        self.state = LineState(
            agent_distances=jnp.zeros((batch_dim, self.n_agents, self.n_agents)),
            line_quality=jnp.zeros(batch_dim),
            spacing_quality=jnp.zeros(batch_dim),
            formation_time=jnp.zeros(batch_dim),
        )

        return world

    def reset_world_at(self, env_index: int | None):
        batch_size = self.world.batch_dim if env_index is None else 1

        # Place agents randomly in the arena
        for i, agent in enumerate(self.world.agents):
            pos = jnp.array(
                [
                    jnp.random.uniform(-self.arena_size + 0.1, self.arena_size - 0.1),
                    jnp.random.uniform(-self.arena_size + 0.1, self.arena_size - 0.1),
                ]
            )
            agent.state.pos = jnp.where(
                env_index is None,
                jnp.tile(pos[None, :], (batch_size, 1)),
                pos,
            )
            agent.state.vel = jnp.zeros_like(agent.state.vel)

        # Reset state
        self.state = LineState(
            agent_distances=jnp.zeros((batch_size, self.n_agents, self.n_agents)),
            line_quality=jnp.zeros(batch_size),
            spacing_quality=jnp.zeros(batch_size),
            formation_time=jnp.zeros(batch_size),
        )

    def reward(self, agent: Agent) -> Float[Array, f"{batch}"]:
        reward = jnp.zeros(self.world.batch_dim)

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

        # Calculate centroid
        centroid = jnp.mean(
            jnp.stack([a.state.pos for a in self.world.agents], axis=1),
            axis=1,
        )

        # Calculate principal direction (using covariance matrix)
        positions = jnp.stack([a.state.pos for a in self.world.agents], axis=1)
        centered_pos = positions - centroid[:, None, :]
        cov = jnp.einsum("bij,bik->bjk", centered_pos, centered_pos) / self.n_agents
        eigenvals, eigenvecs = jnp.linalg.eigh(cov)
        principal_direction = eigenvecs[..., -1]  # Direction of maximum variance

        # Project positions onto principal direction
        projections = jnp.einsum("bij,bi->bj", centered_pos, principal_direction)

        # Calculate line quality (deviation from principal direction)
        perpendicular_deviations = jnp.abs(
            centered_pos - projections[..., None] * principal_direction[:, None, :]
        )
        line_quality = jnp.mean(
            jnp.linalg.norm(perpendicular_deviations, axis=-1), axis=-1
        )
        self.state = self.state.replace(
            line_quality=jnp.where(
                line_quality < self.line_threshold,
                1.0,
                0.0,
            )
        )

        # Calculate spacing quality
        sorted_projections = jnp.sort(projections, axis=-1)
        spacings = sorted_projections[:, 1:] - sorted_projections[:, :-1]
        spacing_errors = jnp.abs(spacings - self.target_spacing)
        spacing_quality = jnp.mean(spacing_errors, axis=-1)
        self.state = self.state.replace(
            spacing_quality=jnp.where(
                spacing_quality < self.spacing_threshold,
                1.0,
                0.0,
            )
        )

        # Update formation time
        good_formation = self.state.line_quality > 0 & self.state.spacing_quality > 0
        self.state = self.state.replace(
            formation_time=jnp.where(
                good_formation,
                self.state.formation_time + 1,
                0.0,
            )
        )

        # Reward components
        # 1. Line formation reward
        reward += self.line_reward * self.state.line_quality

        # 2. Spacing reward
        reward += self.spacing_reward * self.state.spacing_quality

        # Collision penalties
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
            + other_vel,  # Other agents' velocities
            axis=-1,
        )

        return obs

    def done(self) -> Float[Array, f"{batch}"]:
        # Episode ends when agents maintain formation for sufficient time
        return self.state.formation_time >= self.formation_time_threshold
