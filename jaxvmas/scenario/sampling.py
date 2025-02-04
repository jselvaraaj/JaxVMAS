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
class SamplingState:
    """Dynamic state for Sampling scenario."""

    agent_positions: Float[Array, f"{batch} {n_agents} 2"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1, 2))
    )
    target_density: Float[Array, f"{batch} {n_agents}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1))
    )
    current_density: Float[Array, f"{batch} {n_agents}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1))
    )
    density_error: Float[Array, f"{batch}"] = struct.field(
        default_factory=lambda: jnp.zeros((1,))
    )
    convergence_time: Float[Array, f"{batch}"] = struct.field(
        default_factory=lambda: jnp.zeros((1,))
    )


class Sampling(BaseScenario):
    """
    Scenario where agents must learn to sample from a target distribution.
    Agents receive rewards based on how well their positions match the target density.
    """

    def __init__(self):
        super().__init__()
        self.n_agents = 10
        self.agent_size = 0.05
        self.collision_penalty = 1.0
        self.arena_size = 1.0
        self.density_grid_size = 20  # Number of grid cells per dimension
        self.convergence_threshold = 0.1  # Maximum allowed density error
        self.min_convergence_time = 100  # Steps density must stay converged
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
        self.state = SamplingState(
            agent_positions=jnp.zeros((batch_dim, self.n_agents, 2)),
            target_density=jnp.zeros((batch_dim, self.n_agents)),
            current_density=jnp.zeros((batch_dim, self.n_agents)),
            density_error=jnp.zeros(batch_dim),
            convergence_time=jnp.zeros(batch_dim),
        )

        return world

    def reset_world_at(self, env_index: int | None):
        batch_size = self.world.batch_dim if env_index is None else 1

        # Place agents in random positions
        for i, agent in enumerate(self.world.agents):
            pos = jax.random.uniform(
                jax.random.PRNGKey(i),
                (batch_size if env_index is None else 1, 2),
                minval=-self.arena_size,
                maxval=self.arena_size,
            )
            agent.state.pos = jnp.where(
                env_index is None,
                pos,
                pos[0],
            )
            agent.state.vel = jnp.zeros_like(agent.state.vel)

        # Generate target density (mixture of Gaussians)
        grid_x = jnp.linspace(-self.arena_size, self.arena_size, self.density_grid_size)
        grid_y = jnp.linspace(-self.arena_size, self.arena_size, self.density_grid_size)
        X, Y = jnp.meshgrid(grid_x, grid_y)
        positions = jnp.stack([X, Y], axis=-1)

        # Create two Gaussian components
        mu1 = jnp.array([0.5, 0.5])
        mu2 = jnp.array([-0.5, -0.5])
        sigma = 0.3
        density1 = jnp.exp(-jnp.sum((positions - mu1) ** 2, axis=-1) / (2 * sigma**2))
        density2 = jnp.exp(-jnp.sum((positions - mu2) ** 2, axis=-1) / (2 * sigma**2))
        target_density = (density1 + density2) / jnp.sum(density1 + density2)

        # Reset state
        self.state = SamplingState(
            agent_positions=jnp.zeros((batch_size, self.n_agents, 2)),
            target_density=jnp.tile(target_density[None, :], (batch_size, 1)),
            current_density=jnp.zeros((batch_size, self.density_grid_size**2)),
            density_error=jnp.ones(batch_size),
            convergence_time=jnp.zeros(batch_size),
        )

    def compute_density(
        self, positions: Float[Array, f"{batch} {n_agents} 2"]
    ) -> Float[Array, f"{batch} {n_agents}"]:
        """Compute the current density of agents using kernel density estimation."""
        grid_x = jnp.linspace(-self.arena_size, self.arena_size, self.density_grid_size)
        grid_y = jnp.linspace(-self.arena_size, self.arena_size, self.density_grid_size)
        X, Y = jnp.meshgrid(grid_x, grid_y)
        grid_positions = jnp.stack([X, Y], axis=-1)

        # Compute distances between agents and grid points
        dists = jnp.linalg.norm(
            positions[:, :, None, None, :] - grid_positions[None, None, :, :, :],
            axis=-1,
        )

        # Apply Gaussian kernel
        bandwidth = 0.1
        kernel = jnp.exp(-0.5 * (dists / bandwidth) ** 2)
        density = jnp.sum(kernel, axis=1)  # Sum over agents
        density = density.reshape(density.shape[0], -1)  # Flatten spatial dimensions

        # Normalize
        density = density / jnp.sum(density, axis=-1, keepdims=True)
        return density

    def reward(self, agent: Agent) -> Float[Array, f"{batch}"]:
        reward = jnp.zeros(self.world.batch_dim)
        agent_idx = self.world.agents.index(agent)

        # Update agent positions
        self.state = self.state.replace(
            agent_positions=self.state.agent_positions.at[:, agent_idx].set(
                agent.state.pos
            )
        )

        # Compute current density
        current_density = self.compute_density(self.state.agent_positions)
        self.state = self.state.replace(current_density=current_density)

        # Compute density error (KL divergence)
        density_error = jnp.sum(
            self.state.target_density
            * jnp.log(self.state.target_density / (current_density + 1e-10)),
            axis=-1,
        )
        self.state = self.state.replace(density_error=density_error)

        # Check convergence
        converged = density_error < self.convergence_threshold
        new_convergence_time = jnp.where(
            converged,
            self.state.convergence_time + 1,
            jnp.zeros_like(self.state.convergence_time),
        )
        self.state = self.state.replace(convergence_time=new_convergence_time)

        # Reward based on density matching
        reward -= density_error

        # Bonus for maintaining convergence
        reward = jnp.where(
            new_convergence_time >= self.min_convergence_time,
            reward + 1.0,
            reward,
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
            + other_pos  # Other agents' positions
            + other_vel  # Other agents' velocities
            + [self.state.target_density]  # Target density
            + [self.state.current_density]  # Current density
            + [jnp.array([self.state.density_error])]  # Current error
            + [jnp.array([self.state.convergence_time])],  # Time converged
            axis=-1,
        )
        return obs

    def done(self) -> Float[Array, f"{batch}"]:
        # Episode ends when density has converged for minimum time
        return self.state.convergence_time >= self.min_convergence_time
