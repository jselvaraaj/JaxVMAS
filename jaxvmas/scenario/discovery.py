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
n_targets = "n_targets"


@struct.dataclass
class DiscoveryState:
    """Dynamic state for Discovery scenario."""

    agent_target_distances: Float[Array, f"{batch} {n_agents} {n_targets}"] = (
        struct.field(default_factory=lambda: jnp.zeros((1, 1, 1)))
    )
    targets_discovered: Float[Array, f"{batch} {n_targets}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1))
    )
    discovery_times: Float[Array, f"{batch} {n_targets}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1))
    )
    total_discoveries: Float[Array, f"{batch}"] = struct.field(
        default_factory=lambda: jnp.zeros((1,))
    )


class Discovery(BaseScenario):
    """
    Scenario where agents must explore the environment to discover hidden targets.
    Agents receive rewards for finding new targets and must coordinate their search.
    """

    def __init__(self):
        super().__init__()
        self.n_agents = 3
        self.n_targets = 5
        self.agent_size = 0.05
        self.target_size = 0.05
        self.collision_penalty = 1.0
        self.discovery_distance = 0.2
        self.discovery_reward = 5.0
        self.arena_size = 1.5
        self.min_target_distance = 0.3  # Minimum distance between targets
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

        # Add targets (as landmarks)
        for i in range(self.n_targets):
            target = Entity(name=f"target_{i}")
            target.collide = False
            target.movable = False
            target.size = self.target_size
            target.color = jnp.array([0.85, 0.35, 0.35])  # Red targets
            world.add_landmark(target)

        # Initialize scenario state
        self.state = DiscoveryState(
            agent_target_distances=jnp.zeros(
                (batch_dim, self.n_agents, self.n_targets)
            ),
            targets_discovered=jnp.zeros((batch_dim, self.n_targets)),
            discovery_times=jnp.zeros((batch_dim, self.n_targets)),
            total_discoveries=jnp.zeros(batch_dim),
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

        # Place targets in random positions with minimum separation
        for i, target in enumerate(self.world.landmarks):
            valid_position = False
            attempts = 0
            while not valid_position and attempts < 100:
                pos = jax.random.uniform(
                    jax.random.PRNGKey(i + self.n_agents + attempts),
                    (batch_size if env_index is None else 1, 2),
                    minval=-self.arena_size,
                    maxval=self.arena_size,
                )

                # Check distance from other targets
                valid_position = True
                for j in range(i):
                    other_target = self.world.landmarks[j]
                    dist = jnp.linalg.norm(pos - other_target.state.pos, axis=-1)
                    if jnp.any(dist < self.min_target_distance):
                        valid_position = False
                        break
                attempts += 1

            target.state.pos = jnp.where(
                env_index is None,
                pos,
                pos[0],
            )

        # Reset state
        self.state = DiscoveryState(
            agent_target_distances=jnp.zeros(
                (batch_size, self.n_agents, self.n_targets)
            ),
            targets_discovered=jnp.zeros((batch_size, self.n_targets)),
            discovery_times=jnp.zeros((batch_size, self.n_targets)),
            total_discoveries=jnp.zeros(batch_size),
        )

    def reward(self, agent: Agent) -> Float[Array, f"{batch}"]:
        reward = jnp.zeros(self.world.batch_dim)
        agent_idx = self.world.agents.index(agent)

        # Calculate distances to all targets
        for i, target in enumerate(self.world.landmarks):
            dist = jnp.linalg.norm(agent.state.pos - target.state.pos, axis=-1)
            self.state = self.state.replace(
                agent_target_distances=self.state.agent_target_distances.at[
                    :, agent_idx, i
                ].set(dist)
            )

            # Check for new discoveries
            discovered = dist < self.discovery_distance
            was_discovered = self.state.targets_discovered[:, i] > 0

            # Update discovery status and give rewards
            new_discovery = discovered & ~was_discovered
            reward = jnp.where(new_discovery, reward + self.discovery_reward, reward)

            # Update target discovery status
            self.state = self.state.replace(
                targets_discovered=self.state.targets_discovered.at[:, i].set(
                    jnp.where(discovered, 1.0, self.state.targets_discovered[:, i])
                ),
                discovery_times=self.state.discovery_times.at[:, i].set(
                    jnp.where(new_discovery, 1.0, self.state.discovery_times[:, i])
                ),
            )

        # Update total discoveries
        self.state = self.state.replace(
            total_discoveries=jnp.sum(self.state.targets_discovered, axis=-1)
        )

        # Small reward for exploration (negative distance to nearest undiscovered target)
        undiscovered_distances = jnp.where(
            self.state.targets_discovered[:, None] == 0,
            self.state.agent_target_distances[:, agent_idx],
            jnp.inf,
        )
        min_undiscovered_dist = jnp.min(undiscovered_distances, axis=-1)
        reward = jnp.where(
            min_undiscovered_dist < jnp.inf,
            reward - 0.1 * min_undiscovered_dist,
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

        # Get positions of discovered targets relative to agent
        discovered_target_pos = []
        for i, target in enumerate(self.world.landmarks):
            # Only include position if target is discovered
            pos = jnp.where(
                self.state.targets_discovered[:, i : i + 1] > 0,
                target.state.pos - agent.state.pos,
                jnp.zeros_like(target.state.pos),
            )
            discovered_target_pos.append(pos)

        # Stack observations
        obs = jnp.concatenate(
            [agent.state.pos]  # Own position
            + [agent.state.vel]  # Own velocity
            + other_pos  # Other agents' positions
            + other_vel  # Other agents' velocities
            + discovered_target_pos  # Discovered targets' positions
            + [self.state.targets_discovered],  # Discovery status of all targets
            axis=-1,
        )
        return obs

    def done(self) -> Float[Array, f"{batch}"]:
        # Episode ends when all targets are discovered
        return self.state.total_discoveries >= self.n_targets
