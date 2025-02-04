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
n_obstacles = "n_obstacles"


@struct.dataclass
class SimpleTagState:
    """Dynamic state for Simple Tag scenario."""

    agent_distances: Float[Array, f"{batch} {n_agents} {n_agents}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1, 1))
    )
    obstacle_distances: Float[Array, f"{batch} {n_agents} {n_obstacles}"] = (
        struct.field(default_factory=lambda: jnp.zeros((1, 1, 1)))
    )
    tag_cooldowns: Float[Array, f"{batch} {n_agents}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1))
    )


class SimpleTag(BaseScenario):
    """
    Simple tag scenario where predator agents chase prey agents around
    obstacles. Predators are rewarded for catching prey, while prey are
    rewarded for escaping.
    """

    def __init__(self):
        super().__init__()
        self.n_predators = 3
        self.n_prey = 2
        self.n_agents = self.n_predators + self.n_prey
        self.n_obstacles = 4
        self.agent_size = 0.05
        self.obstacle_size = 0.2
        self.collision_penalty = 1.0
        self.arena_size = 1.0
        self.tag_reward = 10.0
        self.tag_cooldown = 10  # Time steps before predator can tag again
        self.tag_distance = 0.1  # Distance at which tagging occurs
        self.state = None

    def make_world(self, batch_dim: int, **kwargs) -> World:
        world = World(batch_dim=batch_dim, dim_p=2)

        # Add predator agents
        for i in range(self.n_predators):
            agent = Agent(name=f"predator_{i}", dynamics=Holonomic())
            agent.color = jnp.array([0.85, 0.35, 0.35])  # Red for predators
            agent.collision_penalty = True
            agent.size = self.agent_size
            agent.predator = True
            agent.max_speed = 1.0
            world.add_agent(agent)

        # Add prey agents
        for i in range(self.n_prey):
            agent = Agent(name=f"prey_{i}", dynamics=Holonomic())
            agent.color = jnp.array([0.35, 0.85, 0.35])  # Green for prey
            agent.collision_penalty = True
            agent.size = self.agent_size
            agent.predator = False
            agent.max_speed = 1.3  # Prey are faster than predators
            world.add_agent(agent)

        # Add obstacles
        for i in range(self.n_obstacles):
            obstacle = Entity(name=f"obstacle_{i}")
            obstacle.collide = True
            obstacle.movable = False
            obstacle.size = self.obstacle_size
            obstacle.color = jnp.array([0.25, 0.25, 0.25])
            world.add_landmark(obstacle)

        # Initialize scenario state
        self.state = SimpleTagState(
            agent_distances=jnp.zeros((batch_dim, self.n_agents, self.n_agents)),
            obstacle_distances=jnp.zeros((batch_dim, self.n_agents, self.n_obstacles)),
            tag_cooldowns=jnp.zeros((batch_dim, self.n_agents)),
        )

        return world

    def reset_world_at(self, env_index: int | None):
        batch_size = self.world.batch_dim if env_index is None else 1

        # Place agents randomly
        for agent in self.world.agents:
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

        # Place obstacles randomly
        for obstacle in self.world.landmarks:
            pos = jnp.array(
                [
                    jnp.random.uniform(-self.arena_size + 0.2, self.arena_size - 0.2),
                    jnp.random.uniform(-self.arena_size + 0.2, self.arena_size - 0.2),
                ]
            )
            obstacle.state.pos = jnp.where(
                env_index is None,
                jnp.tile(pos[None, :], (batch_size, 1)),
                pos,
            )

        # Reset state
        self.state = SimpleTagState(
            agent_distances=jnp.zeros((batch_size, self.n_agents, self.n_agents)),
            obstacle_distances=jnp.zeros((batch_size, self.n_agents, self.n_obstacles)),
            tag_cooldowns=jnp.zeros((batch_size, self.n_agents)),
        )

    def reward(self, agent: Agent) -> Float[Array, f"{batch}"]:
        reward = jnp.zeros(self.world.batch_dim)
        agent_idx = self.world.agents.index(agent)
        is_predator = agent_idx < self.n_predators

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

        # Update distances to obstacles
        for i, obstacle in enumerate(self.world.landmarks):
            dist = jnp.linalg.norm(agent.state.pos - obstacle.state.pos, axis=-1)
            self.state = self.state.replace(
                obstacle_distances=self.state.obstacle_distances.at[
                    :, agent_idx, i
                ].set(dist)
            )

        # Update tag cooldowns
        self.state = self.state.replace(
            tag_cooldowns=jnp.maximum(
                self.state.tag_cooldowns - 1,
                jnp.zeros_like(self.state.tag_cooldowns),
            )
        )

        if is_predator:
            # Predator rewards
            for prey_idx in range(self.n_predators, self.n_agents):
                dist = self.state.agent_distances[:, agent_idx, prey_idx]
                can_tag = (dist < self.tag_distance) & (
                    self.state.tag_cooldowns[:, agent_idx] <= 0
                )

                # Reward for tagging prey
                reward = jnp.where(
                    can_tag,
                    reward + self.tag_reward,
                    reward,
                )

                # Reset tag cooldown when tagging occurs
                self.state = self.state.replace(
                    tag_cooldowns=self.state.tag_cooldowns.at[:, agent_idx].set(
                        jnp.where(
                            can_tag,
                            self.tag_cooldown,
                            self.state.tag_cooldowns[:, agent_idx],
                        )
                    )
                )

            # Small negative reward for distance to nearest prey
            min_dist = jnp.min(
                self.state.agent_distances[:, agent_idx, self.n_predators :],
                axis=-1,
            )
            reward -= 0.1 * min_dist
        else:
            # Prey rewards
            # Reward for distance to nearest predator (want to maximize)
            min_dist = jnp.min(
                self.state.agent_distances[:, agent_idx, : self.n_predators],
                axis=-1,
            )
            reward += min_dist

            # Penalty for being tagged
            for pred_idx in range(self.n_predators):
                dist = self.state.agent_distances[:, agent_idx, pred_idx]
                can_tag = (dist < self.tag_distance) & (
                    self.state.tag_cooldowns[:, pred_idx] <= 0
                )
                reward = jnp.where(
                    can_tag,
                    reward - self.tag_reward,
                    reward,
                )

        # Collision penalties
        if agent.collision_penalty:
            # Agent-agent collisions
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

            # Agent-obstacle collisions
            for obstacle in self.world.landmarks:
                collision_dist = agent.size + obstacle.size
                dist = jnp.linalg.norm(agent.state.pos - obstacle.state.pos, axis=-1)
                reward = jnp.where(
                    dist < collision_dist,
                    reward - self.collision_penalty,
                    reward,
                )

        return reward

    def observation(self, agent: Agent) -> Float[Array, f"{batch} ..."]:
        agent_idx = self.world.agents.index(agent)
        is_predator = agent_idx < self.n_predators

        # Get positions and velocities of other agents relative to this agent
        other_pos = []
        other_vel = []
        for other in self.world.agents:
            if other is agent:
                continue
            other_pos.append(other.state.pos - agent.state.pos)
            other_vel.append(other.state.vel)

        # Get obstacle positions relative to agent
        obstacle_pos = []
        for obstacle in self.world.landmarks:
            obstacle_pos.append(obstacle.state.pos - agent.state.pos)

        # Stack observations
        obs = jnp.concatenate(
            [agent.state.pos]  # Own position
            + [agent.state.vel]  # Own velocity
            + [jnp.array([float(is_predator)])]  # Team identifier
            + [self.state.tag_cooldowns[:, agent_idx, None]]  # Tag cooldown
            + other_pos  # Other agents' positions
            + other_vel  # Other agents' velocities
            + obstacle_pos,  # Obstacle positions
            axis=-1,
        )

        return obs

    def done(self) -> Float[Array, f"{batch}"]:
        # Episode never ends
        return jnp.zeros(self.world.batch_dim, dtype=bool)
