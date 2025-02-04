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
n_adversaries = "n_adversaries"


@struct.dataclass
class AdversarialState:
    """Dynamic state for Adversarial scenario."""

    agent_distances: Float[Array, f"{batch} {n_agents} {n_agents}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1, 1))
    )
    target_distances: Float[Array, f"{batch} {n_agents}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1))
    )
    targets_reached: Float[Array, f"{batch} {n_agents}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1))
    )
    total_reached: Float[Array, f"{batch}"] = struct.field(
        default_factory=lambda: jnp.zeros((1,))
    )
    progress: Float[Array, f"{batch} {n_agents}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1))
    )


class Adversarial(BaseScenario):
    """
    Scenario where regular agents must reach their targets while adversarial
    agents try to block them. Regular agents get rewards for reaching targets,
    while adversaries get rewards for preventing target reaching.
    """

    def __init__(self):
        super().__init__()
        self.n_regular = 3  # Number of regular agents
        self.n_adversaries = 2  # Number of adversarial agents
        self.n_agents = self.n_regular + self.n_adversaries
        self.agent_size = 0.05
        self.target_size = 0.05
        self.collision_penalty = 1.0
        self.arena_size = 1.0
        self.target_reward = 10.0
        self.blocking_reward = 0.1  # Reward for adversaries blocking regular agents
        self.progress_reward_scale = 0.1
        self.target_threshold = 0.1
        self.blocking_distance = 0.2  # Distance at which blocking is effective
        self.state = None

    def make_world(self, batch_dim: int, **kwargs) -> World:
        world = World(batch_dim=batch_dim, dim_p=2)

        # Add regular agents
        for i in range(self.n_regular):
            agent = Agent(name=f"agent_{i}", dynamics=Holonomic())
            agent.color = jnp.array([0.35, 0.35, 0.85])  # Blue regular agents
            agent.collision_penalty = True
            agent.size = self.agent_size
            world.add_agent(agent)

        # Add adversarial agents
        for i in range(self.n_adversaries):
            agent = Agent(name=f"adversary_{i}", dynamics=Holonomic())
            agent.color = jnp.array([0.85, 0.35, 0.35])  # Red adversaries
            agent.collision_penalty = True
            agent.size = self.agent_size
            world.add_agent(agent)

        # Add targets for regular agents
        for i in range(self.n_regular):
            target = Entity(name=f"target_{i}")
            target.collide = False
            target.movable = False
            target.size = self.target_size
            target.color = jnp.array([0.25, 0.85, 0.25])
            world.add_landmark(target)

        # Initialize scenario state
        self.state = AdversarialState(
            agent_distances=jnp.zeros((batch_dim, self.n_agents, self.n_agents)),
            target_distances=jnp.zeros((batch_dim, self.n_regular)),
            targets_reached=jnp.zeros((batch_dim, self.n_regular)),
            total_reached=jnp.zeros(batch_dim),
            progress=jnp.zeros((batch_dim, self.n_regular)),
        )

        return world

    def reset_world_at(self, env_index: int | None):
        batch_size = self.world.batch_dim if env_index is None else 1

        # Place regular agents on the left side
        for i in range(self.n_regular):
            agent = self.world.agents[i]
            y_offset = 0.4 * (i - (self.n_regular - 1) / 2)
            pos = jnp.array([-self.arena_size + 0.1, y_offset])
            agent.state.pos = jnp.where(
                env_index is None,
                jnp.tile(pos[None, :], (batch_size, 1)),
                pos,
            )
            agent.state.vel = jnp.zeros_like(agent.state.vel)

        # Place adversaries in the middle
        for i in range(self.n_adversaries):
            agent = self.world.agents[self.n_regular + i]
            y_offset = 0.4 * (i - (self.n_adversaries - 1) / 2)
            pos = jnp.array([0.0, y_offset])
            agent.state.pos = jnp.where(
                env_index is None,
                jnp.tile(pos[None, :], (batch_size, 1)),
                pos,
            )
            agent.state.vel = jnp.zeros_like(agent.state.vel)

        # Place targets on the right side
        for i, target in enumerate(self.world.landmarks):
            y_offset = 0.4 * (i - (self.n_regular - 1) / 2)
            pos = jnp.array([self.arena_size - 0.1, y_offset])
            target.state.pos = jnp.where(
                env_index is None,
                jnp.tile(pos[None, :], (batch_size, 1)),
                pos,
            )

        # Reset state
        self.state = AdversarialState(
            agent_distances=jnp.zeros((batch_size, self.n_agents, self.n_agents)),
            target_distances=jnp.zeros((batch_size, self.n_regular)),
            targets_reached=jnp.zeros((batch_size, self.n_regular)),
            total_reached=jnp.zeros(batch_size),
            progress=jnp.zeros((batch_size, self.n_regular)),
        )

    def reward(self, agent: Agent) -> Float[Array, f"{batch}"]:
        reward = jnp.zeros(self.world.batch_dim)
        agent_idx = self.world.agents.index(agent)
        is_adversary = agent_idx >= self.n_regular

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

        if not is_adversary:
            # Regular agent rewards
            # Update distance to target
            target = self.world.landmarks[agent_idx]
            dist_to_target = jnp.linalg.norm(
                agent.state.pos - target.state.pos, axis=-1
            )
            self.state = self.state.replace(
                target_distances=self.state.target_distances.at[:, agent_idx].set(
                    dist_to_target
                )
            )

            # Check if target is reached
            reached_target = dist_to_target < self.target_threshold
            was_reached = self.state.targets_reached[:, agent_idx] > 0

            # Update target reached status
            self.state = self.state.replace(
                targets_reached=self.state.targets_reached.at[:, agent_idx].set(
                    jnp.where(
                        reached_target, 1.0, self.state.targets_reached[:, agent_idx]
                    )
                )
            )

            # Update total reached count
            self.state = self.state.replace(
                total_reached=jnp.sum(self.state.targets_reached, axis=-1)
            )

            # Calculate progress (x-position progress towards target)
            old_progress = self.state.progress[:, agent_idx]
            new_progress = agent.state.pos[:, 0] + self.arena_size
            progress_delta = new_progress - old_progress

            # Update progress
            self.state = self.state.replace(
                progress=self.state.progress.at[:, agent_idx].set(new_progress)
            )

            # Reward components for regular agents
            # 1. Target reaching reward
            reward = jnp.where(
                reached_target & ~was_reached,
                reward + self.target_reward,
                reward,
            )

            # 2. Progress reward
            reward += self.progress_reward_scale * progress_delta

            # 3. Penalty for being blocked by adversaries
            for adv_idx in range(self.n_regular, self.n_agents):
                adversary = self.world.agents[adv_idx]
                dist_to_adv = jnp.linalg.norm(
                    agent.state.pos - adversary.state.pos, axis=-1
                )
                reward = jnp.where(
                    dist_to_adv < self.blocking_distance,
                    reward - self.blocking_reward,
                    reward,
                )
        else:
            # Adversary rewards
            # Reward for blocking regular agents
            for reg_idx in range(self.n_regular):
                regular = self.world.agents[reg_idx]
                dist_to_reg = jnp.linalg.norm(
                    agent.state.pos - regular.state.pos, axis=-1
                )
                reward = jnp.where(
                    dist_to_reg < self.blocking_distance,
                    reward + self.blocking_reward,
                    reward,
                )

            # Penalty for regular agents reaching targets
            reward -= self.progress_reward_scale * jnp.sum(self.state.progress, axis=-1)

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
        agent_idx = self.world.agents.index(agent)
        is_adversary = agent_idx >= self.n_regular

        # Get positions and velocities of other agents relative to this agent
        other_pos = []
        other_vel = []
        for other in self.world.agents:
            if other is agent:
                continue
            other_pos.append(other.state.pos - agent.state.pos)
            other_vel.append(other.state.vel)

        if not is_adversary:
            # Regular agent observations
            # Get target position relative to agent
            target = self.world.landmarks[agent_idx]
            target_pos = target.state.pos - agent.state.pos

            # Stack observations
            obs = jnp.concatenate(
                [agent.state.pos]  # Own position
                + [agent.state.vel]  # Own velocity
                + [target_pos]  # Target position
                + other_pos  # Other agents' positions
                + other_vel  # Other agents' velocities
                + [jnp.array([0.0])],  # Team identifier (0 for regular)
                axis=-1,
            )
        else:
            # Adversary observations
            # Get all target positions relative to agent
            target_pos = []
            for target in self.world.landmarks:
                target_pos.append(target.state.pos - agent.state.pos)

            # Stack observations
            obs = jnp.concatenate(
                [agent.state.pos]  # Own position
                + [agent.state.vel]  # Own velocity
                + target_pos  # All target positions
                + other_pos  # Other agents' positions
                + other_vel  # Other agents' velocities
                + [jnp.array([1.0])],  # Team identifier (1 for adversary)
                axis=-1,
            )

        return obs

    def done(self) -> Float[Array, f"{batch}"]:
        # Episode ends when all regular agents reach their targets
        return self.state.total_reached >= self.n_regular
