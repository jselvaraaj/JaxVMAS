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
n_checkpoints = "n_checkpoints"


@struct.dataclass
class RaceState:
    """Dynamic state for Race scenario."""

    agent_distances: Float[Array, f"{batch} {n_agents} {n_agents}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1, 1))
    )
    checkpoint_distances: Float[Array, f"{batch} {n_agents} {n_checkpoints}"] = (
        struct.field(default_factory=lambda: jnp.zeros((1, 1, 1)))
    )
    checkpoints_reached: Float[Array, f"{batch} {n_agents} {n_checkpoints}"] = (
        struct.field(default_factory=lambda: jnp.zeros((1, 1, 1)))
    )
    current_checkpoint: Float[Array, f"{batch} {n_agents}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1))
    )
    race_progress: Float[Array, f"{batch} {n_agents}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1))
    )
    race_finished: Float[Array, f"{batch}"] = struct.field(
        default_factory=lambda: jnp.zeros((1,))
    )
    race_position: Float[Array, f"{batch} {n_agents}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1))
    )


class Race(BaseScenario):
    """
    Scenario where agents compete in a race through checkpoints to reach
    their targets. Agents must navigate around obstacles and are rewarded
    based on their race position and progress.
    """

    def __init__(self):
        super().__init__()
        self.n_agents = 4
        self.n_checkpoints = 6
        self.n_obstacles = 8
        self.agent_size = 0.05
        self.checkpoint_size = 0.05
        self.obstacle_size = 0.1
        self.collision_penalty = 1.0
        self.arena_size = 1.0
        self.checkpoint_reward = 5.0
        self.finish_reward = 10.0
        self.position_reward_scale = 1.0
        self.progress_reward_scale = 0.1
        self.checkpoint_threshold = 0.1
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

        # Add checkpoints
        for i in range(self.n_checkpoints):
            checkpoint = Entity(name=f"checkpoint_{i}")
            checkpoint.collide = False
            checkpoint.movable = False
            checkpoint.size = self.checkpoint_size
            checkpoint.color = jnp.array([0.25, 0.85, 0.25])
            world.add_landmark(checkpoint)

        # Add obstacles
        for i in range(self.n_obstacles):
            obstacle = Entity(name=f"obstacle_{i}")
            obstacle.collide = True
            obstacle.movable = False
            obstacle.size = self.obstacle_size
            obstacle.color = jnp.array([0.85, 0.35, 0.35])
            world.add_landmark(obstacle)

        # Initialize scenario state
        self.state = RaceState(
            agent_distances=jnp.zeros((batch_dim, self.n_agents, self.n_agents)),
            checkpoint_distances=jnp.zeros(
                (batch_dim, self.n_agents, self.n_checkpoints)
            ),
            checkpoints_reached=jnp.zeros(
                (batch_dim, self.n_agents, self.n_checkpoints)
            ),
            current_checkpoint=jnp.zeros((batch_dim, self.n_agents)),
            race_progress=jnp.zeros((batch_dim, self.n_agents)),
            race_finished=jnp.zeros(batch_dim),
            race_position=jnp.zeros((batch_dim, self.n_agents)),
        )

        return world

    def reset_world_at(self, env_index: int | None):
        batch_size = self.world.batch_dim if env_index is None else 1

        # Place agents at starting line
        for i, agent in enumerate(self.world.agents):
            y_offset = 0.4 * (i - (self.n_agents - 1) / 2)
            pos = jnp.array([-self.arena_size + 0.1, y_offset])
            agent.state.pos = jnp.where(
                env_index is None,
                jnp.tile(pos[None, :], (batch_size, 1)),
                pos,
            )
            agent.state.vel = jnp.zeros_like(agent.state.vel)

        # Place checkpoints in a curved path
        for i, checkpoint in enumerate(self.world.landmarks[: self.n_checkpoints]):
            t = i / (self.n_checkpoints - 1)
            x = -self.arena_size + 0.2 + 1.6 * t * self.arena_size
            y = 0.5 * jnp.sin(2 * jnp.pi * t)
            pos = jnp.array([x, y])
            checkpoint.state.pos = jnp.where(
                env_index is None,
                jnp.tile(pos[None, :], (batch_size, 1)),
                pos,
            )

        # Place obstacles around the track
        obstacles = self.world.landmarks[self.n_checkpoints :]
        for i, obstacle in enumerate(obstacles):
            t = (i + 0.5) / self.n_obstacles
            x = -self.arena_size + 0.3 + 1.4 * t * self.arena_size
            y = 0.7 * jnp.sin(2 * jnp.pi * t + jnp.pi)
            pos = jnp.array([x, y])
            obstacle.state.pos = jnp.where(
                env_index is None,
                jnp.tile(pos[None, :], (batch_size, 1)),
                pos,
            )

        # Reset state
        self.state = RaceState(
            agent_distances=jnp.zeros((batch_size, self.n_agents, self.n_agents)),
            checkpoint_distances=jnp.zeros(
                (batch_size, self.n_agents, self.n_checkpoints)
            ),
            checkpoints_reached=jnp.zeros(
                (batch_size, self.n_agents, self.n_checkpoints)
            ),
            current_checkpoint=jnp.zeros((batch_size, self.n_agents)),
            race_progress=jnp.zeros((batch_size, self.n_agents)),
            race_finished=jnp.zeros(batch_size),
            race_position=jnp.zeros((batch_size, self.n_agents)),
        )

    def reward(self, agent: Agent) -> Float[Array, f"{batch}"]:
        reward = jnp.zeros(self.world.batch_dim)
        agent_idx = self.world.agents.index(agent)
        checkpoints = self.world.landmarks[: self.n_checkpoints]
        obstacles = self.world.landmarks[self.n_checkpoints :]

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

        # Update distances to checkpoints
        for i, checkpoint in enumerate(checkpoints):
            dist = jnp.linalg.norm(agent.state.pos - checkpoint.state.pos, axis=-1)
            self.state = self.state.replace(
                checkpoint_distances=self.state.checkpoint_distances.at[
                    :, agent_idx, i
                ].set(dist)
            )

        # Check if current checkpoint is reached
        current_cp = self.state.current_checkpoint[:, agent_idx].astype(jnp.int32)
        dist_to_current = self.state.checkpoint_distances[
            jnp.arange(self.world.batch_dim), agent_idx, current_cp
        ]
        checkpoint_reached = dist_to_current < self.checkpoint_threshold

        # Update checkpoints reached
        self.state = self.state.replace(
            checkpoints_reached=self.state.checkpoints_reached.at[
                jnp.arange(self.world.batch_dim), agent_idx, current_cp
            ].set(jnp.where(checkpoint_reached, 1.0, 0.0))
        )

        # Update current checkpoint
        self.state = self.state.replace(
            current_checkpoint=self.state.current_checkpoint.at[:, agent_idx].set(
                jnp.where(
                    checkpoint_reached,
                    jnp.minimum(current_cp + 1, self.n_checkpoints - 1),
                    current_cp,
                )
            )
        )

        # Calculate race progress
        progress = (
            self.state.current_checkpoint[:, agent_idx]
            + (1 - dist_to_current / (2 * self.arena_size))
        ) / self.n_checkpoints
        self.state = self.state.replace(
            race_progress=self.state.race_progress.at[:, agent_idx].set(progress)
        )

        # Update race positions (sorted by progress)
        sorted_indices = jnp.argsort(-self.state.race_progress, axis=1)
        positions = jnp.zeros_like(sorted_indices)
        positions = positions.at[
            jnp.arange(self.world.batch_dim)[:, None], sorted_indices
        ].set(jnp.arange(self.n_agents))
        self.state = self.state.replace(race_position=positions)

        # Check if race is finished
        finished = jnp.all(self.state.checkpoints_reached[:, agent_idx] > 0, axis=-1)
        self.state = self.state.replace(
            race_finished=jnp.where(
                finished,
                1.0,
                self.state.race_finished,
            )
        )

        # Reward components
        # 1. Checkpoint reward
        reward = jnp.where(
            checkpoint_reached,
            reward + self.checkpoint_reward,
            reward,
        )

        # 2. Finish reward
        reward = jnp.where(
            finished & (self.state.race_finished == 1.0),
            reward
            + self.finish_reward
            * (self.n_agents - self.state.race_position[:, agent_idx]),
            reward,
        )

        # 3. Progress reward
        reward += self.progress_reward_scale * (
            progress - self.state.race_progress[:, agent_idx]
        )

        # 4. Position reward
        reward -= self.position_reward_scale * self.state.race_position[:, agent_idx]

        # Collision penalties
        if agent.collision_penalty:
            # Agent collisions
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

            # Obstacle collisions
            for obstacle in obstacles:
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
        checkpoints = self.world.landmarks[: self.n_checkpoints]
        obstacles = self.world.landmarks[self.n_checkpoints :]

        # Get positions and velocities of other agents relative to this agent
        other_pos = []
        other_vel = []
        for other in self.world.agents:
            if other is agent:
                continue
            other_pos.append(other.state.pos - agent.state.pos)
            other_vel.append(other.state.vel)

        # Get checkpoint positions relative to agent
        checkpoint_pos = []
        for checkpoint in checkpoints:
            checkpoint_pos.append(checkpoint.state.pos - agent.state.pos)

        # Get obstacle positions relative to agent
        obstacle_pos = []
        for obstacle in obstacles:
            obstacle_pos.append(obstacle.state.pos - agent.state.pos)

        # Stack observations
        obs = jnp.concatenate(
            [agent.state.pos]  # Own position
            + [agent.state.vel]  # Own velocity
            + [self.state.current_checkpoint[:, agent_idx, None]]  # Current checkpoint
            + [self.state.race_position[:, agent_idx, None]]  # Race position
            + checkpoint_pos  # Checkpoint positions
            + obstacle_pos  # Obstacle positions
            + other_pos  # Other agents' positions
            + other_vel,  # Other agents' velocities
            axis=-1,
        )

        return obs

    def done(self) -> Float[Array, f"{batch}"]:
        # Episode ends when all agents finish the race
        return jnp.all(
            jnp.sum(self.state.checkpoints_reached, axis=-1) == self.n_checkpoints,
            axis=-1,
        )
