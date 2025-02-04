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
n_crowd = "n_crowd"
n_obstacles = "n_obstacles"


@struct.dataclass
class CrowdState:
    """Dynamic state for Crowd scenario."""

    agent_distances: Float[Array, f"{batch} {n_agents} {n_agents}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1, 1))
    )
    crowd_distances: Float[Array, f"{batch} {n_agents} {n_crowd}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1, 1))
    )
    obstacle_distances: Float[Array, f"{batch} {n_agents} {n_obstacles}"] = (
        struct.field(default_factory=lambda: jnp.zeros((1, 1, 1)))
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


class Crowd(BaseScenario):
    """
    Scenario where agents must navigate through a crowded environment
    with static obstacles and moving crowd agents to reach their targets.
    """

    def __init__(self):
        super().__init__()
        self.n_agents = 3
        self.n_crowd = 10  # Number of crowd agents
        self.n_obstacles = 5  # Number of static obstacles
        self.agent_size = 0.05
        self.crowd_size = 0.05
        self.obstacle_size = 0.1
        self.target_size = 0.05
        self.collision_penalty = 1.0
        self.crowd_penalty = 2.0
        self.obstacle_penalty = 3.0
        self.arena_size = 1.0
        self.target_reward = 10.0
        self.progress_reward_scale = 0.1
        self.target_threshold = 0.1
        self.crowd_speed = 0.05  # Speed of crowd agents
        self.crowd_period = 4.0  # Period of crowd movement
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

        # Add crowd agents (as landmarks)
        for i in range(self.n_crowd):
            crowd = Entity(name=f"crowd_{i}")
            crowd.collide = True
            crowd.movable = True
            crowd.size = self.crowd_size
            crowd.color = jnp.array([0.85, 0.85, 0.35])
            world.add_landmark(crowd)

        # Add static obstacles
        for i in range(self.n_obstacles):
            obstacle = Entity(name=f"obstacle_{i}")
            obstacle.collide = True
            obstacle.movable = False
            obstacle.size = self.obstacle_size
            obstacle.color = jnp.array([0.25, 0.25, 0.25])
            world.add_landmark(obstacle)

        # Add targets
        for i in range(self.n_agents):
            target = Entity(name=f"target_{i}")
            target.collide = False
            target.movable = False
            target.size = self.target_size
            target.color = jnp.array([0.25, 0.85, 0.25])
            world.add_landmark(target)

        # Initialize scenario state
        self.state = CrowdState(
            agent_distances=jnp.zeros((batch_dim, self.n_agents, self.n_agents)),
            crowd_distances=jnp.zeros((batch_dim, self.n_agents, self.n_crowd)),
            obstacle_distances=jnp.zeros((batch_dim, self.n_agents, self.n_obstacles)),
            target_distances=jnp.zeros((batch_dim, self.n_agents)),
            targets_reached=jnp.zeros((batch_dim, self.n_agents)),
            total_reached=jnp.zeros(batch_dim),
            progress=jnp.zeros((batch_dim, self.n_agents)),
        )

        return world

    def reset_world_at(self, env_index: int | None):
        batch_size = self.world.batch_dim if env_index is None else 1

        # Place agents on the left side
        for i, agent in enumerate(self.world.agents):
            y_offset = 0.4 * (i - (self.n_agents - 1) / 2)
            pos = jnp.array([-self.arena_size + 0.1, y_offset])
            agent.state.pos = jnp.where(
                env_index is None,
                jnp.tile(pos[None, :], (batch_size, 1)),
                pos,
            )
            agent.state.vel = jnp.zeros_like(agent.state.vel)

        # Place crowd agents in a grid pattern
        crowd_start = self.n_agents
        crowd_end = crowd_start + self.n_crowd
        for i, crowd in enumerate(self.world.landmarks[crowd_start:crowd_end]):
            x = -0.5 + (i % 5) * 0.25
            y = -0.5 + (i // 5) * 0.25
            pos = jnp.array([x, y])
            crowd.state.pos = jnp.where(
                env_index is None,
                jnp.tile(pos[None, :], (batch_size, 1)),
                pos,
            )
            # Initialize crowd velocities for circular motion
            angle = 2 * jnp.pi * i / self.n_crowd
            vel = self.crowd_speed * jnp.array([jnp.cos(angle), jnp.sin(angle)])
            crowd.state.vel = jnp.where(
                env_index is None,
                jnp.tile(vel[None, :], (batch_size, 1)),
                vel,
            )

        # Place static obstacles
        obstacle_start = crowd_end
        obstacle_end = obstacle_start + self.n_obstacles
        for i, obstacle in enumerate(self.world.landmarks[obstacle_start:obstacle_end]):
            angle = 2 * jnp.pi * i / self.n_obstacles
            radius = 0.6
            pos = jnp.array(
                [
                    radius * jnp.cos(angle),
                    radius * jnp.sin(angle),
                ]
            )
            obstacle.state.pos = jnp.where(
                env_index is None,
                jnp.tile(pos[None, :], (batch_size, 1)),
                pos,
            )

        # Place targets on the right side
        for i, target in enumerate(self.world.landmarks[obstacle_end:]):
            y_offset = 0.4 * (i - (self.n_agents - 1) / 2)
            pos = jnp.array([self.arena_size - 0.1, y_offset])
            target.state.pos = jnp.where(
                env_index is None,
                jnp.tile(pos[None, :], (batch_size, 1)),
                pos,
            )

        # Reset state
        self.state = CrowdState(
            agent_distances=jnp.zeros((batch_size, self.n_agents, self.n_agents)),
            crowd_distances=jnp.zeros((batch_size, self.n_agents, self.n_crowd)),
            obstacle_distances=jnp.zeros((batch_size, self.n_agents, self.n_obstacles)),
            target_distances=jnp.zeros((batch_size, self.n_agents)),
            targets_reached=jnp.zeros((batch_size, self.n_agents)),
            total_reached=jnp.zeros(batch_size),
            progress=jnp.zeros((batch_size, self.n_agents)),
        )

    def reward(self, agent: Agent) -> Float[Array, f"{batch}"]:
        reward = jnp.zeros(self.world.batch_dim)
        agent_idx = self.world.agents.index(agent)

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

        # Update distances to crowd agents and apply penalties
        crowd_start = self.n_agents
        crowd_end = crowd_start + self.n_crowd
        for i, crowd in enumerate(self.world.landmarks[crowd_start:crowd_end]):
            dist = jnp.linalg.norm(agent.state.pos - crowd.state.pos, axis=-1)
            self.state = self.state.replace(
                crowd_distances=self.state.crowd_distances.at[:, agent_idx, i].set(dist)
            )
            # Apply crowd collision penalty
            collision_dist = agent.size + crowd.size
            reward = jnp.where(
                dist < collision_dist,
                reward - self.crowd_penalty,
                reward,
            )

        # Update distances to obstacles and apply penalties
        obstacle_start = crowd_end
        obstacle_end = obstacle_start + self.n_obstacles
        for i, obstacle in enumerate(self.world.landmarks[obstacle_start:obstacle_end]):
            dist = jnp.linalg.norm(agent.state.pos - obstacle.state.pos, axis=-1)
            self.state = self.state.replace(
                obstacle_distances=self.state.obstacle_distances.at[
                    :, agent_idx, i
                ].set(dist)
            )
            # Apply obstacle collision penalty
            collision_dist = agent.size + obstacle.size
            reward = jnp.where(
                dist < collision_dist,
                reward - self.obstacle_penalty,
                reward,
            )

        # Update distances to targets and check for reaching
        target = self.world.landmarks[obstacle_end + agent_idx]
        dist_to_target = jnp.linalg.norm(agent.state.pos - target.state.pos, axis=-1)
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
                jnp.where(reached_target, 1.0, self.state.targets_reached[:, agent_idx])
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

        # Reward components
        # 1. Target reaching reward
        reward = jnp.where(
            reached_target & ~was_reached,
            reward + self.target_reward,
            reward,
        )

        # 2. Progress reward
        reward += self.progress_reward_scale * progress_delta

        # 3. Agent collision penalties
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

        # Get positions and velocities of other agents relative to this agent
        other_pos = []
        other_vel = []
        for other in self.world.agents:
            if other is agent:
                continue
            other_pos.append(other.state.pos - agent.state.pos)
            other_vel.append(other.state.vel)

        # Get crowd positions and velocities relative to agent
        crowd_pos = []
        crowd_vel = []
        crowd_start = self.n_agents
        crowd_end = crowd_start + self.n_crowd
        for crowd in self.world.landmarks[crowd_start:crowd_end]:
            crowd_pos.append(crowd.state.pos - agent.state.pos)
            crowd_vel.append(crowd.state.vel)

        # Get obstacle positions relative to agent
        obstacle_pos = []
        obstacle_start = crowd_end
        obstacle_end = obstacle_start + self.n_obstacles
        for obstacle in self.world.landmarks[obstacle_start:obstacle_end]:
            obstacle_pos.append(obstacle.state.pos - agent.state.pos)

        # Get target position relative to agent
        target = self.world.landmarks[obstacle_end + agent_idx]
        target_pos = target.state.pos - agent.state.pos

        # Stack observations
        obs = jnp.concatenate(
            [agent.state.pos]  # Own position
            + [agent.state.vel]  # Own velocity
            + [target_pos]  # Target position
            + crowd_pos  # Crowd positions
            + crowd_vel  # Crowd velocities
            + obstacle_pos  # Obstacle positions
            + other_pos  # Other agents' positions
            + other_vel,  # Other agents' velocities
            axis=-1,
        )
        return obs

    def done(self) -> Float[Array, f"{batch}"]:
        # Episode ends when all agents reach their targets
        return self.state.total_reached >= self.n_agents

    def update_world(self):
        # Update crowd agent positions with circular motion
        crowd_start = self.n_agents
        crowd_end = crowd_start + self.n_crowd
        for i, crowd in enumerate(self.world.landmarks[crowd_start:crowd_end]):
            # Calculate new velocity for circular motion
            angle = 2 * jnp.pi * i / self.n_crowd
            phase = (self.world.t / self.crowd_period) * 2 * jnp.pi
            vel = self.crowd_speed * jnp.array(
                [
                    jnp.cos(angle + phase),
                    jnp.sin(angle + phase),
                ]
            )
            crowd.state.vel = jnp.tile(vel[None, :], (self.world.batch_dim, 1))
