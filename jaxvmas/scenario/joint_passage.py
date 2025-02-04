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
class JointPassageState:
    """Dynamic state for JointPassage scenario."""

    agent_distances: Float[Array, f"{batch} {n_agents} {n_agents}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1, 1))
    )
    agent_target_distances: Float[Array, f"{batch} {n_agents}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1))
    )
    passage_distances: Float[Array, f"{batch} {n_agents}"] = struct.field(
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


class JointPassage(BaseScenario):
    """
    Scenario where agents must coordinate to pass through narrow passages.
    Agents start on one side and must reach targets on the other side,
    but must navigate through passages that require coordination.
    """

    def __init__(self):
        super().__init__()
        self.n_agents = 4
        self.agent_size = 0.05
        self.target_size = 0.05
        self.wall_thickness = 0.05
        self.passage_width = 0.15  # Width of each passage
        self.collision_penalty = 1.0
        self.arena_size = 1.0
        self.target_reward = 10.0
        self.progress_reward_scale = 0.1
        self.target_threshold = 0.1  # Distance threshold for reaching target
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
        for i in range(self.n_agents):
            target = Entity(name=f"target_{i}")
            target.collide = False
            target.movable = False
            target.size = self.target_size
            target.color = jnp.array([0.25, 0.85, 0.25])  # Green targets
            world.add_landmark(target)

        # Add walls (as landmarks)
        # Central wall with two passages
        wall_center = Entity(name="wall_center")
        wall_center.collide = True
        wall_center.movable = False
        wall_center.size = self.wall_thickness
        wall_center.width = 2.0  # Full width wall
        wall_center.color = jnp.array([0.75, 0.75, 0.75])  # Gray walls
        world.add_landmark(wall_center)

        # Add passage markers (for visualization and distance calculation)
        passage1 = Entity(name="passage1")
        passage1.collide = False
        passage1.movable = False
        passage1.size = self.passage_width / 2
        passage1.color = jnp.array([0.95, 0.95, 0.95])  # Light gray passages
        world.add_landmark(passage1)

        passage2 = Entity(name="passage2")
        passage2.collide = False
        passage2.movable = False
        passage2.size = self.passage_width / 2
        passage2.color = jnp.array([0.95, 0.95, 0.95])
        world.add_landmark(passage2)

        # Initialize scenario state
        self.state = JointPassageState(
            agent_distances=jnp.zeros((batch_dim, self.n_agents, self.n_agents)),
            agent_target_distances=jnp.zeros((batch_dim, self.n_agents)),
            passage_distances=jnp.zeros((batch_dim, self.n_agents)),
            targets_reached=jnp.zeros((batch_dim, self.n_agents)),
            total_reached=jnp.zeros(batch_dim),
            progress=jnp.zeros((batch_dim, self.n_agents)),
        )

        return world

    def reset_world_at(self, env_index: int | None):
        batch_size = self.world.batch_dim if env_index is None else 1

        # Place wall in center
        wall = self.world.landmarks[0]
        wall.state.pos = jnp.zeros_like(wall.state.pos)

        # Place passages
        passage1 = self.world.landmarks[1]
        passage2 = self.world.landmarks[2]

        passage1.state.pos = jnp.where(
            env_index is None,
            jnp.tile(jnp.array([0.0, 0.4]), (batch_size, 1)),
            jnp.array([0.0, 0.4]),
        )

        passage2.state.pos = jnp.where(
            env_index is None,
            jnp.tile(jnp.array([0.0, -0.4]), (batch_size, 1)),
            jnp.array([0.0, -0.4]),
        )

        # Place agents on left side
        for i, agent in enumerate(self.world.agents):
            pos = jax.random.uniform(
                jax.random.PRNGKey(i),
                (batch_size if env_index is None else 1, 2),
                minval=jnp.array([-0.9, -0.9]),
                maxval=jnp.array([-0.3, 0.9]),
            )
            agent.state.pos = jnp.where(
                env_index is None,
                pos,
                pos[0],
            )
            agent.state.vel = jnp.zeros_like(agent.state.vel)

        # Place targets on right side
        for i, target in enumerate(self.world.landmarks[3 : 3 + self.n_agents]):
            pos = jax.random.uniform(
                jax.random.PRNGKey(i + self.n_agents),
                (batch_size if env_index is None else 1, 2),
                minval=jnp.array([0.3, -0.9]),
                maxval=jnp.array([0.9, 0.9]),
            )
            target.state.pos = jnp.where(
                env_index is None,
                pos,
                pos[0],
            )

        # Reset state
        self.state = JointPassageState(
            agent_distances=jnp.zeros((batch_size, self.n_agents, self.n_agents)),
            agent_target_distances=jnp.zeros((batch_size, self.n_agents)),
            passage_distances=jnp.zeros((batch_size, self.n_agents)),
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

        # Update distances to passages
        passage1 = self.world.landmarks[1]
        passage2 = self.world.landmarks[2]
        dist_to_passage1 = jnp.linalg.norm(
            agent.state.pos - passage1.state.pos, axis=-1
        )
        dist_to_passage2 = jnp.linalg.norm(
            agent.state.pos - passage2.state.pos, axis=-1
        )
        min_passage_dist = jnp.minimum(dist_to_passage1, dist_to_passage2)

        self.state = self.state.replace(
            passage_distances=self.state.passage_distances.at[:, agent_idx].set(
                min_passage_dist
            )
        )

        # Update distances to targets and check for reaching targets
        target = self.world.landmarks[3 + agent_idx]  # Offset by wall and passages
        dist_to_target = jnp.linalg.norm(agent.state.pos - target.state.pos, axis=-1)

        self.state = self.state.replace(
            agent_target_distances=self.state.agent_target_distances.at[
                :, agent_idx
            ].set(dist_to_target)
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

        # Calculate progress (reduction in distance to target)
        old_progress = self.state.progress[:, agent_idx]
        new_progress = (
            2.0 * self.arena_size - dist_to_target
        )  # Max distance is 2*arena_size
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

        # 2. Progress reward (weighted by passage proximity)
        # Encourage progress more when agent is near a passage
        passage_weight = 1.0 / (
            min_passage_dist + 1.0
        )  # Higher weight when closer to passage
        reward += self.progress_reward_scale * progress_delta * passage_weight

        # 3. Collision penalties
        if agent.collision_penalty:
            # Agent-agent collisions
            for other in self.world.agents:
                if other is agent:
                    continue
                collision_dist = agent.size + other.size
                dist = jnp.linalg.norm(agent.state.pos - other.state.pos, axis=-1)
                reward = jnp.where(
                    dist < collision_dist, reward - self.collision_penalty, reward
                )

            # Wall collision (more severe penalty)
            wall = self.world.landmarks[0]
            wall_dist = jnp.abs(agent.state.pos[:, 0])  # x-distance to wall
            if wall_dist < (self.wall_thickness + agent.size):
                # Check if not in passage
                y_pos = agent.state.pos[:, 1]
                in_passage1 = (
                    jnp.abs(y_pos - passage1.state.pos[:, 1]) < self.passage_width
                )
                in_passage2 = (
                    jnp.abs(y_pos - passage2.state.pos[:, 1]) < self.passage_width
                )
                in_passage = in_passage1 | in_passage2
                reward = jnp.where(
                    ~in_passage, reward - 2.0 * self.collision_penalty, reward
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

        # Get wall and passage positions relative to agent
        wall_pos = self.world.landmarks[0].state.pos - agent.state.pos
        passage1_pos = self.world.landmarks[1].state.pos - agent.state.pos
        passage2_pos = self.world.landmarks[2].state.pos - agent.state.pos

        # Get target position relative to agent
        target = self.world.landmarks[3 + agent_idx]
        target_pos = target.state.pos - agent.state.pos

        # Stack observations
        obs = jnp.concatenate(
            [agent.state.pos]  # Own position
            + [agent.state.vel]  # Own velocity
            + [target_pos]  # Target position
            + [wall_pos]  # Wall position
            + [passage1_pos]  # Passage 1 position
            + [passage2_pos]  # Passage 2 position
            + other_pos  # Other agents' positions
            + other_vel  # Other agents' velocities
            + [self.state.targets_reached],  # Target reached status
            axis=-1,
        )
        return obs

    def done(self) -> Float[Array, f"{batch}"]:
        # Episode ends when all agents reach their targets
        return self.state.total_reached >= self.n_agents
