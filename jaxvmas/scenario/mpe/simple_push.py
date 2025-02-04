#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import jax.numpy as jnp
from flax import struct
from jaxtyping import Array, Float

from jaxvmas.simulator.core import Agent, Entity, World
from jaxvmas.simulator.dynamics.ball import Ball
from jaxvmas.simulator.dynamics.holonomic import Holonomic
from jaxvmas.simulator.scenario import BaseScenario

# Type dimensions
batch = "batch"
n_agents = "n_agents"
n_objects = "n_objects"


@struct.dataclass
class SimplePushState:
    """Dynamic state for Simple Push scenario."""

    agent_distances: Float[Array, f"{batch} {n_agents} {n_agents}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1, 1))
    )
    object_distances: Float[Array, f"{batch} {n_agents} {n_objects}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1, 1))
    )
    goal_distances: Float[Array, f"{batch} {n_objects}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1))
    )


class SimplePush(BaseScenario):
    """
    Simple push scenario where agents must push objects to reach goals.
    Objects have physical properties and can be moved by agents.
    """

    def __init__(self):
        super().__init__()
        self.n_agents = 2
        self.n_objects = 1
        self.agent_size = 0.05
        self.object_size = 0.05
        self.goal_size = 0.05
        self.collision_penalty = 1.0
        self.arena_size = 1.0
        self.goal_reward = 1.0
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

        # Add objects (with ball dynamics)
        for i in range(self.n_objects):
            obj = Entity(name=f"object_{i}", dynamics=Ball())
            obj.color = jnp.array([0.25, 0.25, 0.25])
            obj.collide = True
            obj.movable = True
            obj.size = self.object_size
            world.add_landmark(obj)

        # Add goals (static landmarks)
        for i in range(self.n_objects):
            goal = Entity(name=f"goal_{i}")
            goal.collide = False
            goal.movable = False
            goal.size = self.goal_size
            goal.color = jnp.array([0.25, 0.85, 0.25])
            world.add_landmark(goal)

        # Initialize scenario state
        self.state = SimplePushState(
            agent_distances=jnp.zeros((batch_dim, self.n_agents, self.n_agents)),
            object_distances=jnp.zeros((batch_dim, self.n_agents, self.n_objects)),
            goal_distances=jnp.zeros((batch_dim, self.n_objects)),
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

        # Place objects randomly
        objects = self.world.landmarks[: self.n_objects]
        for obj in objects:
            pos = jnp.array(
                [
                    jnp.random.uniform(-self.arena_size + 0.1, self.arena_size - 0.1),
                    jnp.random.uniform(-self.arena_size + 0.1, self.arena_size - 0.1),
                ]
            )
            obj.state.pos = jnp.where(
                env_index is None,
                jnp.tile(pos[None, :], (batch_size, 1)),
                pos,
            )
            obj.state.vel = jnp.zeros_like(obj.state.vel)

        # Place goals randomly
        goals = self.world.landmarks[self.n_objects :]
        for goal in goals:
            pos = jnp.array(
                [
                    jnp.random.uniform(-self.arena_size + 0.1, self.arena_size - 0.1),
                    jnp.random.uniform(-self.arena_size + 0.1, self.arena_size - 0.1),
                ]
            )
            goal.state.pos = jnp.where(
                env_index is None,
                jnp.tile(pos[None, :], (batch_size, 1)),
                pos,
            )

        # Reset state
        self.state = SimplePushState(
            agent_distances=jnp.zeros((batch_size, self.n_agents, self.n_agents)),
            object_distances=jnp.zeros((batch_size, self.n_agents, self.n_objects)),
            goal_distances=jnp.zeros((batch_size, self.n_objects)),
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

        # Update distances between agents and objects
        objects = self.world.landmarks[: self.n_objects]
        for i, obj in enumerate(objects):
            dist = jnp.linalg.norm(agent.state.pos - obj.state.pos, axis=-1)
            self.state = self.state.replace(
                object_distances=self.state.object_distances.at[:, agent_idx, i].set(
                    dist
                )
            )

        # Update distances between objects and goals
        goals = self.world.landmarks[self.n_objects :]
        for i, (obj, goal) in enumerate(zip(objects, goals)):
            dist = jnp.linalg.norm(obj.state.pos - goal.state.pos, axis=-1)
            self.state = self.state.replace(
                goal_distances=self.state.goal_distances.at[:, i].set(dist)
            )

        # Reward is negative sum of distances between objects and goals
        reward -= jnp.sum(self.state.goal_distances, axis=-1)

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

            # Agent-object collisions (reduced penalty to encourage pushing)
            for obj in objects:
                collision_dist = agent.size + obj.size
                dist = jnp.linalg.norm(agent.state.pos - obj.state.pos, axis=-1)
                reward = jnp.where(
                    dist < collision_dist,
                    reward - 0.1 * self.collision_penalty,
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

        # Get object positions and velocities relative to agent
        objects = self.world.landmarks[: self.n_objects]
        object_pos = []
        object_vel = []
        for obj in objects:
            object_pos.append(obj.state.pos - agent.state.pos)
            object_vel.append(obj.state.vel)

        # Get goal positions relative to agent
        goals = self.world.landmarks[self.n_objects :]
        goal_pos = []
        for goal in goals:
            goal_pos.append(goal.state.pos - agent.state.pos)

        # Stack observations
        obs = jnp.concatenate(
            [agent.state.pos]  # Own position
            + [agent.state.vel]  # Own velocity
            + other_pos  # Other agents' positions
            + other_vel  # Other agents' velocities
            + object_pos  # Object positions
            + object_vel  # Object velocities
            + goal_pos,  # Goal positions
            axis=-1,
        )

        return obs

    def done(self) -> Float[Array, f"{batch}"]:
        # Episode never ends
        return jnp.zeros(self.world.batch_dim, dtype=bool)
