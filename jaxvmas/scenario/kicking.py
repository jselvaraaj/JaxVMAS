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


@struct.dataclass
class KickingState:
    """Dynamic state for Kicking scenario."""

    agent_distances: Float[Array, f"{batch} {n_agents} {n_agents}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1, 1))
    )
    ball_distances: Float[Array, f"{batch} {n_agents}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1))
    )
    goal_distances: Float[Array, f"{batch} {n_agents}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1))
    )
    goals_scored: Float[Array, f"{batch}"] = struct.field(
        default_factory=lambda: jnp.zeros((1,))
    )
    ball_progress: Float[Array, f"{batch}"] = struct.field(
        default_factory=lambda: jnp.zeros((1,))
    )


class Kicking(BaseScenario):
    """
    Scenario where agents must kick a ball into a goal. Agents get rewards for
    scoring goals and making progress with the ball towards the goal.
    """

    def __init__(self):
        super().__init__()
        self.n_agents = 3
        self.agent_size = 0.05
        self.ball_size = 0.03
        self.goal_size = 0.2
        self.collision_penalty = 1.0
        self.arena_size = 1.0
        self.goal_reward = 10.0
        self.kick_reward = 0.5
        self.progress_reward_scale = 0.1
        self.goal_threshold = 0.1
        self.kick_threshold = 0.1
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

        # Add ball
        ball = Entity(name="ball", dynamics=Ball())
        ball.color = jnp.array([0.85, 0.85, 0.85])
        ball.collide = True
        ball.movable = True
        ball.size = self.ball_size
        world.add_landmark(ball)

        # Add goal posts
        for i in range(2):
            post = Entity(name=f"goal_post_{i}")
            post.collide = True
            post.movable = False
            post.size = self.agent_size
            post.color = jnp.array([0.85, 0.85, 0.35])
            world.add_landmark(post)

        # Initialize scenario state
        self.state = KickingState(
            agent_distances=jnp.zeros((batch_dim, self.n_agents, self.n_agents)),
            ball_distances=jnp.zeros((batch_dim, self.n_agents)),
            goal_distances=jnp.zeros((batch_dim, self.n_agents)),
            goals_scored=jnp.zeros(batch_dim),
            ball_progress=jnp.zeros(batch_dim),
        )

        return world

    def reset_world_at(self, env_index: int | None):
        batch_size = self.world.batch_dim if env_index is None else 1

        # Place agents in a semi-circle on the left side
        for i in range(self.n_agents):
            agent = self.world.agents[i]
            angle = jnp.pi * (i - (self.n_agents - 1) / 2) / (2 * self.n_agents)
            radius = 0.3
            pos = jnp.array(
                [
                    -self.arena_size + 0.2 + radius * jnp.cos(angle),
                    radius * jnp.sin(angle),
                ]
            )
            agent.state.pos = jnp.where(
                env_index is None,
                jnp.tile(pos[None, :], (batch_size, 1)),
                pos,
            )
            agent.state.vel = jnp.zeros_like(agent.state.vel)

        # Place ball in the middle
        ball = self.world.landmarks[0]
        pos = jnp.array([-0.5, 0.0])
        ball.state.pos = jnp.where(
            env_index is None,
            jnp.tile(pos[None, :], (batch_size, 1)),
            pos,
        )
        ball.state.vel = jnp.zeros_like(ball.state.vel)

        # Place goal posts
        goal_y = self.goal_size / 2
        for i, post in enumerate(self.world.landmarks[1:]):
            pos = jnp.array(
                [
                    self.arena_size - 0.1,
                    goal_y if i == 0 else -goal_y,
                ]
            )
            post.state.pos = jnp.where(
                env_index is None,
                jnp.tile(pos[None, :], (batch_size, 1)),
                pos,
            )

        # Reset state
        self.state = KickingState(
            agent_distances=jnp.zeros((batch_size, self.n_agents, self.n_agents)),
            ball_distances=jnp.zeros((batch_size, self.n_agents)),
            goal_distances=jnp.zeros((batch_size, self.n_agents)),
            goals_scored=jnp.zeros(batch_size),
            ball_progress=jnp.zeros(batch_size),
        )

    def reward(self, agent: Agent) -> Float[Array, f"{batch}"]:
        reward = jnp.zeros(self.world.batch_dim)
        agent_idx = self.world.agents.index(agent)
        ball = self.world.landmarks[0]
        goal_posts = self.world.landmarks[1:]

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

        # Update distance to ball
        dist_to_ball = jnp.linalg.norm(agent.state.pos - ball.state.pos, axis=-1)
        self.state = self.state.replace(
            ball_distances=self.state.ball_distances.at[:, agent_idx].set(dist_to_ball)
        )

        # Calculate ball progress (x-position progress towards goal)
        old_progress = self.state.ball_progress
        new_progress = ball.state.pos[:, 0] + self.arena_size
        progress_delta = new_progress - old_progress

        # Update ball progress
        self.state = self.state.replace(ball_progress=new_progress)

        # Check if ball is between goal posts
        goal_y = (goal_posts[0].state.pos[:, 1] + goal_posts[1].state.pos[:, 1]) / 2
        goal_x = goal_posts[0].state.pos[:, 0]
        ball_in_goal = (ball.state.pos[:, 0] >= goal_x - self.goal_threshold) & (
            jnp.abs(ball.state.pos[:, 1] - goal_y) <= self.goal_size / 2
        )

        # Update goals scored
        self.state = self.state.replace(
            goals_scored=jnp.where(
                ball_in_goal,
                1.0,
                self.state.goals_scored,
            )
        )

        # Reward components
        # 1. Goal scoring reward
        reward = jnp.where(
            ball_in_goal & (self.state.goals_scored == 1.0),
            reward + self.goal_reward,
            reward,
        )

        # 2. Kick reward (when agent is close to ball and ball has velocity)
        ball_speed = jnp.linalg.norm(ball.state.vel, axis=-1)
        reward = jnp.where(
            (dist_to_ball < self.kick_threshold) & (ball_speed > 0),
            reward + self.kick_reward * ball_speed,
            reward,
        )

        # 3. Progress reward
        reward += self.progress_reward_scale * progress_delta

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

        # Get ball position and velocity relative to agent
        ball = self.world.landmarks[0]
        ball_pos = ball.state.pos - agent.state.pos
        ball_vel = ball.state.vel

        # Get goal posts positions relative to agent
        goal_posts = self.world.landmarks[1:]
        goal_pos = []
        for post in goal_posts:
            goal_pos.append(post.state.pos - agent.state.pos)

        # Stack observations
        obs = jnp.concatenate(
            [agent.state.pos]  # Own position
            + [agent.state.vel]  # Own velocity
            + [ball_pos]  # Ball position
            + [ball_vel]  # Ball velocity
            + goal_pos  # Goal posts positions
            + other_pos  # Other agents' positions
            + other_vel,  # Other agents' velocities
            axis=-1,
        )

        return obs

    def done(self) -> Float[Array, f"{batch}"]:
        # Episode ends when a goal is scored
        return self.state.goals_scored > 0
