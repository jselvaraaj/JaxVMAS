#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import jax.numpy as jnp
from flax import struct
from jaxtyping import Array, Float

from jaxvmas.simulator.core import Agent, Entity, World
from jaxvmas.simulator.dynamics.holonomic import Holonomic
from jaxvmas.simulator.scenario import BaseScenario
from jaxvmas.simulator.utils import ScenarioUtils

# Type dimensions
batch = "batch"
n_agents = "n_agents"
n_teams = "n_teams"


@struct.dataclass
class FootballState:
    """Dynamic state for Football scenario."""

    ball_possession: Float[Array, f"{batch} {n_teams}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 2))
    )
    agent_ball_distances: Float[Array, f"{batch} {n_agents}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 1))
    )
    team_scores: Float[Array, f"{batch} {n_teams}"] = struct.field(
        default_factory=lambda: jnp.zeros((1, 2))
    )
    ball_touched: Float[Array, f"{batch}"] = struct.field(
        default_factory=lambda: jnp.zeros((1,))
    )


class Football(BaseScenario):
    """
    Scenario where two teams of agents play a simplified version of football.
    Each team tries to score goals while preventing the other team from scoring.
    """

    def init_params(self, **kwargs):
        # Scenario config
        self.viewer_size = kwargs.pop("viewer_size", (1200, 800))

        # Agents config
        self.n_blue_agents = kwargs.pop("n_blue_agents", 3)
        self.n_red_agents = kwargs.pop("n_red_agents", 3)
        # What agents should be learning and what controlled by the heuristic (ai)
        self.ai_red_agents = kwargs.pop("ai_red_agents", True)
        self.ai_blue_agents = kwargs.pop("ai_blue_agents", False)

        # When you have 5 blue agents there is the options of introducing physical differences with the following roles:
        # 1 goalkeeper -> slow and big
        # 2 defenders -> normal size and speed (agent_size, u_multiplier, max_speed)
        # 2 attackers -> small and fast
        self.physically_different = kwargs.pop("physically_different", False)

        # Agent spawning
        self.spawn_in_formation = kwargs.pop("spawn_in_formation", False)
        self.only_blue_formation = kwargs.pop(
            "only_blue_formation", True
        )  # Only spawn blue agents in formation
        self.formation_agents_per_column = kwargs.pop("formation_agents_per_column", 2)
        self.randomise_formation_indices = kwargs.pop(
            "randomise_formation_indices", False
        )  # If False, each agent will always be in the same formation spot
        self.formation_noise = kwargs.pop(
            "formation_noise", 0.2
        )  # Noise on formation positions

        # Ai config
        self.n_traj_points = kwargs.pop(
            "n_traj_points", 0
        )  # Number of spline trajectory points to plot for heuristic (ai) agents
        self.ai_speed_strength = kwargs.pop(
            "ai_strength", 1.0
        )  # The speed of the ai 0<=x<=1
        self.ai_decision_strength = kwargs.pop(
            "ai_decision_strength", 1.0
        )  # The decision strength of the ai 0<=x<=1
        self.ai_precision_strength = kwargs.pop(
            "ai_precision_strength", 1.0
        )  # The precision strength of the ai 0<=x<=1
        self.disable_ai_red = kwargs.pop("disable_ai_red", False)

        # Task sizes
        self.agent_size = kwargs.pop("agent_size", 0.025)
        self.goal_size = kwargs.pop("goal_size", 0.35)
        self.goal_depth = kwargs.pop("goal_depth", 0.1)
        self.pitch_length = kwargs.pop("pitch_length", 3.0)
        self.pitch_width = kwargs.pop("pitch_width", 1.5)
        self.ball_mass = kwargs.pop("ball_mass", 0.25)
        self.ball_size = kwargs.pop("ball_size", 0.02)

        # Actions
        self.u_multiplier = kwargs.pop("u_multiplier", 0.1)

        # Actions shooting
        self.enable_shooting = kwargs.pop(
            "enable_shooting", False
        )  # Whether to enable an extra 2 actions (for rotation and shooting). Only avaioable for non-ai agents
        self.u_rot_multiplier = kwargs.pop("u_rot_multiplier", 0.0003)
        self.u_shoot_multiplier = kwargs.pop("u_shoot_multiplier", 0.6)
        self.shooting_radius = kwargs.pop("shooting_radius", 0.08)
        self.shooting_angle = kwargs.pop("shooting_angle", jnp.pi / 2)

        # Speeds
        self.max_speed = kwargs.pop("max_speed", 0.15)
        self.ball_max_speed = kwargs.pop("ball_max_speed", 0.3)

        # Rewards
        self.dense_reward = kwargs.pop("dense_reward", True)
        self.pos_shaping_factor_ball_goal = kwargs.pop(
            "pos_shaping_factor_ball_goal", 10.0
        )  # Reward for moving the ball towards the opponents' goal. This can be annealed in a curriculum.
        self.pos_shaping_factor_agent_ball = kwargs.pop(
            "pos_shaping_factor_agent_ball", 0.1
        )  # Reward for moving the closest agent to the ball in a team closer to it.
        # This is useful for exploration and can be annealed in a curriculum.
        # This reward does not trigger if the agent is less than distance_to_ball_trigger from the ball or the ball is moving
        self.distance_to_ball_trigger = kwargs.pop("distance_to_ball_trigger", 0.4)
        self.scoring_reward = kwargs.pop(
            "scoring_reward", 100.0
        )  # Discrete reward for scoring

        # Observations
        self.observe_teammates = kwargs.pop("observe_teammates", True)
        self.observe_adversaries = kwargs.pop("observe_adversaries", True)
        self.dict_obs = kwargs.pop("dict_obs", False)

        if kwargs.pop("dense_reward_ratio", None) is not None:
            raise ValueError(
                "dense_reward_ratio in football is deprecated, please use `dense_reward` "
                "which is a bool that turns on/off the dense reward"
            )
        ScenarioUtils.check_kwargs_consumed(kwargs)

    def make_world(self, batch_dim: int, **kwargs) -> World:
        self.init_params(**kwargs)

        self.visualize_semidims = False
        world = self.init_world(batch_dim)
        self.init_agents(world)
        self.init_ball(world)
        self.init_background()
        self.init_walls(world)
        self.init_goals(world)
        self.init_traj_pts(world)

        # Cached values
        self.left_goal_pos = torch.tensor(
            [-self.pitch_length / 2 - self.ball_size / 2, 0],
            device=device,
            dtype=torch.float,
        )
        self.right_goal_pos = -self.left_goal_pos
        self._done = torch.zeros(batch_dim, device=device, dtype=torch.bool)
        self._sparse_reward_blue = torch.zeros(
            batch_dim, device=device, dtype=torch.float32
        )
        self._sparse_reward_red = self._sparse_reward_blue.clone()
        self._render_field = True
        self.min_agent_dist_to_ball_blue = None
        self.min_agent_dist_to_ball_red = None

        self._reset_agent_range = torch.tensor(
            [self.pitch_length / 2, self.pitch_width],
            device=device,
        )
        self._reset_agent_offset_blue = torch.tensor(
            [-self.pitch_length / 2 + self.agent_size, -self.pitch_width / 2],
            device=device,
        )
        self._reset_agent_offset_red = torch.tensor(
            [-self.agent_size, -self.pitch_width / 2], device=device
        )
        self._agents_rel_pos_to_ball = None

        world = World(batch_dim=batch_dim, dim_p=2)

        # Add agents (alternating teams)
        for i in range(self.n_agents):
            agent = Agent(name=f"agent_{i}", dynamics=Holonomic())
            # Team 0 is blue, Team 1 is red
            agent.color = (
                jnp.array([0.35, 0.35, 0.85])
                if i < self.n_agents_per_team
                else jnp.array([0.85, 0.35, 0.35])
            )
            agent.collision_penalty = True
            agent.size = self.agent_size
            world.add_agent(agent)

        # Add ball (as entity)
        ball = Entity(name="ball")
        ball.collide = True
        ball.movable = True
        ball.mass = self.ball_mass
        ball.size = self.ball_size
        ball.color = jnp.array([1.0, 1.0, 1.0])  # White ball
        world.add_entity(ball)

        # Add goals (as landmarks)
        # Left goal (Team 0)
        goal_left_top = Entity(name="goal_left_top")
        goal_left_top.collide = True
        goal_left_top.movable = False
        goal_left_top.size = self.goal_thickness
        goal_left_top.color = jnp.array([0.35, 0.35, 0.85])
        goal_left_top.state.pos = jnp.array(
            [-self.field_length / 2, self.goal_width / 2]
        )
        world.add_landmark(goal_left_top)

        goal_left_bottom = Entity(name="goal_left_bottom")
        goal_left_bottom.collide = True
        goal_left_bottom.movable = False
        goal_left_bottom.size = self.goal_thickness
        goal_left_bottom.color = jnp.array([0.35, 0.35, 0.85])
        goal_left_bottom.state.pos = jnp.array(
            [-self.field_length / 2, -self.goal_width / 2]
        )
        world.add_landmark(goal_left_bottom)

        # Right goal (Team 1)
        goal_right_top = Entity(name="goal_right_top")
        goal_right_top.collide = True
        goal_right_top.movable = False
        goal_right_top.size = self.goal_thickness
        goal_right_top.color = jnp.array([0.85, 0.35, 0.35])
        goal_right_top.state.pos = jnp.array(
            [self.field_length / 2, self.goal_width / 2]
        )
        world.add_landmark(goal_right_top)

        goal_right_bottom = Entity(name="goal_right_bottom")
        goal_right_bottom.collide = True
        goal_right_bottom.movable = False
        goal_right_bottom.size = self.goal_thickness
        goal_right_bottom.color = jnp.array([0.85, 0.35, 0.35])
        goal_right_bottom.state.pos = jnp.array(
            [self.field_length / 2, -self.goal_width / 2]
        )
        world.add_landmark(goal_right_bottom)

        # Initialize scenario state
        self.state = FootballState(
            ball_possession=jnp.zeros((batch_dim, 2)),
            agent_ball_distances=jnp.zeros((batch_dim, self.n_agents)),
            team_scores=jnp.zeros((batch_dim, 2)),
            ball_touched=jnp.zeros(batch_dim),
        )

        return world

    def init_world(self, batch_dim: int):
        # Make world
        world = World(
            batch_dim,
            dt=0.1,
            drag=0.05,
            x_semidim=self.pitch_length / 2 + self.goal_depth - self.agent_size,
            y_semidim=self.pitch_width / 2 - self.agent_size,
            substeps=2,
        )
        world.agent_size = self.agent_size
        world.pitch_width = self.pitch_width
        world.pitch_length = self.pitch_length
        world.goal_size = self.goal_size
        world.goal_depth = self.goal_depth
        return world

    def reset_world_at(self, env_index: int | None):
        batch_size = self.world.batch_dim if env_index is None else 1

        # Place ball at center
        ball = self.world.entities[0]
        ball.state.pos = jnp.zeros_like(ball.state.pos)
        ball.state.vel = jnp.zeros_like(ball.state.vel)

        # Place agents in team formations
        for i, agent in enumerate(self.world.agents):
            team = i // self.n_agents_per_team  # 0 or 1
            position_in_team = i % self.n_agents_per_team

            # Team 0 starts on left, Team 1 on right
            x_base = -0.5 if team == 0 else 0.5
            y_spread = 0.3  # Vertical spread of team formation

            x_pos = x_base * self.field_length
            y_pos = (position_in_team - (self.n_agents_per_team - 1) / 2) * y_spread

            agent.state.pos = jnp.where(
                env_index is None,
                jnp.tile(jnp.array([x_pos, y_pos]), (batch_size, 1)),
                jnp.array([x_pos, y_pos]),
            )
            agent.state.vel = jnp.zeros_like(agent.state.vel)

        # Reset state
        self.state = FootballState(
            ball_possession=jnp.zeros((batch_size, 2)),
            agent_ball_distances=jnp.zeros((batch_size, self.n_agents)),
            team_scores=jnp.zeros((batch_size, 2)),
            ball_touched=jnp.zeros(batch_size),
        )

    def reward(self, agent: Agent) -> Float[Array, f"{batch}"]:
        agent_idx = self.world.agents.index(agent)
        team_idx = agent_idx // self.n_agents_per_team
        reward = jnp.zeros(self.world.batch_dim)

        # Get ball
        ball = self.world.entities[0]

        # Calculate distance to ball
        dist_to_ball = jnp.linalg.norm(agent.state.pos - ball.state.pos, axis=-1)
        self.state = self.state.replace(
            agent_ball_distances=self.state.agent_ball_distances.at[:, agent_idx].set(
                dist_to_ball
            )
        )

        # Ball possession reward
        touching_ball = dist_to_ball < self.ball_touch_distance
        reward = jnp.where(touching_ball, reward + 0.1, reward)

        # Update ball touched status
        self.state = self.state.replace(
            ball_touched=jnp.logical_or(touching_ball, self.state.ball_touched)
        )

        # Update team possession
        self.state = self.state.replace(
            ball_possession=self.state.ball_possession.at[:, team_idx].set(
                jnp.where(touching_ball, 1.0, self.state.ball_possession[:, team_idx])
            )
        )

        # Check for goals
        ball_x = ball.state.pos[:, 0]
        ball_y = ball.state.pos[:, 1]

        # Goal for team 1 (right goal)
        goal_right = (ball_x > self.field_length / 2) & (
            jnp.abs(ball_y) < self.goal_width / 2
        )
        # Goal for team 0 (left goal)
        goal_left = (ball_x < -self.field_length / 2) & (
            jnp.abs(ball_y) < self.goal_width / 2
        )

        # Update scores and give rewards
        if team_idx == 0:  # Team 0 (left goal is opponent's)
            reward = jnp.where(
                goal_left, reward + self.goal_reward, reward
            )  # Reward for scoring
            reward = jnp.where(
                goal_right, reward - self.goal_reward / 2, reward
            )  # Penalty for conceding

            self.state = self.state.replace(
                team_scores=self.state.team_scores.at[:, 0].set(
                    jnp.where(
                        goal_left,
                        self.state.team_scores[:, 0] + 1,
                        self.state.team_scores[:, 0],
                    )
                )
            )
        else:  # Team 1 (right goal is opponent's)
            reward = jnp.where(
                goal_right, reward + self.goal_reward, reward
            )  # Reward for scoring
            reward = jnp.where(
                goal_left, reward - self.goal_reward / 2, reward
            )  # Penalty for conceding

            self.state = self.state.replace(
                team_scores=self.state.team_scores.at[:, 1].set(
                    jnp.where(
                        goal_right,
                        self.state.team_scores[:, 1] + 1,
                        self.state.team_scores[:, 1],
                    )
                )
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
        team_idx = agent_idx // self.n_agents_per_team

        # Get ball state relative to agent
        ball = self.world.entities[0]
        ball_pos = ball.state.pos - agent.state.pos
        ball_vel = ball.state.vel

        # Get goal positions relative to agent
        goals = []
        for goal in self.world.landmarks:
            goals.append(goal.state.pos - agent.state.pos)

        # Get positions and velocities of other agents relative to this agent
        teammates_pos = []
        teammates_vel = []
        opponents_pos = []
        opponents_vel = []

        for other_idx, other in enumerate(self.world.agents):
            if other is agent:
                continue
            other_team = other_idx // self.n_agents_per_team
            if other_team == team_idx:
                teammates_pos.append(other.state.pos - agent.state.pos)
                teammates_vel.append(other.state.vel)
            else:
                opponents_pos.append(other.state.pos - agent.state.pos)
                opponents_vel.append(other.state.vel)

        # Stack observations
        obs = jnp.concatenate(
            [agent.state.pos]  # Own position
            + [agent.state.vel]  # Own velocity
            + [ball_pos]  # Ball position
            + [ball_vel]  # Ball velocity
            + goals  # Goal positions
            + teammates_pos  # Teammate positions
            + teammates_vel  # Teammate velocities
            + opponents_pos  # Opponent positions
            + opponents_vel  # Opponent velocities
            + [self.state.team_scores],  # Current score
            axis=-1,
        )
        return obs

    def done(self) -> Float[Array, f"{batch}"]:
        # Episode can end based on max score or time limit
        # Here we'll just use a simple max score condition
        max_score = 5
        return jnp.any(self.state.team_scores >= max_score, axis=-1)
