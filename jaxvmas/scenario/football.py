#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from enum import Enum

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Int, PRNGKeyArray, jaxtyped

from jaxvmas.equinox_utils import PyTreeNode, dataclass_to_dict_first_layer
from jaxvmas.interactive_rendering import render_interactively
from jaxvmas.simulator.core import (
    Agent,
    Entity,
    Landmark,
    World,
)
from jaxvmas.simulator.core.shapes import Box, Line, Sphere
from jaxvmas.simulator.dynamics.holonomic import Holonomic
from jaxvmas.simulator.dynamics.holonomic_with_rot import HolonomicWithRotation
from jaxvmas.simulator.rendering import Geom
from jaxvmas.simulator.scenario import BaseScenario
from jaxvmas.simulator.utils import Color, JaxUtils, ScenarioUtils, X, Y

env_index_dim = "env_index"


@jaxtyped(typechecker=beartype)
class Splines(PyTreeNode):
    A: Array = eqx.field(
        default_factory=lambda: jnp.asarray(
            [
                [2.0, -2.0, 1.0, 1.0],
                [-3.0, 3.0, -2.0, -1.0],
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
        )
    )
    U_matmul_A: dict[tuple[int, float], Array] = eqx.field(default_factory=dict)

    @jaxtyped(typechecker=beartype)
    def hermite(
        self, p0: Array, p1: Array, p0dot: Array, p1dot: Array, u: float, deriv: int
    ):
        # A trajectory specified by the initial pos p0, initial vel p0dot, end pos p1,
        #     and end vel p1dot.
        # Evaluated at the given value of u, which is between 0 and 1 (0 being the start
        #     of the trajectory, and 1 being the end). This yields a position.
        # When called with deriv=n, we instead return the nth time derivative of the trajectory.
        #     For example, deriv=1 will give the velocity evaluated at time u.
        # assert isinstance(u, float)
        U_matmul_A = None
        if U_matmul_A is None:
            u_jax_array = jnp.asarray([u])
            U = jnp.stack(
                [
                    self.nPr(3, deriv) * (u_jax_array ** max(0, 3 - deriv)),
                    self.nPr(2, deriv) * (u_jax_array ** max(0, 2 - deriv)),
                    self.nPr(1, deriv) * (u_jax_array ** max(0, 1 - deriv)),
                    self.nPr(0, deriv) * (u_jax_array**0),
                ],
                axis=1,
            )
            U_matmul_A = U[:, None, :] @ self.A[None, :, :]
        P = jnp.stack([p0, p1, p0dot, p1dot], axis=1)

        ans = (
            jnp.broadcast_to(U_matmul_A, (P.shape[0], 1, 4)) @ P
        )  # Matmul [batch x 1 x 4] @ [batch x 4 x 2] -> [batch x 1 x 2]
        ans = ans.squeeze(1)
        return self, ans

    @jaxtyped(typechecker=beartype)
    def nPr(self, n: int, r: int) -> int:
        # calculates n! / (n-r)!
        if r > n:
            return 0
        ans = 1
        for k in range(n, max(1, n - r), -1):
            ans = ans * k
        return ans


# Agent Policy
@jaxtyped(typechecker=beartype)
class AgentPolicy(PyTreeNode):
    team_name: str
    otherteam_name: str
    speed_strength: float
    decision_strength: float
    precision_strength: float
    strength_multiplier: float
    pos_lookahead: float
    vel_lookahead: float
    possession_lookahead: float
    dribble_speed: float
    shooting_radius: float
    shooting_angle: float
    take_shot_angle: float
    max_shot_dist: float
    nsamples: int
    sigma: float
    replan_margin: float
    initialised: bool
    disabled: bool
    team_color: Enum | None
    enable_shooting: bool
    objectives: dict[str, dict[str, Array]]
    agent_possession: dict[int, Array]
    team_possession: Array | None
    team_disps: dict[str, Array]
    splines: Splines

    @classmethod
    @jaxtyped(typechecker=beartype)
    def create(
        cls,
        team: str,
        speed_strength=1.0,
        decision_strength=1.0,
        precision_strength=1.0,
        disabled: bool = False,
    ) -> "AgentPolicy":
        team_name = team
        otherteam_name = "Blue" if (team_name == "Red") else "Red"

        # affects the speed of the agents
        speed_strength = speed_strength**2

        # affects off-the-ball movement
        # (who is assigned to the ball and the positioning of the non-dribbling agents)
        # so with poor decision strength they might decide that an agent that is actually in a worse position should go for the ball
        decision_strength = decision_strength

        # affects the ability to execute planned manoeuvres,
        # it will add some error to the target position and velocity
        precision_strength = precision_strength

        strength_multiplier = 25.0

        pos_lookahead = 0.01

        vel_lookahead = 0.01

        possession_lookahead = 0.5

        dribble_speed = 0.16 + 0.16 * speed_strength

        shooting_radius = 0.08

        shooting_angle = jnp.pi / 2

        take_shot_angle = jnp.pi / 4

        max_shot_dist = 0.5

        nsamples = 2

        sigma = 0.5

        replan_margin = 0.0

        initialised = False

        disabled = disabled

        team_color = None

        enable_shooting = False

        objectives = {}

        agent_possession = {}

        team_possession = None

        team_disps = {}

        splines = Splines()

        return cls(
            team_name=team_name,
            otherteam_name=otherteam_name,
            speed_strength=speed_strength,
            decision_strength=decision_strength,
            precision_strength=precision_strength,
            strength_multiplier=strength_multiplier,
            pos_lookahead=pos_lookahead,
            vel_lookahead=vel_lookahead,
            possession_lookahead=possession_lookahead,
            dribble_speed=dribble_speed,
            shooting_radius=shooting_radius,
            shooting_angle=shooting_angle,
            take_shot_angle=take_shot_angle,
            max_shot_dist=max_shot_dist,
            nsamples=nsamples,
            sigma=sigma,
            replan_margin=replan_margin,
            initialised=initialised,
            disabled=disabled,
            team_color=team_color,
            enable_shooting=enable_shooting,
            objectives=objectives,
            agent_possession=agent_possession,
            team_possession=team_possession,
            team_disps=team_disps,
            splines=splines,
        )

    @jaxtyped(typechecker=beartype)
    def get_dynamic_params(self, world: "FootballWorld"):
        ball = world.ball
        if self.team_name == "Red":
            teammates = world.red_agents
            opposition = world.blue_agents
            own_net = world.red_net
            target_net = world.blue_net
        elif self.team_name == "Blue":
            teammates = world.blue_agents
            opposition = world.red_agents
            own_net = world.blue_net
            target_net = world.red_net

        return ball, teammates, opposition, own_net, target_net

    @jaxtyped(typechecker=beartype)
    def init(self, world: "FootballWorld") -> "AgentPolicy":
        initialised = True

        ball, teammates, opposition, own_net, target_net = self.get_dynamic_params(
            world
        )

        team_color = teammates[0].color if len(teammates) > 0 else None
        enable_shooting = teammates[0].action_size == 4 if len(teammates) > 0 else False

        objectives = {
            agent.name: {
                "shot_power": jnp.zeros((world.batch_dim,)),
                "target_ang": jnp.zeros((world.batch_dim,)),
                "target_pos_rel": jnp.zeros((world.batch_dim, world.dim_p)),
                "target_pos": jnp.zeros((world.batch_dim, world.dim_p)),
                "target_vel": jnp.zeros((world.batch_dim, world.dim_p)),
                "start_pos": jnp.zeros((world.batch_dim, world.dim_p)),
                "start_vel": jnp.zeros((world.batch_dim, world.dim_p)),
            }
            for agent in teammates
        }

        agent_possession = {
            agent.id: jnp.zeros(world.batch_dim, dtype=jnp.bool_) for agent in teammates
        }

        team_possession = jnp.zeros(world.batch_dim, dtype=jnp.bool_)

        team_disps = {}

        return self.replace(
            initialised=initialised,
            team_color=team_color,
            enable_shooting=enable_shooting,
            objectives=objectives,
            agent_possession=agent_possession,
            team_possession=team_possession,
            team_disps=team_disps,
        )

    @jaxtyped(typechecker=beartype)
    def reset(self, world: "FootballWorld", env_index=Ellipsis) -> "AgentPolicy":
        team_disps = {}
        objectives = dict(self.objectives)
        ball, teammates, opposition, own_net, target_net = self.get_dynamic_params(
            world
        )
        for agent in teammates:
            objectives[agent.name]["shot_power"] = (
                objectives[agent.name]["shot_power"].at[env_index].set(0)
            )
            objectives[agent.name]["target_ang"] = (
                objectives[agent.name]["target_ang"].at[env_index].set(0)
            )
            objectives[agent.name]["target_pos_rel"] = (
                objectives[agent.name]["target_pos_rel"]
                .at[env_index]
                .set(jnp.zeros(world.dim_p))
            )
            objectives[agent.name]["target_pos"] = (
                objectives[agent.name]["target_pos"]
                .at[env_index]
                .set(jnp.zeros(world.dim_p))
            )
            objectives[agent.name]["target_vel"] = (
                objectives[agent.name]["target_vel"]
                .at[env_index]
                .set(jnp.zeros(world.dim_p))
            )
            objectives[agent.name]["start_pos"] = (
                objectives[agent.name]["start_pos"]
                .at[env_index]
                .set(jnp.zeros(world.dim_p))
            )
            objectives[agent.name]["start_vel"] = (
                objectives[agent.name]["start_vel"]
                .at[env_index]
                .set(jnp.zeros(world.dim_p))
            )
        return self.replace(objectives=objectives, team_disps=team_disps)

    @jaxtyped(typechecker=beartype)
    def dribble_policy(
        self,
        PRNG_key: PRNGKeyArray,
        agent: "FootballAgent",
        world: "FootballWorld",
    ) -> tuple["AgentPolicy", "FootballWorld"]:
        possession_mask = self.agent_possession[agent.id]
        PRNG_key, subkey = jax.random.split(PRNG_key)
        self, world = self.dribble_to_goal(
            subkey, agent, world, env_index=possession_mask
        )
        move_mask = ~possession_mask
        PRNG_key, subkey = jax.random.split(PRNG_key)
        self, best_pos = self.check_better_positions(
            subkey, agent, world, env_index=move_mask
        )
        PRNG_key, subkey = jax.random.split(PRNG_key)
        self, world = self.go_to(
            subkey,
            agent,
            world,
            pos=best_pos,
            aggression=1.0,
            env_index=move_mask,
        )
        return self, world

    @jaxtyped(typechecker=beartype)
    def passing_policy(
        self,
        PRNG_key: PRNGKeyArray,
        agent: "FootballAgent",
        world: "FootballWorld",
    ) -> tuple["AgentPolicy", "FootballWorld"]:
        ball, teammates, opposition, own_net, target_net = self.get_dynamic_params(
            world
        )
        possession_mask = self.agent_possession[agent.id]
        otheragent = None
        for a in teammates:
            if a != agent:
                otheragent = a
                break
        # min_dist_mask = (agent.state.pos - otheragent.state.pos).norm(dim=-1) > self.max_shot_dist * 0.75
        PRNG_key, subkey = jax.random.split(PRNG_key)
        self, world = self.shoot(
            subkey, agent, world, otheragent.state.pos, env_index=possession_mask
        )
        move_mask = ~possession_mask
        PRNG_key, subkey = jax.random.split(PRNG_key)
        self, best_pos = self.check_better_positions(
            subkey, agent, world, env_index=move_mask
        )
        PRNG_key, subkey = jax.random.split(PRNG_key)
        self, world = self.go_to(
            subkey,
            agent,
            world,
            pos=best_pos,
            aggression=1.0,
            env_index=move_mask,
        )
        return self, world

    @jaxtyped(typechecker=beartype)
    def disable(self):
        return self.replace(disabled=True)

    @jaxtyped(typechecker=beartype)
    def enable(self):
        return self.replace(disabled=False)

    @eqx.filter_jit
    @jaxtyped(typechecker=beartype)
    def run(
        self,
        PRNG_key: PRNGKeyArray,
        agent: "FootballAgent",
        world: "FootballWorld",
    ) -> tuple["AgentPolicy", "FootballWorld"]:
        print("Compiling for agent", agent.name)
        if not self.disabled:
            if "0" in agent.name:
                self = self.replace(team_disps={})
                PRNG_key, subkey = jax.random.split(PRNG_key)
                self = self.check_possession(subkey, world)
            PRNG_key, subkey = jax.random.split(PRNG_key)
            self, world = self.dribble_policy(subkey, agent, world)
            self, control = self.get_action(agent)
            control = jnp.clip(control, min=-agent.u_range, max=agent.u_range)
            expanded_multiplier = jnp.broadcast_to(
                agent.action.u_multiplier_jax_array[None], control.shape
            )
            u = control * expanded_multiplier
            agent = agent.replace(action=agent.action.replace(u=u))
        else:
            agent = agent.replace(
                action=agent.action.replace(
                    u=jnp.zeros((self.world.batch_dim, agent.action_size))
                )
            )
        world = world.replace(
            red_controller=self if self.team_name == "Red" else world.red_controller,
            blue_controller=self if self.team_name == "Blue" else world.blue_controller,
        )

        return agent, world

    @jaxtyped(typechecker=beartype)
    def dribble_to_goal(
        self,
        PRNG_key: PRNGKeyArray,
        agent: "FootballAgent",
        world: "FootballWorld",
        env_index=Ellipsis,
    ) -> tuple["AgentPolicy", "FootballWorld"]:
        ball, teammates, opposition, own_net, target_net = self.get_dynamic_params(
            world
        )

        # Create target position based on env_index
        target_pos = target_net.state.pos
        if isinstance(env_index, Array) and env_index.dtype == jnp.bool_:
            # Use boolean masking for indexed environments
            target_pos = jnp.where(env_index[:, None], target_pos, target_pos)

        PRNG_key, subkey = jax.random.split(PRNG_key)
        self, world = self.dribble(
            subkey,
            agent,
            world,
            target_pos,
            env_index=env_index,
        )
        return self, world

    @jaxtyped(typechecker=beartype)
    def dribble(
        self,
        PRNG_key: PRNGKeyArray,
        agent: "FootballAgent",
        world: "FootballWorld",
        pos: Array,
        env_index=Ellipsis,
    ) -> tuple["AgentPolicy", "FootballWorld"]:
        PRNG_key, subkey = jax.random.split(PRNG_key)
        self, world = self.update_dribble(
            subkey,
            agent,
            world,
            pos=pos,
            env_index=env_index,
        )
        return self, world

    @jaxtyped(typechecker=beartype)
    def update_dribble(
        self,
        PRNG_key: PRNGKeyArray,
        agent: "FootballAgent",
        world: "FootballWorld",
        pos: Array,
        env_index=Ellipsis,
    ) -> tuple["AgentPolicy", "FootballWorld"]:
        # Specifies a new location to dribble towards.

        agent_pos = agent.state.pos
        ball_pos = world.ball.state.pos
        if isinstance(env_index, Array) and env_index.dtype == jnp.bool_:
            # Use boolean masking for indexed environments
            agent_pos = jnp.where(env_index[:, None], agent_pos, agent_pos)
            ball_pos = jnp.where(env_index[:, None], ball_pos, ball_pos)

        ball_disp = pos - ball_pos
        ball_dist = jnp.linalg.norm(ball_disp, axis=-1)
        direction = ball_disp / ball_dist[:, None]
        hit_vel = direction * self.dribble_speed
        start_vel = self.get_start_vel(ball_pos, hit_vel, agent_pos, aggression=0.0)
        start_vel_mag = jnp.linalg.norm(start_vel, axis=-1)
        # Calculate hit_pos, the adjusted position to strike the ball so it goes where we want
        offset = start_vel.copy()
        start_vel_mag_mask = start_vel_mag > 0
        offset = jnp.where(
            start_vel_mag_mask[:, None],  # broadcast mask to match dimensions
            offset
            / jnp.where(start_vel_mag_mask[:, None], start_vel_mag[:, None], 1.0),
            offset,
        )
        new_direction = direction + 0.5 * offset
        new_direction /= jnp.linalg.norm(new_direction, axis=-1)[:, None]
        hit_pos = (
            ball_pos
            - new_direction * (world.ball.shape.radius + agent.shape.radius) * 0.7
        )
        PRNG_key, subkey = jax.random.split(PRNG_key)
        # Execute dribble with a go_to command
        self, world = self.go_to(
            subkey,
            agent,
            world,
            hit_pos,
            hit_vel,
            start_vel=start_vel,
            env_index=env_index,
        )
        return self, world

    @jaxtyped(typechecker=beartype)
    def shoot(
        self,
        PRNG_key: PRNGKeyArray,
        agent: "FootballAgent",
        world: "FootballWorld",
        pos: Array,
        env_index=Ellipsis,
    ) -> tuple["AgentPolicy", "FootballWorld"]:
        agent_pos = agent.state.pos
        ball_disp = world.ball.state.pos - agent_pos
        ball_dist = jnp.linalg.norm(ball_disp, axis=-1)
        within_range_mask = ball_dist <= self.shooting_radius
        target_disp = pos - agent_pos
        target_dist = jnp.linalg.norm(target_disp, axis=-1)
        ball_rel_angle = self.get_rel_ang(ang1=agent.state.rot, vec2=ball_disp)
        target_rel_angle = self.get_rel_ang(ang1=agent.state.rot, vec2=target_disp)
        ball_within_angle_mask = jnp.abs(ball_rel_angle) < self.shooting_angle / 2
        rot_within_angle_mask = jnp.abs(target_rel_angle) < self.take_shot_angle / 2
        shooting_mask = (
            within_range_mask & ball_within_angle_mask & rot_within_angle_mask
        )
        objectives = dict(self.objectives)
        # Pre-shooting
        target_ang = jnp.atan2(target_disp[:, 1], target_disp[:, 0])
        if isinstance(env_index, Array) and env_index.dtype == jnp.bool_:
            # Use boolean masking for indexed environments
            objectives[agent.name]["target_ang"] = jnp.where(
                env_index, target_ang, objectives[agent.name]["target_ang"]
            )
        else:
            objectives[agent.name]["target_ang"] = target_ang
        PRNG_key, subkey = jax.random.split(PRNG_key)
        self, world = self.dribble(subkey, agent, world, pos, env_index=env_index)
        # Shooting
        objectives[agent.name]["shot_power"][:] = -1
        objectives[agent.name]["shot_power"][
            self.combine_mask(shooting_mask, env_index)
        ] = jnp.minimum(target_dist[shooting_mask] / self.max_shot_dist, 1.0)
        return self.replace(objectives=objectives), world

    @jaxtyped(typechecker=beartype)
    def combine_mask(self, mask, env_index):
        if env_index == Ellipsis:
            return mask
        elif (
            env_index.shape[0] == self.world.batch_dim and env_index.dtype == jnp.bool_
        ):
            return mask & env_index
        raise ValueError("Expected env_index to be : or boolean tensor")

    @jaxtyped(typechecker=beartype)
    def go_to(
        self,
        PRNG_key: PRNGKeyArray,
        agent: "FootballAgent",
        world: "FootballWorld",
        pos: Array,
        vel: Array | None = None,
        start_vel: Array | None = None,
        aggression: float = 1.0,
        env_index=Ellipsis,
    ):
        start_pos = agent.state.pos
        if isinstance(env_index, Array) and env_index.dtype == jnp.bool_:
            # Use boolean masking for indexed environments
            start_pos = jnp.where(env_index[:, None], start_pos, start_pos)

        if vel is None:
            vel = jnp.zeros_like(pos)
        if start_vel is None:
            aggression = (jnp.linalg.norm(pos - start_pos, axis=-1) > 0.1) * aggression
            start_vel = self.get_start_vel(pos, vel, start_pos, aggression=aggression)

        target_pos = self.objectives[agent.name]["target_pos"]
        if isinstance(env_index, Array) and env_index.dtype == jnp.bool_:
            # Use boolean masking for indexed environments
            target_pos = jnp.where(env_index[:, None], target_pos, target_pos)
        diff = jnp.linalg.norm(target_pos - pos, axis=-1)[..., None]

        if self.precision_strength != 1:
            exp_diff = jnp.exp(-diff)
            PRNG_key, subkey = jax.random.split(PRNG_key)
            pos += (
                jax.random.normal(subkey, pos.shape)
                * 10
                * (1 - self.precision_strength)
                * (1 - exp_diff)
            )
            PRNG_key, subkey = jax.random.split(PRNG_key)
            vel += (
                jax.random.normal(subkey, pos.shape)
                * 10
                * (1 - self.precision_strength)
                * (1 - exp_diff)
            )
        objectives = dict(self.objectives)
        ball_pos = world.ball.state.pos
        if isinstance(env_index, Array) and env_index.dtype == jnp.bool_:
            # Use boolean masking for indexed environments
            ball_pos = jnp.where(env_index[:, None], ball_pos, ball_pos)

            objectives[agent.name]["target_pos_rel"] = jnp.where(
                env_index[:, None],
                pos - ball_pos,
                objectives[agent.name]["target_pos_rel"],
            )

            objectives[agent.name]["target_pos"] = jnp.where(
                env_index[:, None], pos, objectives[agent.name]["target_pos"]
            )

            objectives[agent.name]["target_vel"] = jnp.where(
                env_index[:, None], vel, objectives[agent.name]["target_vel"]
            )

            objectives[agent.name]["start_pos"] = jnp.where(
                env_index[:, None], start_pos, objectives[agent.name]["start_pos"]
            )

            objectives[agent.name]["start_vel"] = jnp.where(
                env_index[:, None], start_vel, objectives[agent.name]["start_vel"]
            )
        else:
            objectives[agent.name]["target_pos_rel"] = pos - ball_pos
            objectives[agent.name]["target_pos"] = pos
            objectives[agent.name]["target_vel"] = vel
            objectives[agent.name]["start_pos"] = start_pos
            objectives[agent.name]["start_vel"] = start_vel
        self = self.replace(objectives=objectives)
        self, world = self.plot_traj(agent, world, env_index=env_index)
        return self, world

    @jaxtyped(typechecker=beartype)
    def get_start_vel(
        self, pos: Array, vel: Array, start_pos: Array, aggression: float = 0.0
    ):
        # Calculates the starting velocity for a planned trajectory ending at position pos at velocity vel
        # The initial velocity is not directly towards the goal because we want a curved path
        #     that reaches the goal at the moment it achieves a given velocity.
        # Since we replan trajectories a lot, the magnitude of the initial velocity highly influences the
        #     overall speed. To modulate this, we introduce an aggression parameter.
        # aggression=0 will set the magnitude of the initial velocity to the current velocity, while
        #     aggression=1 will set the magnitude of the initial velocity to 1.0.
        vel_mag = 1.0 * aggression + jnp.linalg.norm(vel, axis=-1) * (1 - aggression)
        goal_disp = pos - start_pos
        goal_dist = jnp.linalg.norm(goal_disp, axis=-1)
        vel_dir = vel.copy()
        vel_mag_great_0 = vel_mag > 0
        vel_dir = jnp.where(
            vel_mag_great_0[:, None],  # broadcast mask to match dimensions
            vel_dir / jnp.where(vel_mag_great_0[:, None], vel_mag[:, None], 1.0),
            vel_dir,
        )
        dist_behind_target = 0.6 * goal_dist
        target_pos = pos - vel_dir * dist_behind_target[:, None]
        target_disp = target_pos - start_pos
        target_dist = jnp.linalg.norm(target_disp, axis=-1)
        start_vel_aug_dir = target_disp
        target_dist_great_0 = target_dist > 0
        start_vel_aug_dir = jnp.where(
            target_dist_great_0[:, None],  # broadcast mask to match dimensions
            start_vel_aug_dir
            / jnp.where(target_dist_great_0[:, None], target_dist[:, None], 1.0),
            start_vel_aug_dir,
        )
        start_vel = start_vel_aug_dir * vel_mag[:, None]
        return start_vel

    @jaxtyped(typechecker=beartype)
    def get_action(self, agent: "FootballAgent", env_index=Ellipsis):
        # Gets the action computed by the policy for the given agent.
        # All the logic in AgentPolicy (dribbling, moving, shooting, etc) uses the go_to command
        #     as an interface to specify a desired trajectory.
        # After AgentPolicy has computed its desired trajectories, get_action looks up the parameters
        #     specifying those trajectories, and computes an action from them using splines.
        # To compute the action, we generate a hermite spline and take the first position and velocity
        #     along that trajectory (or, to be more precise, we look in the future by pos_lookahead
        #     and vel_lookahead. The velocity is simply the first derivative of the position spline.
        # Given these open-loop position and velocity controls, we use the error in the position and
        #     velocity to compute the closed-loop control.
        # The strength modifier (between 0 and 1) times some multiplier modulates the magnitude of the
        #     resulting action, controlling the speed.
        curr_pos = agent.state.pos
        curr_vel = agent.state.vel
        start_pos = self.objectives[agent.name]["start_pos"]
        target_pos = self.objectives[agent.name]["target_pos"]
        start_vel = self.objectives[agent.name]["start_vel"]
        target_vel = self.objectives[agent.name]["target_vel"]

        if isinstance(env_index, Array) and env_index.dtype == jnp.bool_:
            # Use boolean masking for indexed environments
            curr_pos = jnp.where(env_index[:, None], curr_pos, curr_pos)
            curr_vel = jnp.where(env_index[:, None], curr_vel, curr_vel)
            start_pos = jnp.where(env_index[:, None], start_pos, start_pos)
            target_pos = jnp.where(env_index[:, None], target_pos, target_pos)
            start_vel = jnp.where(env_index[:, None], start_vel, start_vel)
            target_vel = jnp.where(env_index[:, None], target_vel, target_vel)

        splines, des_curr_pos = self.splines.hermite(
            start_pos,
            target_pos,
            start_vel,
            target_vel,
            u=min(self.pos_lookahead, 1),
            deriv=0,
        )
        self = self.replace(splines=splines)

        splines, des_curr_vel = self.splines.hermite(
            start_pos,
            target_pos,
            start_vel,
            target_vel,
            u=min(self.vel_lookahead, 1),
            deriv=1,
        )
        self = self.replace(splines=splines)

        des_curr_pos = jnp.asarray(des_curr_pos)
        des_curr_vel = jnp.asarray(des_curr_vel)
        movement_control = 0.5 * (des_curr_pos - curr_pos) + 0.5 * (
            des_curr_vel - curr_vel
        )
        movement_control *= self.speed_strength * self.strength_multiplier
        if agent.action_size == 2:
            return self, movement_control
        shooting_control = jnp.zeros_like(movement_control)
        shooting_control[:, 1] = self.objectives[agent.name]["shot_power"]
        rel_ang = self.get_rel_ang(
            ang1=self.objectives[agent.name]["target_ang"], ang2=agent.state.rot
        ).squeeze(-1)
        shooting_control[:, 0] = jnp.sin(rel_ang)
        shooting_control[rel_ang > jnp.pi / 2, 0] = 1
        shooting_control[rel_ang < -jnp.pi / 2, 0] = -1
        control = jnp.concatenate([movement_control, shooting_control], axis=-1)
        return self, control

    @jaxtyped(typechecker=beartype)
    def get_rel_ang(
        self,
        vec1: Array | None = None,
        vec2: Array | None = None,
        ang1: Array | None = None,
        ang2: Array | None = None,
    ):
        if vec1 is not None:
            ang1 = jnp.atan2(vec1[:, 1], vec1[:, 0])
        if vec2 is not None:
            ang2 = jnp.atan2(vec2[:, 1], vec2[:, 0])
        if ang1.ndim == 2:
            ang1 = ang1.squeeze(-1)
        if ang2.dim() == 2:
            ang2 = ang2.squeeze(-1)
        return (ang1 - ang2 + jnp.pi) % (2 * jnp.pi) - jnp.pi

    @jaxtyped(typechecker=beartype)
    def plot_traj(
        self, agent: "FootballAgent", world: "FootballWorld", env_index=0
    ) -> tuple["AgentPolicy", "FootballWorld"]:
        for i, u in enumerate(
            jnp.linspace(
                0,
                1,
                len(world.traj_points[self.team_name][agent.id]),
            )
        ):
            pointi_index = world.traj_points[self.team_name][agent.id][i]
            pointi = world.landmarks[pointi_index]
            start_pos = self.objectives[agent.name]["start_pos"]
            target_pos = self.objectives[agent.name]["target_pos"]
            start_vel = self.objectives[agent.name]["start_vel"]
            target_vel = self.objectives[agent.name]["target_vel"]

            if isinstance(env_index, Array) and env_index.dtype == jnp.bool_:
                # Use boolean masking for indexed environments
                start_pos = jnp.where(env_index[:, None], start_pos, start_pos)
                target_pos = jnp.where(env_index[:, None], target_pos, target_pos)
                start_vel = jnp.where(env_index[:, None], start_vel, start_vel)
                target_vel = jnp.where(env_index[:, None], target_vel, target_vel)

            splines, posi = self.splines.hermite(
                start_pos,
                target_pos,
                start_vel,
                target_vel,
                u=u,
                deriv=0,
            )
            self = self.replace(splines=splines)
            # Simplify position setting with boolean masking
            if isinstance(env_index, Array) and env_index.dtype == jnp.bool_:
                pointi = pointi.set_pos(
                    jnp.asarray(posi),
                    batch_index=None,
                )
            elif isinstance(env_index, int):
                pointi = pointi.set_pos(
                    jnp.asarray(posi),
                    batch_index=env_index,
                )
            elif isinstance(env_index, list):
                for envi in env_index:
                    pointi = pointi.set_pos(
                        jnp.asarray(posi)[envi, :],
                        batch_index=env_index[envi],
                    )
            elif (
                isinstance(env_index, Array)
                and env_index.dtype == jnp.bool_
                and jnp.any(env_index)
            ):
                envs = jnp.where(env_index)
                for i, envi in enumerate(envs):
                    pointi = pointi.set_pos(
                        jnp.asarray(posi)[i, :],
                        batch_index=envi[0],
                    )
            all_landmarks = world.landmarks
            all_landmarks = (
                all_landmarks[:pointi_index]
                + [pointi]
                + all_landmarks[pointi_index + 1 :]
            )
            world = world.replace(landmarks=all_landmarks)
        return self, world

    @jaxtyped(typechecker=beartype)
    def clamp_pos(
        self, world: "FootballWorld", pos: Array, return_bool: bool = False
    ) -> Array:
        orig_pos = pos.copy()

        agent_size = world.agent_size
        pitch_y = world.pitch_width / 2 - agent_size
        pitch_x = world.pitch_length / 2 - agent_size
        goal_y = world.goal_size / 2 - agent_size
        goal_x = world.goal_depth
        pos = pos.at[:, Y].set(jnp.clip(pos[:, Y], -pitch_y, pitch_y))
        inside_goal_y_mask = jnp.abs(pos[:, Y]) < goal_y
        pos = pos.at[:, X].set(
            jnp.where(
                ~inside_goal_y_mask,
                jnp.clip(pos[:, X], -pitch_x, pitch_x),
                jnp.clip(pos[:, X], -pitch_x - goal_x, pitch_x + goal_x),
            )
        )
        if return_bool:
            return jnp.any(pos != orig_pos, axis=-1)
        else:
            return pos

    @jaxtyped(typechecker=beartype)
    def check_possession(self, PRNG_key: PRNGKeyArray, world: "FootballWorld"):
        ball, teammates, opposition, own_net, target_net = self.get_dynamic_params(
            world
        )
        agents_pos = jnp.stack(
            [agent.state.pos for agent in teammates + opposition],
            axis=1,
        )
        agents_vel = jnp.stack(
            [agent.state.vel for agent in teammates + opposition],
            axis=1,
        )
        ball_pos = ball.state.pos
        ball_vel = ball.state.vel
        ball_disps = ball_pos[:, None, :] - agents_pos
        relvels = ball_vel[:, None, :] - agents_vel
        dists = jnp.linalg.norm(
            ball_disps + relvels * self.possession_lookahead, axis=-1
        )
        mindist_team = jnp.argmin(dists, axis=-1) < len(teammates)
        team_possession = mindist_team
        self = self.replace(team_possession=team_possession)
        net_disps = target_net.state.pos[:, None, :] - agents_pos
        ball_dir = ball_disps / jnp.linalg.norm(ball_disps, axis=-1, keepdims=True)
        net_dir = net_disps / jnp.linalg.norm(net_disps, axis=-1, keepdims=True)
        side_dot_prod = (ball_dir * net_dir).sum(axis=-1)
        dists -= 0.5 * side_dot_prod * self.decision_strength
        if self.decision_strength != 1:
            PRNG_key, subkey = jax.random.split(PRNG_key)
            dists += (
                0.5
                * jax.random.normal(subkey, shape=dists.shape)
                * (1 - self.decision_strength) ** 2
            )
        mindist_agents = jnp.argmin(dists[:, : len(teammates)], axis=-1)
        agent_possession = dict(self.agent_possession)
        for i, agent in enumerate(teammates):
            agent_possession[agent.id] = mindist_agents == i
        self = self.replace(agent_possession=agent_possession)
        return self

    @jaxtyped(typechecker=beartype)
    def check_better_positions(
        self,
        PRNG_key: PRNGKeyArray,
        agent: "FootballAgent",
        world: "FootballWorld",
        env_index=Ellipsis,
    ) -> tuple["AgentPolicy", Array]:
        ball, teammates, opposition, own_net, target_net = self.get_dynamic_params(
            world
        )
        ball_pos = ball.state.pos
        if isinstance(env_index, Array) and env_index.dtype == jnp.bool_:
            # Use boolean masking for indexed environments
            ball_pos = jnp.where(env_index[:, None], ball_pos, ball_pos)
        target_pos_rel = self.objectives[agent.name]["target_pos_rel"]
        if isinstance(env_index, Array) and env_index.dtype == jnp.bool_:
            # Use boolean masking for indexed environments
            target_pos_rel = jnp.where(
                env_index[:, None], target_pos_rel, target_pos_rel
            )
        curr_target = target_pos_rel + ball_pos
        PRNG_key, subkey = jax.random.split(PRNG_key)
        samples = (
            jax.random.normal(
                subkey,
                shape=(ball_pos.shape[0], self.nsamples, world.dim_p),
            )
            * self.sigma
            * (1 + 3 * (1 - self.decision_strength))
        )
        samples = samples.at[:, ::2].set(samples[:, ::2] + ball_pos[:, None])

        agent_pos = agent.state.pos
        if isinstance(env_index, Array) and env_index.dtype == jnp.bool_:
            # Use boolean masking for indexed environments
            agent_pos = jnp.where(env_index[:, None], agent_pos, agent_pos)
        samples = samples.at[:, 1::2].set(samples[:, 1::2] + agent_pos[:, None])

        test_pos = jnp.concatenate([curr_target[:, None, :], samples], axis=1)
        test_pos_shape = test_pos.shape
        test_pos = self.clamp_pos(
            world,
            test_pos.reshape(test_pos_shape[0] * test_pos_shape[1], test_pos_shape[2]),
        ).reshape(*test_pos_shape)
        PRNG_key, subkey = jax.random.split(PRNG_key)
        self, values = self.get_pos_value(
            subkey,
            test_pos,
            agent=agent,
            world=world,
            env_index=env_index,
        )
        values = values.at[:, 0].set(
            values[:, 0] + self.replan_margin + 3 * (1 - self.decision_strength)
        )
        highest_value = values.argmax(axis=1)
        _highest_value = highest_value[None, ..., None]
        _highest_value = jnp.broadcast_to(
            _highest_value,
            _highest_value.shape[:-1] + (world.dim_p,),
        )
        if test_pos.shape[0] == 0 and _highest_value.shape[0] != 0:
            new_shape = (_highest_value.shape[0], test_pos.shape[1], test_pos.shape[2])
            test_pos = jnp.empty(new_shape, dtype=test_pos.dtype)
        best_pos = jnp.take_along_axis(
            test_pos,
            _highest_value,
            axis=1,
        )
        return self, best_pos[0]

    @jaxtyped(typechecker=beartype)
    def get_pos_value(
        self,
        PRNG_key: PRNGKeyArray,
        pos: Array,
        agent: "FootballAgent",
        world: "FootballWorld",
        env_index=Ellipsis,
    ) -> tuple["AgentPolicy", Array]:
        ball, teammates, opposition, own_net, target_net = self.get_dynamic_params(
            world
        )
        ball_pos = ball.state.pos
        target_net_pos = target_net.state.pos
        own_net_pos = own_net.state.pos

        if isinstance(env_index, Array) and env_index.dtype == jnp.bool_:
            # Use boolean masking for indexed environments
            ball_pos = jnp.where(env_index[:, None], ball_pos, ball_pos)
            target_net_pos = jnp.where(
                env_index[:, None], target_net_pos, target_net_pos
            )
            own_net_pos = jnp.where(env_index[:, None], own_net_pos, own_net_pos)

        # Add extra dimension to match original shape
        ball_pos = ball_pos[:, None]
        target_net_pos = target_net_pos[:, None]
        own_net_pos = own_net_pos[:, None]

        ball_vec = ball_pos - pos
        ball_vec /= jnp.linalg.norm(ball_vec, axis=-1, keepdims=True)
        ball_vec = jnp.where(jnp.isnan(ball_vec), jnp.zeros_like(ball_vec), ball_vec)
        # ball_dist_value prioritises positions relatively close to the ball
        ball_dist = jnp.linalg.norm(pos - ball_pos, axis=-1)
        ball_dist_value = jnp.exp(-2 * ball_dist**4)

        # side_value prevents being between the ball and the target goal
        net_vec = target_net_pos - pos
        net_vec /= jnp.linalg.norm(net_vec, axis=-1, keepdims=True)
        side_dot_prod = (ball_vec * net_vec).sum(axis=-1)
        side_value = jnp.minimum(side_dot_prod + 1.25, jnp.array(1))

        # defend_value prioritises being between the ball and your own goal while on defence
        own_net_vec = own_net_pos - pos
        own_net_vec /= jnp.linalg.norm(own_net_vec, axis=-1, keepdims=True)
        defend_dot_prod = (ball_vec * -own_net_vec).sum(axis=-1)
        defend_value = jnp.maximum(defend_dot_prod, jnp.array(0))

        # other_agent_value disincentivises being close to a teammate
        if len(teammates) > 1:
            agent_index = [i for i, a in enumerate(teammates) if a.id != agent.id][0]
            self, team_disps = self.get_separations(world, teammate=True)
            team_disps = jnp.concatenate(
                [team_disps[:, 0:agent_index], team_disps[:, agent_index + 1 :]], axis=1
            )
            if isinstance(env_index, Array) and env_index.dtype == jnp.bool_:
                # Use boolean masking for indexed environments
                team_disps_masked = jnp.where(
                    env_index[:, None], team_disps, team_disps
                )
            else:
                team_disps_masked = team_disps

            team_dists = jnp.linalg.norm(
                team_disps_masked[:, None] - pos[:, :, None], axis=-1
            )
            other_agent_value = jnp.linalg.norm(-jnp.exp(-5 * team_dists), axis=-1) + 1
        else:
            other_agent_value = 0

        # wall_value disincentivises being close to a wall
        wall_disps = self.get_wall_separations(world, pos)
        wall_dists = jnp.linalg.norm(wall_disps, axis=-1)
        wall_value = jnp.linalg.norm(-jnp.exp(-8 * wall_dists), axis=-1) + 1

        value = (
            wall_value + other_agent_value + ball_dist_value + side_value + defend_value
        ) / 5
        if self.decision_strength != 1:
            PRNG_key, subkey = jax.random.split(PRNG_key)
            value += jax.random.normal(subkey, value.shape) * (
                1 - self.decision_strength
            )
        return self, value

    @jaxtyped(typechecker=beartype)
    def get_wall_separations(self, world: "FootballWorld", pos: Array) -> Array:
        top_wall_dist = -pos[:, Y] + world.pitch_width / 2
        bottom_wall_dist = pos[:, Y] + world.pitch_width / 2
        left_wall_dist = pos[:, X] + world.pitch_length / 2
        right_wall_dist = -pos[:, X] + world.pitch_length / 2
        vertical_wall_disp = jnp.zeros(pos.shape)
        vertical_wall_disp = vertical_wall_disp.at[:, Y].set(
            jnp.minimum(top_wall_dist, bottom_wall_dist)
        )
        mask = bottom_wall_dist < top_wall_dist
        _vertical_wall_disp = vertical_wall_disp[:, Y]
        _vertical_wall_disp = jnp.where(
            mask, _vertical_wall_disp * -1, _vertical_wall_disp
        )
        vertical_wall_disp = vertical_wall_disp.at[:, Y].set(_vertical_wall_disp)
        horizontal_wall_disp = jnp.zeros(pos.shape)
        horizontal_wall_disp = horizontal_wall_disp.at[:, X].set(
            jnp.minimum(left_wall_dist, right_wall_dist)
        )
        mask = left_wall_dist < right_wall_dist
        _horizontal_wall_disp = horizontal_wall_disp[:, X]
        _horizontal_wall_disp = jnp.where(
            mask, _horizontal_wall_disp * -1, _horizontal_wall_disp
        )
        horizontal_wall_disp = horizontal_wall_disp.at[:, X].set(_horizontal_wall_disp)
        return jnp.stack([vertical_wall_disp, horizontal_wall_disp], axis=-2)

    @jaxtyped(typechecker=beartype)
    def get_separations(
        self,
        world: "FootballWorld",
        teammate=False,
        opposition=False,
        vel=False,
    ) -> tuple["AgentPolicy", Array]:
        assert teammate or opposition, "One of teammate or opposition must be True"
        ball, teammates, _, own_net, target_net = self.get_dynamic_params(world)
        key = (teammate, opposition, vel)
        if key in self.team_disps:
            return self, self.team_disps[key]
        disps = []
        if teammate:
            for otheragent in teammates:
                if vel:
                    agent_disp = otheragent.state.vel
                else:
                    agent_disp = otheragent.state.pos
                disps.append(agent_disp)
        if opposition:
            for otheragent in opposition:
                if vel:
                    agent_disp = otheragent.state.vel
                else:
                    agent_disp = otheragent.state.pos
                disps.append(agent_disp)
        out = jnp.stack(disps, axis=1)

        team_disps = dict(self.team_disps)
        team_disps[key] = out
        self = self.replace(team_disps=team_disps)

        return self, out


@jaxtyped(typechecker=beartype)
class FootballWorld(World):
    n_blue_agents: int
    n_red_agents: int
    n_traj_points: int
    agent_size: float
    pitch_width: float
    pitch_length: float
    goal_size: float
    goal_depth: float
    traj_points: dict[str, dict[int, list[int]]]
    red_controller: AgentPolicy | None
    blue_controller: AgentPolicy | None

    @classmethod
    def create(
        cls,
        batch_dim: int,
        red_controller: AgentPolicy | None,
        blue_controller: AgentPolicy | None,
        **kwargs,
    ):
        n_blue_agents = kwargs.pop("n_blue_agents")
        n_red_agents = kwargs.pop("n_red_agents")
        n_traj_points = kwargs.pop("n_traj_points")
        agent_size = kwargs.pop("agent_size")
        pitch_width = kwargs.pop("pitch_width")
        pitch_length = kwargs.pop("pitch_length")
        goal_size = kwargs.pop("goal_size")
        goal_depth = kwargs.pop("goal_depth")
        base_world = World.create(batch_dim=batch_dim, **kwargs)
        return cls(
            **dataclass_to_dict_first_layer(base_world),
            n_blue_agents=n_blue_agents,
            n_red_agents=n_red_agents,
            n_traj_points=n_traj_points,
            agent_size=agent_size,
            pitch_width=pitch_width,
            pitch_length=pitch_length,
            goal_size=goal_size,
            goal_depth=goal_depth,
            traj_points={},
            red_controller=red_controller,
            blue_controller=blue_controller,
        )

    def replace(self, **kwargs) -> "FootballWorld":
        if "blue_agents" in kwargs:
            blue_agents = kwargs.pop("blue_agents")
            agents = self.agents
            agents = blue_agents + agents[self.n_blue_agents :]
            self = self.replace(agents=agents)
        if "red_agents" in kwargs:
            red_agents = kwargs.pop("red_agents")
            agents = self.agents
            agents = (
                agents[: self.n_blue_agents]
                + red_agents
                + agents[self.n_blue_agents + self.n_red_agents :]
            )
            self = self.replace(agents=agents)
        if "ball" in kwargs:
            ball = kwargs.pop("ball")
            agents = self.agents
            agents = agents[: self.n_blue_agents + self.n_red_agents] + [ball]
            self = self.replace(agents=agents)

        return World.replace(self, **kwargs)

    @property
    def blue_agents(self) -> list["FootballAgent"]:
        return self.agents[: self.n_blue_agents]

    @property
    def red_agents(self) -> list["FootballAgent"]:
        return self.agents[self.n_blue_agents : self.n_blue_agents + self.n_red_agents]

    @property
    def ball(self) -> "BallAgent":
        return self.agents[self.n_blue_agents + self.n_red_agents]

    @property
    def right_top_wall(self):
        return self.landmarks[0]

    @property
    def left_top_wall(self):
        return self.landmarks[1]

    @property
    def right_bottom_wall(self):
        return self.landmarks[2]

    @property
    def left_bottom_wall(self):
        return self.landmarks[3]

    @property
    def blue_net(self):
        return self.landmarks[4]

    @property
    def red_net(self):
        return self.landmarks[5]


@jaxtyped(typechecker=beartype)
class FootballColor(Enum):
    BLUE = (0.22, 0.49, 0.72)
    RED = (0.89, 0.10, 0.11)
    GRAY = (0.5, 0.5, 0.5, 0.5)


@jaxtyped(typechecker=beartype)
class Scenario(BaseScenario[FootballWorld]):
    n_blue_agents: int
    n_red_agents: int
    ai_red_agents: bool
    ai_blue_agents: bool
    physically_different: bool
    spawn_in_formation: bool
    only_blue_formation: bool
    formation_agents_per_column: int
    randomise_formation_indices: bool
    formation_noise: float
    n_traj_points: int
    ai_speed_strength: float
    ai_decision_strength: float
    ai_precision_strength: float
    disable_ai_red: bool
    agent_size: float
    goal_size: float
    goal_depth: float
    pitch_length: float
    pitch_width: float
    ball_mass: float
    ball_size: float
    u_multiplier: float
    enable_shooting: bool
    u_rot_multiplier: float
    u_shoot_multiplier: float
    shooting_radius: float
    shooting_angle: float
    max_speed: float
    ball_max_speed: float
    dense_reward: bool
    pos_shaping_factor_ball_goal: float
    pos_shaping_factor_agent_ball: float
    distance_to_ball_trigger: float
    scoring_reward: float
    observe_teammates: bool
    observe_adversaries: bool
    dict_obs: bool

    blue_color: FootballColor
    red_color: FootballColor

    background_entities: list[Landmark]

    left_goal_pos: Array | None
    right_goal_pos: Array | None
    _done: Array | None
    _sparse_reward_blue: Array | None
    _sparse_reward_red: Array | None
    _dense_reward_blue: Array | None
    _dense_reward_red: Array | None
    _render_field: bool | None
    min_agent_dist_to_ball_blue: Array | None
    min_agent_dist_to_ball_red: Array | None
    _reset_agent_range: Array | None
    _reset_agent_offset_blue: Array | None
    _reset_agent_offset_red: Array | None
    _agents_rel_pos_to_ball: Array | None
    _agent_dist_to_ball: Array | None
    _agents_closest_to_ball: Array | None
    reset: bool

    @classmethod
    def create(cls, **kwargs):
        # Scenario config
        viewer_size = kwargs.pop("viewer_size", (1200, 800))
        visualize_semidims = False

        base_scenario = BaseScenario.create(
            viewer_size=viewer_size, visualize_semidims=visualize_semidims
        )

        # Agents config
        n_blue_agents = kwargs.pop("n_blue_agents", 3)
        n_red_agents = kwargs.pop("n_red_agents", 3)
        # What agents should be learning and what controlled by the heuristic (ai)
        ai_red_agents = kwargs.pop("ai_red_agents", True)
        ai_blue_agents = kwargs.pop("ai_blue_agents", False)

        # When you have 5 blue agents there is the options of introducing physical differences with the following roles:
        # 1 goalkeeper -> slow and big
        # 2 defenders -> normal size and speed (agent_size, u_multiplier, max_speed)
        # 2 attackers -> small and fast
        physically_different = kwargs.pop("physically_different", False)

        # Agent spawning
        spawn_in_formation = kwargs.pop("spawn_in_formation", False)
        only_blue_formation = kwargs.pop(
            "only_blue_formation", True
        )  # Only spawn blue agents in formation
        formation_agents_per_column = kwargs.pop("formation_agents_per_column", 2)
        randomise_formation_indices = kwargs.pop(
            "randomise_formation_indices", False
        )  # If False, each agent will always be in the same formation spot
        formation_noise = kwargs.pop(
            "formation_noise", 0.2
        )  # Noise on formation positions

        # Ai config
        n_traj_points = kwargs.pop(
            "n_traj_points", 0
        )  # Number of spline trajectory points to plot for heuristic (ai) agents
        ai_speed_strength = kwargs.pop(
            "ai_strength", 1.0
        )  # The speed of the ai 0<=x<=1
        ai_decision_strength = kwargs.pop(
            "ai_decision_strength", 1.0
        )  # The decision strength of the ai 0<=x<=1
        ai_precision_strength = kwargs.pop(
            "ai_precision_strength", 1.0
        )  # The precision strength of the ai 0<=x<=1
        disable_ai_red = kwargs.pop("disable_ai_red", False)

        # Task sizes
        agent_size = kwargs.pop("agent_size", 0.025)
        goal_size = kwargs.pop("goal_size", 0.35)
        goal_depth = kwargs.pop("goal_depth", 0.1)
        pitch_length = kwargs.pop("pitch_length", 3.0)
        pitch_width = kwargs.pop("pitch_width", 1.5)
        ball_mass = kwargs.pop("ball_mass", 0.25)
        ball_size = kwargs.pop("ball_size", 0.02)

        # Actions
        u_multiplier = kwargs.pop("u_multiplier", 0.1)

        # Actions shooting
        enable_shooting = kwargs.pop(
            "enable_shooting", False
        )  # Whether to enable an extra 2 actions (for rotation and shooting). Only avaioable for non-ai agents
        u_rot_multiplier = kwargs.pop("u_rot_multiplier", 0.0003)
        u_shoot_multiplier = kwargs.pop("u_shoot_multiplier", 0.6)
        shooting_radius = kwargs.pop("shooting_radius", 0.08)
        shooting_angle = kwargs.pop("shooting_angle", jnp.pi / 2)

        # Speeds
        max_speed = kwargs.pop("max_speed", 0.15)
        ball_max_speed = kwargs.pop("ball_max_speed", 0.3)

        # Rewards
        dense_reward = kwargs.pop("dense_reward", True)
        pos_shaping_factor_ball_goal = kwargs.pop(
            "pos_shaping_factor_ball_goal", 10.0
        )  # Reward for moving the ball towards the opponents' goal. This can be annealed in a curriculum.
        pos_shaping_factor_agent_ball = kwargs.pop(
            "pos_shaping_factor_agent_ball", 0.1
        )  # Reward for moving the closest agent to the ball in a team closer to it.
        # This is useful for exploration and can be annealed in a curriculum.
        # This reward does not trigger if the agent is less than distance_to_ball_trigger from the ball or the ball is moving
        distance_to_ball_trigger = kwargs.pop("distance_to_ball_trigger", 0.4)
        scoring_reward = kwargs.pop(
            "scoring_reward", 100.0
        )  # Discrete reward for scoring

        # Observations
        observe_teammates = kwargs.pop("observe_teammates", True)
        observe_adversaries = kwargs.pop("observe_adversaries", True)
        dict_obs = kwargs.pop("dict_obs", False)

        if kwargs.pop("dense_reward_ratio", None) is not None:
            raise ValueError(
                "dense_reward_ratio in football is deprecated, please use `dense_reward` "
                "which is a bool that turns on/off the dense reward"
            )
        ScenarioUtils.check_kwargs_consumed(kwargs)

        blue_color = FootballColor.BLUE
        red_color = FootballColor.RED

        scenario = cls(
            **dataclass_to_dict_first_layer(base_scenario),
            n_blue_agents=n_blue_agents,
            n_red_agents=n_red_agents,
            ai_red_agents=ai_red_agents,
            ai_blue_agents=ai_blue_agents,
            physically_different=physically_different,
            spawn_in_formation=spawn_in_formation,
            only_blue_formation=only_blue_formation,
            formation_agents_per_column=formation_agents_per_column,
            randomise_formation_indices=randomise_formation_indices,
            formation_noise=formation_noise,
            n_traj_points=n_traj_points,
            ai_speed_strength=ai_speed_strength,
            ai_decision_strength=ai_decision_strength,
            ai_precision_strength=ai_precision_strength,
            disable_ai_red=disable_ai_red,
            agent_size=agent_size,
            goal_size=goal_size,
            goal_depth=goal_depth,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
            ball_mass=ball_mass,
            ball_size=ball_size,
            u_multiplier=u_multiplier,
            enable_shooting=enable_shooting,
            u_rot_multiplier=u_rot_multiplier,
            u_shoot_multiplier=u_shoot_multiplier,
            shooting_radius=shooting_radius,
            shooting_angle=shooting_angle,
            max_speed=max_speed,
            ball_max_speed=ball_max_speed,
            dense_reward=dense_reward,
            pos_shaping_factor_ball_goal=pos_shaping_factor_ball_goal,
            pos_shaping_factor_agent_ball=pos_shaping_factor_agent_ball,
            distance_to_ball_trigger=distance_to_ball_trigger,
            scoring_reward=scoring_reward,
            observe_teammates=observe_teammates,
            observe_adversaries=observe_adversaries,
            dict_obs=dict_obs,
            blue_color=blue_color,
            red_color=red_color,
            background_entities=[],
            left_goal_pos=None,
            right_goal_pos=None,
            _done=None,
            _sparse_reward_blue=None,
            _sparse_reward_red=None,
            _dense_reward_blue=None,
            _dense_reward_red=None,
            _render_field=None,
            min_agent_dist_to_ball_blue=None,
            min_agent_dist_to_ball_red=None,
            _reset_agent_range=None,
            _reset_agent_offset_blue=None,
            _reset_agent_offset_red=None,
            _agents_rel_pos_to_ball=None,
            _agent_dist_to_ball=None,
            _agents_closest_to_ball=None,
            reset=False,
        )

        def init_background():
            # Add landmarks
            background = Landmark.create(
                name="Background",
                collide=False,
                movable=False,
                shape=Box(length=pitch_length, width=pitch_width),
                color=Color.GREEN,
            )

            centre_circle_outer = Landmark.create(
                name="Centre Circle Outer",
                collide=False,
                movable=False,
                shape=Sphere(radius=goal_size / 2),
                color=Color.WHITE,
            )

            centre_circle_inner = Landmark.create(
                name="Centre Circle Inner",
                collide=False,
                movable=False,
                shape=Sphere(goal_size / 2 - 0.02),
                color=Color.GREEN,
            )

            centre_line = Landmark.create(
                name="Centre Line",
                collide=False,
                movable=False,
                shape=Line(length=pitch_width - 2 * agent_size),
                color=Color.WHITE,
            )

            right_line = Landmark.create(
                name="Right Line",
                collide=False,
                movable=False,
                shape=Line(length=pitch_width - 2 * agent_size),
                color=Color.WHITE,
            )

            left_line = Landmark.create(
                name="Left Line",
                collide=False,
                movable=False,
                shape=Line(length=pitch_width - 2 * agent_size),
                color=Color.WHITE,
            )

            top_line = Landmark.create(
                name="Top Line",
                collide=False,
                movable=False,
                shape=Line(length=pitch_length - 2 * agent_size),
                color=Color.WHITE,
            )

            bottom_line = Landmark.create(
                name="Bottom Line",
                collide=False,
                movable=False,
                shape=Line(length=pitch_length - 2 * agent_size),
                color=Color.WHITE,
            )

            background_entities = [
                background,
                centre_circle_outer,
                centre_circle_inner,
                centre_line,
                right_line,
                left_line,
                top_line,
                bottom_line,
            ]
            return background_entities

        background_entities = init_background()
        scenario = scenario.replace(background_entities=background_entities)
        return scenario

    def replace(self, **kwargs) -> "Scenario":
        if "blue_agents" in kwargs:
            blue_agents = kwargs.pop("blue_agents")
            self = self.replace(world=self.world.replace(blue_agents=blue_agents))
        if "red_agents" in kwargs:
            red_agents = kwargs.pop("red_agents")
            self = self.replace(world=self.world.replace(red_agents=red_agents))
        if "ball" in kwargs:
            ball = kwargs.pop("ball")
            self = self.replace(world=self.world.replace(ball=ball))
        if "red_controller" in kwargs:
            red_controller = kwargs.pop("red_controller")
            self = self.replace(world=self.world.replace(red_controller=red_controller))
        if "blue_controller" in kwargs:
            blue_controller = kwargs.pop("blue_controller")
            self = self.replace(
                world=self.world.replace(blue_controller=blue_controller)
            )

        return super(BaseScenario, self).replace(**kwargs)

    @property
    def red_controller(self) -> "AgentPolicy":
        return self.world.red_controller

    @property
    def blue_controller(self) -> "AgentPolicy":
        return self.world.blue_controller

    @property
    def blue_agents(self) -> list["FootballAgent"]:
        return self.world.blue_agents

    @property
    def red_agents(self) -> list["FootballAgent"]:
        return self.world.red_agents

    @property
    def ball(self) -> "BallAgent":
        return self.world.ball

    @property
    def background(self) -> list["Landmark"]:
        return self.background_entities[0]

    @property
    def centre_circle_outer(self) -> "Landmark":
        return self.background_entities[1]

    @property
    def centre_circle_inner(self) -> "Landmark":
        return self.background_entities[2]

    @property
    def right_top_wall(self) -> "Landmark":
        return self.world.right_top_wall

    @property
    def left_top_wall(self) -> "Landmark":
        return self.world.left_top_wall

    @property
    def right_bottom_wall(self) -> "Landmark":
        return self.world.right_bottom_wall

    @property
    def left_bottom_wall(self) -> "Landmark":
        return self.world.left_bottom_wall

    @property
    def blue_net(self) -> "Landmark":
        return self.world.blue_net

    @property
    def red_net(self) -> "Landmark":
        return self.world.red_net

    @property
    def traj_points(self):
        return self.world.traj_points

    @jaxtyped(typechecker=beartype)
    def make_world(self, batch_dim: int, **kwargs) -> "Scenario":
        world = self.init_world(batch_dim)
        world = self.init_agents(world)
        world = self.init_ball(world)
        world = self.init_walls(world)
        world = self.init_goals(world)
        world = self.init_traj_pts(world)

        # Cached values
        left_goal_pos = jnp.asarray([-self.pitch_length / 2 - self.ball_size / 2, 0])
        right_goal_pos = -left_goal_pos
        _done = jnp.zeros(batch_dim, dtype=jnp.bool_)
        _sparse_reward_blue = jnp.zeros(batch_dim)
        _sparse_reward_red = _sparse_reward_blue
        _dense_reward_blue = jnp.zeros(batch_dim)
        _dense_reward_red = _dense_reward_blue
        _render_field = True
        min_agent_dist_to_ball_blue = None
        min_agent_dist_to_ball_red = None

        _reset_agent_range = jnp.asarray([self.pitch_length / 2, self.pitch_width])
        _reset_agent_offset_blue = jnp.asarray(
            [-self.pitch_length / 2 + self.agent_size, -self.pitch_width / 2]
        )
        _reset_agent_offset_red = jnp.asarray([-self.agent_size, -self.pitch_width / 2])

        _agents_rel_pos_to_ball = jnp.zeros(
            (batch_dim, self.n_blue_agents + self.n_red_agents, world.dim_p)
        )
        _agent_dist_to_ball = jnp.zeros(
            (batch_dim, self.n_blue_agents + self.n_red_agents)
        )
        _agents_closest_to_ball = _agent_dist_to_ball == _agent_dist_to_ball.min(
            axis=-1, keepdims=True
        )
        self = self.replace(
            world=world,
            left_goal_pos=left_goal_pos,
            right_goal_pos=right_goal_pos,
            _done=_done,
            _sparse_reward_blue=_sparse_reward_blue,
            _sparse_reward_red=_sparse_reward_red,
            _dense_reward_blue=_dense_reward_blue,
            _dense_reward_red=_dense_reward_red,
            _render_field=_render_field,
            min_agent_dist_to_ball_blue=min_agent_dist_to_ball_blue,
            min_agent_dist_to_ball_red=min_agent_dist_to_ball_red,
            _reset_agent_range=_reset_agent_range,
            _reset_agent_offset_blue=_reset_agent_offset_blue,
            _reset_agent_offset_red=_reset_agent_offset_red,
            _agents_rel_pos_to_ball=_agents_rel_pos_to_ball,
            _agent_dist_to_ball=_agent_dist_to_ball,
            _agents_closest_to_ball=_agents_closest_to_ball,
        )
        return self

    @jaxtyped(typechecker=beartype)
    def init_world(self, batch_dim: int) -> FootballWorld:
        # Add agents
        red_controller = (
            AgentPolicy.create(
                team="Red",
                disabled=self.disable_ai_red,
                speed_strength=(
                    self.ai_speed_strength[1]
                    if isinstance(self.ai_speed_strength, tuple)
                    else self.ai_speed_strength
                ),
                precision_strength=(
                    self.ai_precision_strength[1]
                    if isinstance(self.ai_precision_strength, tuple)
                    else self.ai_precision_strength
                ),
                decision_strength=(
                    self.ai_decision_strength[1]
                    if isinstance(self.ai_decision_strength, tuple)
                    else self.ai_decision_strength
                ),
            )
            if self.ai_red_agents
            else None
        )
        blue_controller = (
            AgentPolicy.create(
                team="Blue",
                speed_strength=(
                    self.ai_speed_strength[0]
                    if isinstance(self.ai_speed_strength, tuple)
                    else self.ai_speed_strength
                ),
                precision_strength=(
                    self.ai_precision_strength[0]
                    if isinstance(self.ai_precision_strength, tuple)
                    else self.ai_precision_strength
                ),
                decision_strength=(
                    self.ai_decision_strength[0]
                    if isinstance(self.ai_decision_strength, tuple)
                    else self.ai_decision_strength
                ),
            )
            if self.ai_blue_agents
            else None
        )

        # Make world
        world = FootballWorld.create(
            batch_dim=batch_dim,
            dt=0.1,
            drag=0.05,
            x_semidim=self.pitch_length / 2 + self.goal_depth - self.agent_size,
            y_semidim=self.pitch_width / 2 - self.agent_size,
            substeps=2,
            red_controller=red_controller,
            blue_controller=blue_controller,
            **dataclass_to_dict_first_layer(self),
        )
        return world

    @jaxtyped(typechecker=beartype)
    def init_agents(self, world: FootballWorld) -> FootballWorld:
        def get_physically_different_agents():
            assert self.n_blue_agents == 5, "Physical differences only for 5 agents"

            def attacker(i):
                attacker_shoot_multiplier_decrease = -0.2
                attacker_multiplier_increase = 0.1
                attacker_speed_increase = 0.05
                attacker_radius_decrease = -0.005
                return FootballAgent.create(
                    name=f"agent_blue_{i}",
                    shape=Sphere(radius=self.agent_size + attacker_radius_decrease),
                    action_script=(
                        self.blue_controller.run if self.ai_blue_agents else None
                    ),
                    u_multiplier=(
                        [
                            self.u_multiplier + attacker_multiplier_increase,
                            self.u_multiplier + attacker_multiplier_increase,
                        ]
                        if not self.enable_shooting
                        else [
                            self.u_multiplier + attacker_multiplier_increase,
                            self.u_multiplier + attacker_multiplier_increase,
                            self.u_rot_multiplier,
                            self.u_shoot_multiplier
                            + attacker_shoot_multiplier_decrease,
                        ]
                    ),
                    max_speed=self.max_speed + attacker_speed_increase,
                    dynamics=(
                        Holonomic()
                        if not self.enable_shooting
                        else HolonomicWithRotation()
                    ),
                    action_size=2 if not self.enable_shooting else 4,
                    color=self.blue_color,
                    alpha=1,
                )

            def defender(i):

                return FootballAgent.create(
                    name=f"agent_blue_{i}",
                    shape=Sphere(radius=self.agent_size),
                    action_script=(
                        self.blue_controller.run if self.ai_blue_agents else None
                    ),
                    u_multiplier=(
                        [self.u_multiplier, self.u_multiplier]
                        if not self.enable_shooting
                        else [
                            self.u_multiplier,
                            self.u_multiplier,
                            self.u_rot_multiplier,
                            self.u_shoot_multiplier,
                        ]
                    ),
                    max_speed=self.max_speed,
                    dynamics=(
                        Holonomic()
                        if not self.enable_shooting
                        else HolonomicWithRotation()
                    ),
                    action_size=2 if not self.enable_shooting else 4,
                    color=self.blue_color,
                    alpha=1,
                )

            def goal_keeper(i):
                goalie_shoot_multiplier_increase = 0.2
                goalie_radius_increase = 0.01
                goalie_speed_decrease = -0.1
                goalie_multiplier_decrease = -0.05
                return FootballAgent.create(
                    name=f"agent_blue_{i}",
                    shape=Sphere(radius=self.agent_size + goalie_radius_increase),
                    action_script=(
                        self.blue_controller.run if self.ai_blue_agents else None
                    ),
                    u_multiplier=(
                        [
                            self.u_multiplier + goalie_multiplier_decrease,
                            self.u_multiplier + goalie_multiplier_decrease,
                        ]
                        if not self.enable_shooting
                        else [
                            self.u_multiplier + goalie_multiplier_decrease,
                            self.u_multiplier + goalie_multiplier_decrease,
                            self.u_rot_multiplier + goalie_shoot_multiplier_increase,
                            self.u_shoot_multiplier,
                        ]
                    ),
                    max_speed=self.max_speed + goalie_speed_decrease,
                    dynamics=(
                        Holonomic()
                        if not self.enable_shooting
                        else HolonomicWithRotation()
                    ),
                    action_size=2 if not self.enable_shooting else 4,
                    color=self.blue_color,
                    alpha=1,
                )

            agents = [
                attacker(0),
                attacker(1),
                defender(2),
                defender(3),
                goal_keeper(4),
            ]
            return agents

        if self.physically_different:
            blue_agents = get_physically_different_agents()
            for agent in blue_agents:
                world = world.add_agent(agent)
        else:
            for i in range(self.n_blue_agents):

                def action_script(
                    PRNG_key: PRNGKeyArray,
                    agent: "FootballAgent",
                    world: "FootballWorld",
                ):
                    return AgentPolicy.run(
                        world.blue_controller, PRNG_key, agent, world
                    )

                agent = FootballAgent.create(
                    name=f"agent_blue_{i}",
                    shape=Sphere(radius=self.agent_size),
                    action_script=(action_script if self.ai_blue_agents else None),
                    u_multiplier=(
                        [self.u_multiplier, self.u_multiplier]
                        if not self.enable_shooting
                        else [
                            self.u_multiplier,
                            self.u_multiplier,
                            self.u_rot_multiplier,
                            self.u_shoot_multiplier,
                        ]
                    ),
                    max_speed=self.max_speed,
                    dynamics=(
                        Holonomic()
                        if not self.enable_shooting
                        else HolonomicWithRotation()
                    ),
                    action_size=2 if not self.enable_shooting else 4,
                    color=self.blue_color,
                    alpha=1,
                )
                world = world.add_agent(agent)

        for i in range(self.n_red_agents):

            def action_script(
                PRNG_key: PRNGKeyArray, agent: "FootballAgent", world: "FootballWorld"
            ):
                return AgentPolicy.run(world.red_controller, PRNG_key, agent, world)

            agent = FootballAgent.create(
                name=f"agent_red_{i}",
                shape=Sphere(radius=self.agent_size),
                action_script=(action_script if self.ai_red_agents else None),
                u_multiplier=(
                    [self.u_multiplier, self.u_multiplier]
                    if not self.enable_shooting or self.ai_red_agents
                    else [
                        self.u_multiplier,
                        self.u_multiplier,
                        self.u_rot_multiplier,
                        self.u_shoot_multiplier,
                    ]
                ),
                max_speed=self.max_speed,
                dynamics=(
                    Holonomic()
                    if not self.enable_shooting or self.ai_red_agents
                    else HolonomicWithRotation()
                ),
                action_size=2 if not self.enable_shooting or self.ai_red_agents else 4,
                color=self.red_color,
                alpha=1,
            )
            world = world.add_agent(agent)

        return world

    @jaxtyped(typechecker=beartype)
    def init_ball(self, world: FootballWorld):
        # Add Ball
        ball = BallAgent.create(
            name="Ball",
            shape=Sphere(radius=self.ball_size),
            action_script=get_ball_action_script(
                self.agent_size, self.pitch_width, self.pitch_length, self.goal_size
            ),
            max_speed=self.ball_max_speed,
            mass=self.ball_mass,
            alpha=1,
            color=Color.BLACK,
        )
        world = world.add_agent(ball)
        return world

    @jaxtyped(typechecker=beartype)
    def init_walls(self, world: FootballWorld):
        right_top_wall = Landmark.create(
            name="Right Top Wall",
            collide=True,
            movable=False,
            shape=Line(
                length=self.pitch_width / 2 - self.agent_size - self.goal_size / 2,
            ),
            color=Color.WHITE,
        )
        world = world.add_landmark(right_top_wall)

        left_top_wall = Landmark.create(
            name="Left Top Wall",
            collide=True,
            movable=False,
            shape=Line(
                length=self.pitch_width / 2 - self.agent_size - self.goal_size / 2,
            ),
            color=Color.WHITE,
        )
        world = world.add_landmark(left_top_wall)

        right_bottom_wall = Landmark.create(
            name="Right Bottom Wall",
            collide=True,
            movable=False,
            shape=Line(
                length=self.pitch_width / 2 - self.agent_size - self.goal_size / 2,
            ),
            color=Color.WHITE,
        )
        world = world.add_landmark(right_bottom_wall)

        left_bottom_wall = Landmark.create(
            name="Left Bottom Wall",
            collide=True,
            movable=False,
            shape=Line(
                length=self.pitch_width / 2 - self.agent_size - self.goal_size / 2,
            ),
            color=Color.WHITE,
        )
        world = world.add_landmark(left_bottom_wall)

        return world

    @jaxtyped(typechecker=beartype)
    def init_goals(self, world: FootballWorld):
        right_goal_back = Landmark.create(
            name="Right Goal Back",
            collide=True,
            movable=False,
            shape=Line(length=self.goal_size),
            color=Color.WHITE,
        )
        world = world.add_landmark(right_goal_back)

        left_goal_back = Landmark.create(
            name="Left Goal Back",
            collide=True,
            movable=False,
            shape=Line(length=self.goal_size),
            color=Color.WHITE,
        )
        world = world.add_landmark(left_goal_back)

        right_goal_top = Landmark.create(
            name="Right Goal Top",
            collide=True,
            movable=False,
            shape=Line(length=self.goal_depth),
            color=Color.WHITE,
        )
        world = world.add_landmark(right_goal_top)

        left_goal_top = Landmark.create(
            name="Left Goal Top",
            collide=True,
            movable=False,
            shape=Line(length=self.goal_depth),
            color=Color.WHITE,
        )
        world = world.add_landmark(left_goal_top)

        right_goal_bottom = Landmark.create(
            name="Right Goal Bottom",
            collide=True,
            movable=False,
            shape=Line(length=self.goal_depth),
            color=Color.WHITE,
        )
        world = world.add_landmark(right_goal_bottom)

        left_goal_bottom = Landmark.create(
            name="Left Goal Bottom",
            collide=True,
            movable=False,
            shape=Line(length=self.goal_depth),
            color=Color.WHITE,
        )
        world = world.add_landmark(left_goal_bottom)

        blue_net = Landmark.create(
            name="Blue Net",
            collide=False,
            movable=False,
            shape=Box(length=self.goal_depth, width=self.goal_size),
            color=FootballColor.GRAY,
        )
        world = world.add_landmark(blue_net)

        red_net = Landmark.create(
            name="Red Net",
            collide=False,
            movable=False,
            shape=Box(length=self.goal_depth, width=self.goal_size),
            color=FootballColor.GRAY,
        )
        world = world.add_landmark(red_net)

        return world

    @jaxtyped(typechecker=beartype)
    def init_traj_pts(self, world: FootballWorld):
        traj_points = {"Red": {}, "Blue": {}}
        if self.ai_red_agents:
            for i, agent in enumerate(world.red_agents):
                _traj_points = []
                for j in range(self.n_traj_points):
                    pointj = Landmark.create(
                        name="Red {agent} Trajectory {pt}".format(
                            agent=agent.name, pt=j
                        ),
                        collide=False,
                        movable=False,
                        shape=Sphere(radius=0.01),
                        color=Color.GRAY,
                    )
                    world = world.add_landmark(pointj)
                    _traj_points.append(pointj.id)
                traj_points["Red"][agent.id] = _traj_points
        if self.ai_blue_agents:
            for i, agent in enumerate(world.blue_agents):
                _traj_points = []
                for j in range(self.n_traj_points):
                    pointj = Landmark.create(
                        name="Blue {agent} Trajectory {pt}".format(
                            agent=agent.name, pt=j
                        ),
                        collide=False,
                        movable=False,
                        shape=Sphere(radius=0.01),
                        color=Color.GRAY,
                    )
                    world = world.add_landmark(pointj)
                    _traj_points.append(pointj.id)
                traj_points["Blue"][agent.id] = _traj_points
        world = world.replace(traj_points=traj_points)
        return world

    @eqx.filter_jit
    @jaxtyped(typechecker=beartype)
    def reset_world_at(
        self,
        PRNG_key: PRNGKeyArray,
        env_index: Int[Array, f"{env_index_dim}"] | None = None,
    ):
        print("starting reset_world_at")
        PRNG_key, subkey = jax.random.split(PRNG_key)
        self = self.reset_agents(subkey, env_index)
        self = self.reset_ball(env_index)
        self = self.reset_walls(env_index)
        self = self.reset_goals(env_index)
        self = self.reset_controllers(env_index)
        _done = jnp.where(
            env_index is None,
            jnp.zeros_like(self._done, dtype=jnp.bool_),
            self._done.at[env_index].set(False),
        )
        self = self.replace(_done=_done)
        self = self.replace(reset=True)
        print("ending reset_world_at")
        return self

    @eqx.filter_jit
    @jaxtyped(typechecker=beartype)
    def reset_agents(
        self,
        PRNG_key: PRNGKeyArray,
        env_index: Int[Array, f"{env_index_dim}"] | None = None,
    ):
        PRNG_key, subkey = jax.random.split(PRNG_key)
        batch_index = env_index if env_index is not None else jnp.asarray(-1)
        if self.spawn_in_formation:
            agents = self._spawn_formation(subkey, self.blue_agents, True, env_index)
            self = self.replace(blue_agents=agents)
            if not self.only_blue_formation:
                PRNG_key, subkey = jax.random.split(PRNG_key)
                agents = self._spawn_formation(
                    subkey, self.red_agents, False, env_index
                )
                self = self.replace(red_agents=agents)
        else:
            blue_agents = []
            for agent in self.blue_agents:
                PRNG_key, subkey = jax.random.split(PRNG_key)
                pos = self._get_random_spawn_position(
                    subkey, blue=True, env_index=env_index
                )
                agent = agent.set_pos(
                    pos,
                    batch_index=batch_index,
                )
                blue_agents.append(agent)
            self = self.replace(blue_agents=blue_agents)

        if (
            self.spawn_in_formation and self.only_blue_formation
        ) or not self.spawn_in_formation:
            red_agents = []
            for agent in self.red_agents:
                PRNG_key, subkey = jax.random.split(PRNG_key)
                pos = self._get_random_spawn_position(
                    subkey, blue=False, env_index=env_index
                )
                agent = agent.set_pos(
                    pos,
                    batch_index=batch_index,
                )
                agent = agent.set_rot(
                    jnp.array([jnp.pi]),
                    batch_index=batch_index,
                )
                red_agents.append(agent)
            self = self.replace(red_agents=red_agents)
        return self

    @jaxtyped(typechecker=beartype)
    def reset_controllers(
        self,
        env_index: Int[Array, f"{env_index_dim}"] | None = None,
    ):
        red_controller = self.red_controller
        blue_controller = self.blue_controller
        if red_controller is not None:
            if not red_controller.initialised:
                red_controller = red_controller.init(self.world)
            red_controller = red_controller.reset(self.world, env_index)
        if blue_controller is not None:
            if not blue_controller.initialised:
                blue_controller = blue_controller.init(self.world)
            blue_controller = blue_controller.reset(self.world, env_index)
        self = self.replace(
            red_controller=red_controller, blue_controller=blue_controller
        )
        return self

    @jaxtyped(typechecker=beartype)
    def reset_ball(
        self,
        env_index: Int[Array, f"{env_index_dim}"] | None = None,
    ):
        if not self.ai_blue_agents:
            min_agent_dist_to_ball_blue = self.get_closest_agent_to_ball(
                self.blue_agents, env_index
            )
            if env_index is None:
                self = self.replace(
                    min_agent_dist_to_ball_blue=min_agent_dist_to_ball_blue
                )
            else:
                min_agent_dist_to_ball_blue = jnp.where(
                    env_index[:, None],
                    min_agent_dist_to_ball_blue,
                    min_agent_dist_to_ball_blue,
                )
                self = self.replace(
                    min_agent_dist_to_ball_blue=min_agent_dist_to_ball_blue
                )
        if not self.ai_red_agents:
            min_agent_dist_to_ball_red = self.get_closest_agent_to_ball(
                self.red_agents, env_index
            )
            if env_index is None:
                self = self.replace(
                    min_agent_dist_to_ball_red=min_agent_dist_to_ball_red
                )
            else:
                min_agent_dist_to_ball_red = jnp.where(
                    env_index[:, None],
                    min_agent_dist_to_ball_red,
                    min_agent_dist_to_ball_red,
                )
                self = self.replace(
                    min_agent_dist_to_ball_red=min_agent_dist_to_ball_red
                )

        if env_index is None:
            if not self.ai_blue_agents:
                ball = self.ball
                pos_shaping_blue = (
                    jnp.linalg.vector_norm(
                        self.ball.state.pos - self.right_goal_pos,
                        axis=-1,
                    )
                    * self.pos_shaping_factor_ball_goal
                )
                pos_shaping_agent_blue = (
                    self.min_agent_dist_to_ball_blue
                    * self.pos_shaping_factor_agent_ball
                )
                ball = ball.replace(
                    pos_shaping_blue=pos_shaping_blue,
                    pos_shaping_agent_blue=pos_shaping_agent_blue,
                )
                self = self.replace(ball=ball)
            if not self.ai_red_agents:
                ball = self.ball
                pos_shaping_red = (
                    jnp.linalg.vector_norm(
                        self.ball.state.pos - self.left_goal_pos,
                        axis=-1,
                    )
                    * self.pos_shaping_factor_ball_goal
                )

                pos_shaping_agent_red = (
                    self.min_agent_dist_to_ball_red * self.pos_shaping_factor_agent_ball
                )
                ball = ball.replace(
                    pos_shaping_red=pos_shaping_red,
                    pos_shaping_agent_red=pos_shaping_agent_red,
                )
                self = self.replace(ball=ball)

            if self.enable_shooting:
                ball = self.ball
                kicking_action = ball.kicking_action.at[:].set(0.0)
                ball = ball.replace(kicking_action=kicking_action)
                self = self.replace(ball=ball)
        else:
            if not self.ai_blue_agents:
                ball = self.ball
                pos_shaping_blue = (
                    jnp.linalg.vector_norm(
                        self.ball.state.pos[env_index] - self.right_goal_pos
                    )
                    * self.pos_shaping_factor_ball_goal
                )
                pos_shaping_agent_blue = (
                    self.min_agent_dist_to_ball_blue[env_index]
                    * self.pos_shaping_factor_agent_ball
                )
                ball = ball.replace(
                    pos_shaping_blue=pos_shaping_blue,
                    pos_shaping_agent_blue=pos_shaping_agent_blue,
                )
                self = self.replace(ball=ball)
            if not self.ai_red_agents:
                ball = self.ball
                pos_shaping_red = (
                    jnp.linalg.vector_norm(
                        self.ball.state.pos[env_index] - self.left_goal_pos
                    )
                    * self.pos_shaping_factor_ball_goal
                )

                pos_shaping_agent_red = (
                    self.min_agent_dist_to_ball_red[env_index]
                    * self.pos_shaping_factor_agent_ball
                )
                ball = ball.replace(
                    pos_shaping_red=pos_shaping_red,
                    pos_shaping_agent_red=pos_shaping_agent_red,
                )
                self = self.replace(ball=ball)
            if self.enable_shooting:
                ball = self.ball
                kicking_action = ball.kicking_action.at[env_index].set(0.0)
                ball = ball.replace(kicking_action=kicking_action)
                self = self.replace(ball=ball)

        return self

    @jaxtyped(typechecker=beartype)
    def reset_walls(
        self,
        env_index: Int[Array, f"{env_index_dim}"] | None = None,
    ):
        landmarks = []
        batch_index = env_index if env_index is not None else jnp.asarray(-1)
        for landmark in self.world.landmarks:
            if landmark.name == "Left Top Wall":
                landmark = landmark.set_pos(
                    jnp.asarray(
                        [
                            -self.pitch_length / 2,
                            self.pitch_width / 4 + self.goal_size / 4,
                        ],
                    ),
                    batch_index=batch_index,
                )
                landmark = landmark.set_rot(
                    jnp.asarray([jnp.pi / 2]),
                    batch_index=batch_index,
                )
                landmarks.append(landmark)
            elif landmark.name == "Left Bottom Wall":
                landmark = landmark.set_pos(
                    jnp.asarray(
                        [
                            -self.pitch_length / 2,
                            -self.pitch_width / 4 - self.goal_size / 4,
                        ],
                    ),
                    batch_index=batch_index,
                )
                landmark = landmark.set_rot(
                    jnp.asarray([jnp.pi / 2]),
                    batch_index=batch_index,
                )
                landmarks.append(landmark)
            elif landmark.name == "Right Top Wall":
                landmark = landmark.set_pos(
                    jnp.asarray(
                        [
                            self.pitch_length / 2,
                            self.pitch_width / 4 + self.goal_size / 4,
                        ],
                    ),
                    batch_index=batch_index,
                )
                landmark = landmark.set_rot(
                    jnp.asarray([jnp.pi / 2]),
                    batch_index=batch_index,
                )
                landmarks.append(landmark)
            elif landmark.name == "Right Bottom Wall":
                landmark = landmark.set_pos(
                    jnp.asarray(
                        [
                            self.pitch_length / 2,
                            -self.pitch_width / 4 - self.goal_size / 4,
                        ],
                    ),
                    batch_index=batch_index,
                )
                landmark = landmark.set_rot(
                    jnp.asarray([jnp.pi / 2]),
                    batch_index=batch_index,
                )
                landmarks.append(landmark)
            else:
                landmarks.append(landmark)
        self = self.replace(world=self.world.replace(landmarks=landmarks))
        return self

    @jaxtyped(typechecker=beartype)
    def reset_goals(
        self,
        env_index: Int[Array, f"{env_index_dim}"] | None = None,
    ):
        landmarks = []
        batch_index = env_index if env_index is not None else jnp.asarray(-1)
        for landmark in self.world.landmarks:
            if landmark.name == "Left Goal Back":
                landmark = landmark.set_pos(
                    jnp.asarray(
                        [
                            -self.pitch_length / 2 - self.goal_depth + self.agent_size,
                            0.0,
                        ],
                    ),
                    batch_index=batch_index,
                )
                landmark = landmark.set_rot(
                    jnp.asarray([jnp.pi / 2]),
                    batch_index=batch_index,
                )
                landmarks.append(landmark)
            elif landmark.name == "Right Goal Back":
                landmark = landmark.set_pos(
                    jnp.asarray(
                        [
                            self.pitch_length / 2 + self.goal_depth - self.agent_size,
                            0.0,
                        ],
                    ),
                    batch_index=batch_index,
                )
                landmark = landmark.set_rot(
                    jnp.asarray([jnp.pi / 2]),
                    batch_index=batch_index,
                )
                landmarks.append(landmark)
            elif landmark.name == "Left Goal Top":
                landmark = landmark.set_pos(
                    jnp.asarray(
                        [
                            -self.pitch_length / 2
                            - self.goal_depth / 2
                            + self.agent_size,
                            self.goal_size / 2,
                        ],
                    ),
                    batch_index=batch_index,
                )
                landmarks.append(landmark)
            elif landmark.name == "Left Goal Bottom":
                landmark = landmark.set_pos(
                    jnp.asarray(
                        [
                            -self.pitch_length / 2
                            - self.goal_depth / 2
                            + self.agent_size,
                            -self.goal_size / 2,
                        ],
                    ),
                    batch_index=batch_index,
                )
                landmarks.append(landmark)
            elif landmark.name == "Right Goal Top":
                landmark = landmark.set_pos(
                    jnp.asarray(
                        [
                            self.pitch_length / 2
                            + self.goal_depth / 2
                            - self.agent_size,
                            self.goal_size / 2,
                        ],
                    ),
                    batch_index=batch_index,
                )
                landmarks.append(landmark)
            elif landmark.name == "Right Goal Bottom":
                landmark = landmark.set_pos(
                    jnp.asarray(
                        [
                            self.pitch_length / 2
                            + self.goal_depth / 2
                            - self.agent_size,
                            -self.goal_size / 2,
                        ],
                    ),
                    batch_index=batch_index,
                )
                landmarks.append(landmark)
            elif landmark.name == "Red Net":
                landmark = landmark.set_pos(
                    jnp.asarray(
                        [
                            self.pitch_length / 2
                            + self.goal_depth / 2
                            - self.agent_size / 2,
                            0.0,
                        ],
                    ),
                    batch_index=batch_index,
                )
                landmarks.append(landmark)
            elif landmark.name == "Blue Net":
                landmark = landmark.set_pos(
                    jnp.asarray(
                        [
                            -self.pitch_length / 2
                            - self.goal_depth / 2
                            + self.agent_size / 2,
                            0.0,
                        ],
                    ),
                    batch_index=batch_index,
                )
                landmarks.append(landmark)
            else:
                landmarks.append(landmark)
        self = self.replace(world=self.world.replace(landmarks=landmarks))
        return self

    @eqx.filter_jit
    @jaxtyped(typechecker=beartype)
    @chex.assert_max_traces(1)
    def _spawn_formation(
        self,
        PRNG_key: PRNGKeyArray,
        agents: list["FootballAgent"],
        blue: bool,
        env_index: Int[Array, f"{env_index_dim}"] | None = None,
    ):
        def compute_formation_positions(
            PRNG_key: PRNGKeyArray,
            batch_dim: int,
            dim_p: int,
            num_agents: int,
            blue: bool,
            pitch_length: float,
            goal_depth: float,
            formation_agents_per_column: int,
            pitch_width: float,
            formation_noise: float,
            env_index: Int[Array, f"{env_index_dim}"] | None,
        ):
            # Compute the target endpoint (depending on blue)
            endpoint = -(pitch_length / 2 + goal_depth) * (1 if blue else -1)
            # Determine number of columns and compute x positions
            n_columns = num_agents // formation_agents_per_column + 3
            x_lin = jnp.linspace(0, endpoint, n_columns)
            valid_x = x_lin[(x_lin != 0) & (x_lin != endpoint)]

            # Pre-allocate an array to store positions (shape: [num_agents, 2])
            positions = jnp.zeros((num_agents, 2))

            # The scan will iterate over each valid x and update the positions array.
            # The carry is a tuple (agent_idx, positions) where agent_idx tracks how many
            # agents have been assigned so far.
            def body_fn(carry, x):
                agent_idx, positions = carry

                # If no agents remain, do nothing.
                def do_nothing(carry):
                    return carry, None

                # Otherwise, update the positions for this column.
                def update(carry):
                    agent_idx, positions = carry
                    remaining = num_agents - agent_idx
                    # Use up to formation_agents_per_column agents in this column.
                    n_agents_in_col = jnp.minimum(
                        formation_agents_per_column, remaining
                    )
                    # Create a full linspace along y, then select the inner values.
                    y_lin = jnp.linspace(
                        pitch_width / 2,
                        -pitch_width / 2,
                        formation_agents_per_column + 2,
                    )
                    # Here we need only the inner points; note that n_agents_in_col might be less than formation_agents_per_column.
                    # Since slicing with a dynamic size is not directly supported in jitted code,
                    # we can use jax.lax.dynamic_slice.
                    start = 1  # skip the first element
                    slice_size = (n_agents_in_col,)
                    y_vals = jax.lax.dynamic_slice(y_lin, (start,), slice_size)
                    # Create an array for the x-coordinate (same shape as y_vals).
                    x_vals = jnp.full((n_agents_in_col,), x)
                    col_positions = jnp.stack(
                        [x_vals, y_vals], axis=-1
                    )  # shape: (n_agents_in_col, 2)
                    # Update the preallocated positions array at the appropriate index.
                    positions = jax.lax.dynamic_update_slice(
                        positions, col_positions, (agent_idx, 0)
                    )
                    agent_idx = agent_idx + n_agents_in_col
                    return (agent_idx, positions)

                new_carry, _ = jax.lax.cond(
                    agent_idx >= num_agents, do_nothing, update, (agent_idx, positions)
                )
                return new_carry, None

            # Run the scan over all valid x positions.
            (final_idx, positions), _ = jax.lax.scan(body_fn, (0, positions), valid_x)

            # Generate random noise for each agent.
            keys = jax.random.split(PRNG_key, num_agents)

            def sample_noise(key):
                shape = (dim_p,) if env_index is None else (batch_dim, dim_p)
                return jax.random.uniform(key, shape) - 0.5

            noise = jax.vmap(sample_noise)(keys)

            # Adjust positions shape for broadcasting.
            if env_index is None:
                base_positions = jnp.broadcast_to(
                    positions[:, None, :], (num_agents, batch_dim, dim_p)
                )
            else:
                base_positions = positions[:, None, :]

            # Compute the final positions by adding the noise scaled by formation_noise.
            final_positions = base_positions + noise * formation_noise
            return final_positions

        PRNG_key, subkey = jax.random.split(PRNG_key)
        if self.randomise_formation_indices:
            order = jax.random.permutation(subkey, len(agents)).tolist()
            agents = [agents[i] for i in order]
        # Compute all the new positions in pure JAX.
        new_positions = compute_formation_positions(
            PRNG_key,
            self.world.batch_dim,
            self.world.dim_p,
            len(agents),
            blue,
            self.pitch_length,
            self.goal_depth,
            self.formation_agents_per_column,
            self.pitch_width,
            self.formation_noise,
            env_index,
        )

        # Finally, update each agent with its computed position.
        # (This loop happens in Python because FootballAgent is a Python object.)
        batch_index = env_index if env_index is not None else jnp.asarray(-1)
        for i, agent in enumerate(agents):
            agents[i] = agent.set_pos(
                new_positions[i],
                batch_index=batch_index,
            )

        return agents

    @jaxtyped(typechecker=beartype)
    def _get_random_spawn_position(
        self,
        PRNG_key: PRNGKeyArray,
        blue: bool,
        env_index: Int[Array, f"{env_index_dim}"] | None = None,
    ):
        PRNG_key, subkey = jax.random.split(PRNG_key)
        return jax.random.uniform(
            subkey,
            (
                (1, self.world.dim_p)
                if env_index is None
                else (self.world.batch_dim, self.world.dim_p)
            ),
        ) * self._reset_agent_range + (
            self._reset_agent_offset_blue if blue else self._reset_agent_offset_red
        )

    @jaxtyped(typechecker=beartype)
    def get_closest_agent_to_ball(
        self,
        team: list[Agent],
        env_index: Int[Array, f"{env_index_dim}"] | None = None,
    ):
        pos = jnp.stack(
            [a.state.pos for a in team], axis=-2
        )  # shape == (batch_dim, n_agents, 2)
        ball_pos = self.ball.state.pos[..., None, :]
        if env_index is not None:
            pos = pos[env_index][None]
            ball_pos = ball_pos[env_index][None]
        dist = jnp.linalg.norm(
            pos[:, :, None, :] - ball_pos[:, None, :, :], axis=-1
        )  # -> (B, n, m)
        dist = dist.squeeze(axis=-1)
        min_dist = dist.min(axis=-1)
        if env_index is not None:
            min_dist = min_dist.squeeze(0)

        return min_dist

    def render_field(self, render: bool):
        _render_field = render
        left_top_wall = self.left_top_wall.is_rendering.at[:].set(render)
        left_bottom_wall = self.left_bottom_wall.is_rendering.at[:].set(render)
        right_top_wall = self.right_top_wall.is_rendering.at[:].set(render)
        right_bottom_wall = self.right_bottom_wall.is_rendering.at[:].set(render)

        self = self.replace(
            _render_field=_render_field,
            left_top_wall=left_top_wall,
            left_bottom_wall=left_bottom_wall,
            right_top_wall=right_top_wall,
            right_bottom_wall=right_bottom_wall,
        )
        return self

    @eqx.filter_jit
    @jaxtyped(typechecker=beartype)
    def process_action(self, agent: "FootballAgent"):
        print("starting process_action")
        assert (
            self.reset == True
        ), "Please reset the environment before processing actions"
        if agent is self.ball:
            return self, agent
        blue = agent.id in [a.id for a in self.blue_agents]
        if agent.action_script is None and not blue:  # Non AI
            u_x = -agent.action.u[..., X]  # Red agents have the action X flipped
            u = agent.action.u.at[..., X].set(u_x)
            agent = agent.replace(action=agent.action.replace(u=u))
            if self.enable_shooting:
                u_rot = -agent.action.u[
                    ..., 2
                ]  # Red agents have the action rotation flipped
                u = agent.action.u.at[..., 2].set(u_rot)
                agent = agent.replace(action=agent.action.replace(u=u))

        # You can shoot the ball only if you hae that action, are the closest to the ball, and the ball is within range and angle
        if self.enable_shooting and not agent.is_scripted_agent:
            agent_index = [
                i for i, a in enumerate(self.world.agents) if a.id == agent.id
            ][0]
            rel_pos = self._agents_rel_pos_to_ball[:, agent_index]
            ball_within_range = (
                self._agent_dist_to_ball[:, agent_index] <= self.shooting_radius
            )
            agent = agent.replace(ball_within_range=ball_within_range)

            rel_pos_angle = jnp.arctan2(rel_pos[Y], rel_pos[X])
            a = (agent.state.rot.squeeze(-1) - rel_pos_angle + jnp.pi) % (
                2 * jnp.pi
            ) - jnp.pi
            ball_within_angle = (-self.shooting_angle / 2 <= a) * (
                a <= self.shooting_angle / 2
            )
            agent = agent.replace(ball_within_angle=ball_within_angle)

            shoot_force = jnp.zeros((self.world.batch_dim, 2))
            _shoot_force = agent.action.u[..., -1] + self.u_shoot_multiplier
            shoot_force = shoot_force.at[..., X].set(_shoot_force)
            shoot_force = JaxUtils.rotate_vector(shoot_force, agent.state.rot)
            agent = agent.replace(shoot_force=shoot_force)
            shoot_force = jnp.where(
                (
                    agent.ball_within_angle
                    * agent.ball_within_range
                    * self._agents_closest_to_ball[:, agent_index]
                )[..., None],
                shoot_force,
                0.0,
            )

            kicking_action = self.ball.kicking_action + shoot_force
            self = self.replace(ball=self.ball.replace(kicking_action=kicking_action))
            u = agent.action.u[:, :-1]
            agent = agent.replace(action=agent.action.replace(u=u))
        print("ending process_action")
        return self, agent

    @eqx.filter_jit
    @jaxtyped(typechecker=beartype)
    def pre_step(self):
        print("starting pre_step")
        if self.enable_shooting:
            agents_exclude_ball = [a for a in self.world.agents if a is not self.ball]

            _agents_rel_pos_to_ball = jnp.stack(
                [self.ball.state.pos - a.state.pos for a in agents_exclude_ball],
                axis=1,
            )
            _agent_dist_to_ball = jnp.linalg.norm(_agents_rel_pos_to_ball, axis=-1)
            _agents_closest_to_ball = _agent_dist_to_ball == _agent_dist_to_ball.min(
                axis=-1, keepdims=True
            )
            self = self.replace(
                _agents_rel_pos_to_ball=_agents_rel_pos_to_ball,
                _agent_dist_to_ball=_agent_dist_to_ball,
                _agents_closest_to_ball=_agents_closest_to_ball,
            )

            u = self.ball.action.u + self.ball.kicking_action
            self.ball = self.ball.replace(action=self.ball.action.replace(u=u))
            kicking_action = self.ball.kicking_action.at[:].set(0)
            self = self.replace(ball=self.ball.replace(kicking_action=kicking_action))

        print("ending pre_step")
        return self

    @eqx.filter_jit
    @jaxtyped(typechecker=beartype)
    def reward(self, agent: "FootballAgent"):
        print("starting reward")
        # Called with agent=None when only AIs are playing to compute the _done
        if agent is None or agent.id == self.world.agents[0].id:
            # Sparse Reward
            over_right_line = (
                self.ball.state.pos[:, X] > self.pitch_length / 2 + self.ball_size / 2
            )
            over_left_line = (
                self.ball.state.pos[:, X] < -self.pitch_length / 2 - self.ball_size / 2
            )
            goal_mask = (self.ball.state.pos[:, Y] <= self.goal_size / 2) * (
                self.ball.state.pos[:, Y] >= -self.goal_size / 2
            )
            blue_score = over_right_line * goal_mask
            red_score = over_left_line * goal_mask
            _sparse_reward_blue = (
                self.scoring_reward * blue_score - self.scoring_reward * red_score
            )
            _sparse_reward_red = -_sparse_reward_blue

            _done = blue_score | red_score
            # Dense Reward
            _dense_reward_blue = 0
            _dense_reward_red = 0
            self = self.replace(
                _sparse_reward_blue=_sparse_reward_blue,
                _sparse_reward_red=_sparse_reward_red,
                _done=_done,
            )
            if self.dense_reward and agent is not None:
                if not self.ai_blue_agents:
                    _dense_reward_blue = self.reward_ball_to_goal(
                        blue=True
                    ) + self.reward_all_agent_to_ball(blue=True)
                if not self.ai_red_agents:
                    _dense_reward_red = self.reward_ball_to_goal(
                        blue=False
                    ) + self.reward_all_agent_to_ball(blue=False)
            self = self.replace(
                _dense_reward_blue=_dense_reward_blue,
                _dense_reward_red=_dense_reward_red,
            )

        blue = agent.id in [a.id for a in self.blue_agents]
        if blue:
            reward = self._sparse_reward_blue + self._dense_reward_blue
        else:
            reward = self._sparse_reward_red + self._dense_reward_red
        print("ending reward")
        return reward

    @jaxtyped(typechecker=beartype)
    def reward_ball_to_goal(self, blue: bool):
        if blue:
            distance_to_goal_blue = jnp.linalg.norm(
                self.ball.state.pos - self.right_goal_pos,
                axis=-1,
            )
            distance_to_goal = distance_to_goal_blue
            # self = self.replace(
            #     ball=self.ball.replace(distance_to_goal_blue=distance_to_goal_blue)
            # )
        else:
            distance_to_goal_red = jnp.linalg.norm(
                self.ball.state.pos - self.left_goal_pos,
                axis=-1,
            )
            distance_to_goal = distance_to_goal_red
            # self = self.replace(
            #     ball=self.ball.replace(distance_to_goal_red=distance_to_goal_red)
            # )

        pos_shaping = distance_to_goal * self.pos_shaping_factor_ball_goal

        if blue:
            pos_rew_blue = self.ball.pos_shaping_blue - pos_shaping
            # self = self.replace(
            #     ball=self.ball.replace(
            #         pos_shaping_blue=pos_shaping, pos_rew_blue=pos_rew_blue
            #     )
            # )
            pos_rew = pos_rew_blue
        else:
            pos_rew_red = self.ball.pos_shaping_red - pos_shaping
            # self = self.replace(
            #     ball=self.ball.replace(
            #         pos_shaping_red=pos_shaping, pos_rew_red=pos_rew_red
            #     )
            # )
            pos_rew = pos_rew_red
        return pos_rew

    @jaxtyped(typechecker=beartype)
    def reward_all_agent_to_ball(self, blue: bool):
        min_dist_to_ball = self.get_closest_agent_to_ball(
            team=self.blue_agents if blue else self.red_agents, env_index=None
        )
        if blue:
            self = self.replace(min_agent_dist_to_ball_blue=min_dist_to_ball)
        else:
            self = self.replace(min_agent_dist_to_ball_red=min_dist_to_ball)
        pos_shaping = min_dist_to_ball * self.pos_shaping_factor_agent_ball

        ball_moving = jnp.linalg.norm(self.ball.state.vel, axis=-1) > 1e-6
        agent_close_to_goal = min_dist_to_ball < self.distance_to_ball_trigger

        if blue:
            self = self.replace(
                ball=self.ball.replace(
                    pos_rew_agent_blue=jnp.where(
                        agent_close_to_goal + ball_moving,
                        0.0,
                        self.ball.pos_shaping_agent_blue - pos_shaping,
                    ),
                    pos_shaping_agent_blue=pos_shaping,
                )
            )
            pos_rew_agent = self.ball.pos_rew_agent_blue
        else:
            self = self.replace(
                ball=self.ball.replace(
                    pos_rew_agent_red=jnp.where(
                        agent_close_to_goal + ball_moving,
                        0.0,
                        self.ball.pos_shaping_agent_red - pos_shaping,
                    ),
                    pos_shaping_agent_red=pos_shaping,
                )
            )
            pos_rew_agent = self.ball.pos_rew_agent_red

        return pos_rew_agent

    @eqx.filter_jit
    @jaxtyped(typechecker=beartype)
    def observation(
        self,
        agent: Agent,
        agent_pos=None,
        agent_rot=None,
        agent_vel=None,
        agent_force=None,
        teammate_poses=None,
        teammate_forces=None,
        teammate_vels=None,
        adversary_poses=None,
        adversary_forces=None,
        adversary_vels=None,
        ball_pos=None,
        ball_vel=None,
        ball_force=None,
        blue=None,
        env_index=Ellipsis,
    ):
        print("starting observation")
        if blue:
            assert agent.id in [a.id for a in self.blue_agents]
        else:
            blue = agent.id in [a.id for a in self.blue_agents]

        if not blue:
            my_team, other_team = (self.red_agents, self.blue_agents)
            goal_pos = self.left_goal_pos
        else:
            my_team, other_team = (self.blue_agents, self.red_agents)
            goal_pos = self.right_goal_pos

        actual_adversary_poses = []
        actual_adversary_forces = []
        actual_adversary_vels = []
        if self.observe_adversaries:
            for a in other_team:
                actual_adversary_poses.append(a.state.pos[env_index])
                actual_adversary_vels.append(a.state.vel[env_index])
                actual_adversary_forces.append(a.state.force[env_index])

        actual_teammate_poses = []
        actual_teammate_forces = []
        actual_teammate_vels = []
        if self.observe_teammates:
            for a in my_team:
                if a.id != agent.id:
                    actual_teammate_poses.append(a.state.pos[env_index])
                    actual_teammate_vels.append(a.state.vel[env_index])
                    actual_teammate_forces.append(a.state.force[env_index])

        obs = self.observation_base(
            agent.state.pos[env_index] if agent_pos is None else agent_pos,
            agent.state.rot[env_index] if agent_rot is None else agent_rot,
            agent.state.vel[env_index] if agent_vel is None else agent_vel,
            agent.state.force[env_index] if agent_force is None else agent_force,
            goal_pos=goal_pos,
            ball_pos=self.ball.state.pos[env_index] if ball_pos is None else ball_pos,
            ball_vel=self.ball.state.vel[env_index] if ball_vel is None else ball_vel,
            ball_force=(
                self.ball.state.force[env_index] if ball_force is None else ball_force
            ),
            adversary_poses=(
                actual_adversary_poses if adversary_poses is None else adversary_poses
            ),
            adversary_forces=(
                actual_adversary_forces
                if adversary_forces is None
                else adversary_forces
            ),
            adversary_vels=(
                actual_adversary_vels if adversary_vels is None else adversary_vels
            ),
            teammate_poses=(
                actual_teammate_poses if teammate_poses is None else teammate_poses
            ),
            teammate_forces=(
                actual_teammate_forces if teammate_forces is None else teammate_forces
            ),
            teammate_vels=(
                actual_teammate_vels if teammate_vels is None else teammate_vels
            ),
            blue=blue,
        )
        print("ending observation")
        return obs

    @jaxtyped(typechecker=beartype)
    def observation_base(
        self,
        agent_pos: Array,
        agent_rot: Array,
        agent_vel: Array,
        agent_force: Array,
        teammate_poses: Array,
        teammate_forces: Array,
        teammate_vels: Array,
        adversary_poses: Array,
        adversary_forces: Array,
        adversary_vels: Array,
        ball_pos: Array,
        ball_vel: Array,
        ball_force: Array,
        goal_pos: Array,
        blue: bool,
    ):
        # Make all inputs same batch size (this is needed when this function is called for rendering
        input = [
            agent_pos,
            agent_rot,
            agent_vel,
            agent_force,
            ball_pos,
            ball_vel,
            ball_force,
            goal_pos,
            teammate_poses,
            teammate_forces,
            teammate_vels,
            adversary_poses,
            adversary_forces,
            adversary_vels,
        ]
        for o in input:
            if isinstance(o, Array) and len(o.shape) > 1:
                batch_dim = o.shape[0]
                break
        for j in range(len(input)):
            if isinstance(input[j], Array):
                if len(input[j].shape) == 1:
                    input[j] = jnp.broadcast_to(
                        input[j][None], (batch_dim, *input[j].shape)
                    )
                input[j] = input[j].copy()

            else:
                o = input[j]
                for i in range(len(o)):
                    if len(o[i].shape) == 1:
                        o[i] = jnp.broadcast_to(o[i][None], (batch_dim, *o[i].shape))
                    o[i] = o[i].copy()

        (
            agent_pos,
            agent_rot,
            agent_vel,
            agent_force,
            ball_pos,
            ball_vel,
            ball_force,
            goal_pos,
            teammate_poses,
            teammate_forces,
            teammate_vels,
            adversary_poses,
            adversary_forces,
            adversary_vels,
        ) = input
        #  End rendering code

        if (
            not blue
        ):  # If agent is red we have to flip the x of sign of each observation
            [
                agent_pos,
                agent_vel,
                agent_force,
                ball_pos,
                ball_vel,
                ball_force,
                goal_pos,
                teammate_poses,
                teammate_forces,
                teammate_vels,
                adversary_poses,
                adversary_forces,
                adversary_vels,
            ] = jax.tree.map(
                lambda jax_array: jax_array.at[..., X].set(-jax_array[..., X]),
                [
                    agent_pos,
                    agent_vel,
                    agent_force,
                    ball_pos,
                    ball_vel,
                    ball_force,
                    goal_pos,
                    teammate_poses,
                    teammate_forces,
                    teammate_vels,
                    adversary_poses,
                    adversary_forces,
                    adversary_vels,
                ],
            )

            agent_rot = agent_rot - jnp.pi
        obs = {
            "obs": [
                agent_force,
                agent_pos - ball_pos,
                agent_vel - ball_vel,
                ball_pos - goal_pos,
                ball_vel,
                ball_force,
            ],
            "pos": [agent_pos - goal_pos],
            "vel": [agent_vel],
        }
        if self.enable_shooting:
            obs["obs"].append(agent_rot)

        if self.observe_adversaries and len(adversary_poses):
            obs["adversaries"] = []
            for adversary_pos, adversary_force, adversary_vel in zip(
                adversary_poses, adversary_forces, adversary_vels
            ):
                obs["adversaries"].append(
                    jnp.concatenate(
                        [
                            agent_pos - adversary_pos,
                            agent_vel - adversary_vel,
                            adversary_vel,
                            adversary_force,
                        ],
                        axis=-1,
                    )
                )
            obs["adversaries"] = [
                (
                    jnp.stack(obs["adversaries"], axis=-2)
                    if self.dict_obs
                    else jnp.concatenate(obs["adversaries"], axis=-1)
                )
            ]

        if self.observe_teammates:
            obs["teammates"] = []
            for teammate_pos, teammate_force, teammate_vel in zip(
                teammate_poses, teammate_forces, teammate_vels
            ):
                obs["teammates"].append(
                    jnp.concatenate(
                        [
                            agent_pos - teammate_pos,
                            agent_vel - teammate_vel,
                            teammate_vel,
                            teammate_force,
                        ],
                        axis=-1,
                    )
                )
            obs["teammates"] = [
                (
                    jnp.stack(obs["teammates"], axis=-2)
                    if self.dict_obs
                    else jnp.concatenate(obs["teammates"], axis=-1)
                )
            ]

        for key, value in obs.items():
            obs[key] = jnp.concatenate(value, axis=-1)
        if self.dict_obs:
            return obs
        else:
            return jnp.concatenate(list(obs.values()), axis=-1)

    @eqx.filter_jit
    @jaxtyped(typechecker=beartype)
    def done(self):
        print("starting done")
        if self.ai_blue_agents and self.ai_red_agents:
            self.reward(None)
        print("ending done")
        return self._done

    @jaxtyped(typechecker=beartype)
    def _compute_coverage(self, blue: bool, env_index=None):
        team = self.blue_agents if blue else self.red_agents
        pos = jnp.stack(
            [a.state.pos for a in team], axis=-2
        )  # shape == (batch_dim, n_agents, 2)
        avg_point = pos.mean(-2)[..., None, :]
        if isinstance(env_index, int):
            pos = pos[env_index][None]
            avg_point = avg_point[env_index][None]
        dist = jnp.linalg.norm(pos - avg_point, axis=-1)
        dist = dist.squeeze(-1)
        max_dist = dist.max(axis=-1)[0]
        if isinstance(env_index, int):
            max_dist = max_dist.squeeze(0)
        return max_dist

    @eqx.filter_jit
    @jaxtyped(typechecker=beartype)
    def info(self, agent: Agent):
        print("starting info")

        blue = agent.id in [a.id for a in self.blue_agents]
        info = {
            "sparse_reward": (
                self._sparse_reward_blue if blue else self._sparse_reward_red
            ),
            "ball_goal_pos_rew": (
                self.ball.pos_rew_blue if blue else self.ball.pos_rew_red
            ),
            "all_agent_ball_pos_rew": (
                self.ball.pos_rew_agent_blue if blue else self.ball.pos_rew_agent_red
            ),
            "ball_pos": self.ball.state.pos,
            "dist_ball_to_goal": (
                self.ball.pos_shaping_blue if blue else self.ball.pos_shaping_red
            )
            / self.pos_shaping_factor_ball_goal,
        }
        if blue and self.min_agent_dist_to_ball_blue is not None:
            info["min_agent_dist_to_ball"] = self.min_agent_dist_to_ball_blue
            info["touching_ball"] = (
                self.min_agent_dist_to_ball_blue
                <= self.agent_size + self.ball_size + 1e-2
            )
        elif not blue and self.min_agent_dist_to_ball_red is not None:
            info["min_agent_dist_to_ball"] = self.min_agent_dist_to_ball_red
            info["touching_ball"] = (
                self.min_agent_dist_to_ball_red
                <= self.agent_size + self.ball_size + 1e-2
            )
        print("ending info")
        return info

    @jaxtyped(typechecker=beartype)
    def extra_render(self, env_index: int = 0) -> "list[Geom]":
        from jaxvmas.simulator import rendering

        # Background
        # You can disable background rendering in case you are plotting the a function on the field
        geoms: list[Geom] = (
            self._get_background_geoms(self.background_entities)
            if self._render_field
            else self._get_background_geoms(self.background_entities[3:])
        )

        geoms += ScenarioUtils.render_agent_indices(
            self, env_index, start_from=1, exclude=self.red_agents + [self.ball]
        )

        # Agent rotation and shooting
        if self.enable_shooting:
            for agent in self.blue_agents:
                color = agent.color
                if (
                    agent.ball_within_angle[env_index]
                    and agent.ball_within_range[env_index]
                ):
                    color = Color.PINK.value
                sector = rendering.make_circle(
                    radius=self.shooting_radius, angle=self.shooting_angle, filled=True
                )
                xform = rendering.Transform()
                xform.set_rotation(agent.state.rot[env_index])
                xform.set_translation(*agent.state.pos[env_index])
                sector.add_attr(xform)
                sector.set_color(*color, alpha=agent._alpha / 2)
                geoms.append(sector)

                shoot_intensity = jnp.linalg.norm(agent.shoot_force[env_index]) / (
                    self.u_shoot_multiplier * 2
                )
                l, r, t, b = (
                    0,
                    self.shooting_radius * shoot_intensity,
                    self.agent_size / 2,
                    -self.agent_size / 2,
                )
                line = rendering.make_polygon([(l, b), (l, t), (r, t), (r, b)])
                xform = rendering.Transform()
                xform.set_rotation(agent.state.rot[env_index])
                xform.set_translation(*agent.state.pos[env_index])
                line.add_attr(xform)
                line.set_color(*color, alpha=agent._alpha)
                geoms.append(line)

        return geoms

    @jaxtyped(typechecker=beartype)
    def _get_background_geoms(self, landmarks: list[Entity]) -> list[Geom]:
        def _get_geom(entity: Entity, pos: Array, rot: Array = 0.0):
            from jaxvmas.simulator import rendering

            geom = entity.shape.get_geometry()
            xform = rendering.Transform()
            geom.add_attr(xform)
            xform.set_translation(*pos)
            xform.set_rotation(rot)
            color = entity.color.value
            geom.set_color(*color)
            return geom

        geoms: list[Geom] = []
        for landmark in landmarks:
            if landmark.name == "Centre Line":
                geoms.append(_get_geom(landmark, [0.0, 0.0], jnp.pi / 2))
            elif landmark.name == "Right Line":
                geoms.append(
                    _get_geom(
                        landmark,
                        [self.pitch_length / 2 - self.agent_size, 0.0],
                        jnp.pi / 2,
                    )
                )
            elif landmark.name == "Left Line":
                geoms.append(
                    _get_geom(
                        landmark,
                        [-self.pitch_length / 2 + self.agent_size, 0.0],
                        jnp.pi / 2,
                    )
                )
            elif landmark.name == "Top Line":
                geoms.append(
                    _get_geom(landmark, [0.0, self.pitch_width / 2 - self.agent_size])
                )
            elif landmark.name == "Bottom Line":
                geoms.append(
                    _get_geom(landmark, [0.0, -self.pitch_width / 2 + self.agent_size])
                )
            else:
                geoms.append(_get_geom(landmark, [0, 0]))
        return geoms


# Ball Physics


@jaxtyped(typechecker=beartype)
def get_ball_action_script(
    agent_size: float, pitch_width: float, pitch_length: float, goal_size: float
):
    def ball_action_script(
        PRNG_key: PRNGKeyArray, ball: "BallAgent", world: FootballWorld
    ):
        # Avoid getting stuck against the wall
        dist_thres = agent_size * 2
        vel_thres = 0.3
        impulse = 0.05
        upper = (
            1
            - jnp.minimum(
                pitch_width / 2 - ball.state.pos[:, 1],
                jnp.array(dist_thres),
            )
            / dist_thres
        )
        lower = (
            1
            - jnp.minimum(
                pitch_width / 2 + ball.state.pos[:, 1],
                jnp.array(dist_thres),
            )
            / dist_thres
        )
        right = (
            1
            - jnp.minimum(
                pitch_length / 2 - ball.state.pos[:, 0],
                jnp.array(dist_thres),
            )
            / dist_thres
        )
        left = (
            1
            - jnp.minimum(
                pitch_length / 2 + ball.state.pos[:, 0],
                jnp.array(dist_thres),
            )
            / dist_thres
        )
        vertical_vel = (
            1
            - jnp.minimum(
                jnp.abs(ball.state.vel[:, 1]),
                jnp.array(vel_thres),
            )
            / vel_thres
        )
        horizontal_vel = (
            1
            - jnp.minimum(
                jnp.abs(ball.state.vel[:, 1]),
                jnp.array(vel_thres),
            )
            / vel_thres
        )
        dist_action = jnp.stack([left - right, lower - upper], axis=1)
        vel_action = jnp.stack([horizontal_vel, vertical_vel], axis=1)
        actions = dist_action * vel_action * impulse
        goal_mask = (ball.state.pos[:, 1] < goal_size / 2) * (
            ball.state.pos[:, 1] > -goal_size / 2
        )

        _actions = actions[:, 0]
        _actions = jnp.where(goal_mask, jnp.zeros_like(_actions), _actions)
        actions = actions.at[:, 0].set(_actions)

        ball = ball.replace(
            action=ball.action.replace(
                u=actions,
            ),
        )

        return ball, world

    return ball_action_script


@jaxtyped(typechecker=beartype)
class FootballAgent(Agent):
    ball_within_angle: Array | None
    ball_within_range: Array | None
    shoot_force: Array | None

    @jaxtyped(typechecker=beartype)
    @classmethod
    def create(cls, **kwargs):
        base_agent = Agent.create(**kwargs)
        return cls(
            **dataclass_to_dict_first_layer(base_agent),
            ball_within_angle=None,
            ball_within_range=None,
            shoot_force=None,
        )

    @jaxtyped(typechecker=beartype)
    def _spawn(self, id: int, batch_dim: int, dim_c: int, dim_p: int):
        self = self.replace(
            ball_within_angle=jnp.zeros(batch_dim, dtype=jnp.bool_),
            ball_within_range=jnp.zeros(batch_dim, dtype=jnp.bool_),
            shoot_force=jnp.zeros((batch_dim, 2), dtype=jnp.float32),
        )
        return super(FootballAgent, self)._spawn(
            id=id, batch_dim=batch_dim, dim_c=dim_c, dim_p=dim_p
        )


@jaxtyped(typechecker=beartype)
class BallAgent(Agent):
    pos_rew_blue: Array | None
    pos_rew_red: Array | None
    pos_rew_agent_blue: Array | None
    pos_rew_agent_red: Array | None
    kicking_action: Array | None
    pos_shaping_blue: Array | None
    pos_shaping_red: Array | None
    pos_shaping_agent_blue: Array | None
    pos_shaping_agent_red: Array | None
    # distance_to_goal_blue: Array | None
    # distance_to_goal_red: Array | None

    @classmethod
    @jaxtyped(typechecker=beartype)
    def create(cls, **kwargs):
        base_agent = Agent.create(**kwargs)
        return cls(
            **dataclass_to_dict_first_layer(base_agent),
            pos_rew_blue=None,
            pos_rew_red=None,
            pos_rew_agent_blue=None,
            pos_rew_agent_red=None,
            kicking_action=None,
            pos_shaping_blue=None,
            pos_shaping_red=None,
            pos_shaping_agent_blue=None,
            pos_shaping_agent_red=None,
            # distance_to_goal_blue=None,
            # distance_to_goal_red=None,
        )

    @jaxtyped(typechecker=beartype)
    def _spawn(self, id: int, batch_dim: int, dim_c: int, dim_p: int):
        self = self.replace(
            pos_rew_blue=jnp.zeros(batch_dim),
            pos_rew_red=jnp.zeros(batch_dim),
            pos_rew_agent_blue=jnp.zeros(batch_dim),
            pos_rew_agent_red=jnp.zeros(batch_dim),
            kicking_action=jnp.zeros((batch_dim, dim_p)),
            pos_shaping_blue=jnp.zeros(batch_dim),
            pos_shaping_red=jnp.zeros(batch_dim),
            pos_shaping_agent_blue=jnp.zeros(batch_dim),
            pos_shaping_agent_red=jnp.zeros(batch_dim),
            # distance_to_goal_blue=jnp.zeros(batch_dim),
            # distance_to_goal_red=jnp.zeros(batch_dim),
        )
        return super(BallAgent, self)._spawn(
            id=id, batch_dim=batch_dim, dim_c=dim_c, dim_p=dim_p
        )


# Helper Functions


# Run
if __name__ == "__main__":
    render_interactively(
        __file__,
        control_two_agents=True,
        n_blue_agents=5,
        n_red_agents=5,
        ai_blue_agents=False,
        ai_red_agents=True,
        ai_strength=1.0,
        ai_decision_strength=1.0,
        ai_precision_strength=1.0,
        n_traj_points=8,
    )
