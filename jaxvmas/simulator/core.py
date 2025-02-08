#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Callable, Sequence

import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxvmas.equinox_utils import PyTreeNode
from jaxvmas.simulator.dynamics.common import Dynamics
from jaxvmas.simulator.dynamics.holonomic import Holonomic
from jaxvmas.simulator.joints import Joint, JointConstraint
from jaxvmas.simulator.physics import (
    _get_closest_box_box,
    _get_closest_line_box,
    _get_closest_point_box,
    _get_closest_point_line,
    _get_closest_points_line_line,
    _get_inner_point_box,
)
from jaxvmas.simulator.rendering import Geom
from jaxvmas.simulator.sensors import Sensor
from jaxvmas.simulator.utils import (
    ANGULAR_FRICTION,
    COLLISION_FORCE,
    DRAG,
    JOINT_FORCE,
    LINE_MIN_DIST,
    LINEAR_FRICTION,
    TORQUE_CONSTRAINT_FORCE,
    Color,
    JaxUtils,
    Observable,
    X,
    Y,
)

# Dimension type variables (add near top of file)
batch_dim = "batch"
pos_dim = "dim_p"
comm_dim = "dim_c"
action_size_dim = "action_size"
angles_dim = "angles"
boxes_dim = "boxes"
spheres_dim = "spheres"
lines_dim = "lines"
dots_dim = "..."


class JaxVectorizedObject(PyTreeNode):
    batch_dim: int

    @classmethod
    def create(cls, batch_dim: int):
        return cls(batch_dim)

    def _check_batch_index(self, batch_index: int):
        if batch_index is not None:
            assert (
                0 <= batch_index < self.batch_dim
            ), f"Index must be between 0 and {self.batch_dim}, got {batch_index}"


class Shape(ABC):
    @abstractmethod
    def moment_of_inertia(self, mass: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_delta_from_anchor(self, anchor: tuple[float, float]) -> tuple[float, float]:
        raise NotImplementedError

    @abstractmethod
    def get_geometry(self):
        raise NotImplementedError

    @abstractmethod
    def circumscribed_radius(self) -> float:
        raise NotImplementedError


class Box(Shape):
    def __init__(self, length: float = 0.3, width: float = 0.1, hollow: bool = False):
        super().__init__()
        assert length > 0, f"Length must be > 0, got {length}"
        assert width > 0, f"Width must be > 0, got {length}"
        self._length = length
        self._width = width
        self.hollow = hollow

    @property
    def length(self):
        return self._length

    @property
    def width(self):
        return self._width

    def get_delta_from_anchor(self, anchor: tuple[float, float]) -> tuple[float, float]:
        return anchor[X] * self.length / 2, anchor[Y] * self.width / 2

    def moment_of_inertia(self, mass: float):
        return (1 / 12) * mass * (self.length**2 + self.width**2)

    def circumscribed_radius(self):
        return jnp.sqrt((self.length / 2) ** 2 + (self.width / 2) ** 2)

    def get_geometry(self) -> "Geom":
        from jaxvmas.simulator import rendering

        l, r, t, b = (
            -self.length / 2,
            self.length / 2,
            self.width / 2,
            -self.width / 2,
        )
        return rendering.make_polygon([(l, b), (l, t), (r, t), (r, b)])


class Sphere(Shape):
    def __init__(self, radius: float = 0.05):
        super().__init__()
        assert radius > 0, f"Radius must be > 0, got {radius}"
        self._radius = radius

    @property
    def radius(self):
        return self._radius

    def get_delta_from_anchor(self, anchor: tuple[float, float]) -> tuple[float, float]:
        delta = jnp.array([anchor[X] * self.radius, anchor[Y] * self.radius])
        delta_norm = jnp.linalg.vector_norm(delta)
        if delta_norm > self.radius:
            delta /= delta_norm * self.radius
        return tuple(delta.tolist())

    def moment_of_inertia(self, mass: float):
        return (1 / 2) * mass * self.radius**2

    def circumscribed_radius(self):
        return self.radius

    def get_geometry(self) -> "Geom":
        from jaxvmas.simulator import rendering

        return rendering.make_circle(self.radius)


class Line(Shape):
    def __init__(self, length: float = 0.5):
        super().__init__()
        assert length > 0, f"Length must be > 0, got {length}"
        self._length = length
        self._width = 2

    @property
    def length(self):
        return self._length

    @property
    def width(self):
        return self._width

    def moment_of_inertia(self, mass: float) -> float:
        return (1 / 12) * mass * (self.length**2)

    def circumscribed_radius(self) -> float:
        return self.length / 2

    def get_delta_from_anchor(self, anchor: tuple[float, float]) -> tuple[float, float]:
        return anchor[X] * self.length / 2, 0.0

    def get_geometry(self) -> "Geom":
        from jaxvmas.simulator import rendering

        return rendering.Line(
            (-self.length / 2, 0),
            (self.length / 2, 0),
            width=self.width,
        )


class EntityState(JaxVectorizedObject):
    pos: Float[Array, f"{batch_dim} {pos_dim}"]
    vel: Float[Array, f"{batch_dim} {pos_dim}"]
    rot: Float[Array, f"{batch_dim} 1"]
    ang_vel: Float[Array, f"{batch_dim} 1"]

    @classmethod
    def create(cls, batch_dim: int, dim_c: int, dim_p: int):
        # physical position
        pos = jnp.zeros((batch_dim, dim_p))
        # physical velocity
        vel = jnp.zeros((batch_dim, dim_p))
        # physical rotation -- from -pi to pi
        rot = jnp.zeros((batch_dim, 1))
        # angular velocity
        ang_vel = jnp.zeros((batch_dim, 1))
        return cls(batch_dim, pos, vel, rot, ang_vel)

    def _reset(self, env_index: int | None = None) -> "EntityState":
        if env_index is None:
            return self.replace(
                pos=jnp.zeros_like(self.pos),
                vel=jnp.zeros_like(self.vel),
                rot=jnp.zeros_like(self.rot),
                ang_vel=jnp.zeros_like(self.ang_vel),
            )
        return self.replace(
            pos=JaxUtils.where_from_index(
                env_index, jnp.zeros_like(self.pos), self.pos
            ),
            vel=JaxUtils.where_from_index(
                env_index, jnp.zeros_like(self.vel), self.vel
            ),
            rot=JaxUtils.where_from_index(
                env_index, jnp.zeros_like(self.rot), self.rot
            ),
            ang_vel=JaxUtils.where_from_index(
                env_index, jnp.zeros_like(self.ang_vel), self.ang_vel
            ),
        )

    # Resets state for all entities
    def _spawn(self, dim_c: int, dim_p: int) -> "EntityState":
        return self.replace(
            pos=jnp.zeros((self.batch_dim, dim_p)),
            vel=jnp.zeros((self.batch_dim, dim_p)),
            rot=jnp.zeros((self.batch_dim, 1)),
            ang_vel=jnp.zeros((self.batch_dim, 1)),
        )


class AgentState(EntityState):
    c: Float[Array, f"{batch_dim} {comm_dim}"]
    force: Float[Array, f"{batch_dim} {pos_dim}"]
    torque: Float[Array, f"{batch_dim} 1"]

    @classmethod
    def create(cls, batch_dim: int, dim_c: int, dim_p: int):
        entity_state = EntityState.create(batch_dim, dim_c, dim_p)
        # communication utterance
        c = jnp.zeros((batch_dim, dim_c))
        # Agent force from actions
        force = jnp.zeros((batch_dim, dim_p))
        # Agent torque from actions
        torque = jnp.zeros((batch_dim, 1))
        return cls(
            batch_dim,
            entity_state.pos,
            entity_state.vel,
            entity_state.rot,
            entity_state.ang_vel,
            c,
            force,
            torque,
        )

    def _reset(self, env_index: int | None = None) -> "AgentState":
        if env_index is None:
            return self.replace(
                c=jnp.zeros_like(self.c),
                force=jnp.zeros_like(self.force),
                torque=jnp.zeros_like(self.torque),
            )
        return self.replace(
            c=JaxUtils.where_from_index(env_index, jnp.zeros_like(self.c), self.c),
            force=JaxUtils.where_from_index(
                env_index, jnp.zeros_like(self.force), self.force
            ),
            torque=JaxUtils.where_from_index(
                env_index, jnp.zeros_like(self.torque), self.torque
            ),
        )

    def _spawn(self, dim_c: int, dim_p: int) -> "AgentState":
        self = self.replace(c=jnp.zeros((self.batch_dim, dim_c)))
        self = self.replace(
            force=jnp.zeros((self.batch_dim, dim_p)),
            torque=jnp.zeros((self.batch_dim, 1)),
        )
        return super(AgentState, self)._spawn(dim_c, dim_p)


class Action(JaxVectorizedObject):
    u: Float[Array, f"{batch_dim} {action_size_dim}"]
    c: Float[Array, f"{batch_dim} {comm_dim}"]

    u_range: float | Sequence[float]
    u_multiplier: float | Sequence[float]
    u_noise: float | Sequence[float]
    action_size: int

    _u_range_jax_array: Sequence[float] | None
    _u_multiplier_jax_array: Sequence[float] | None
    _u_noise_jax_array: Sequence[float] | None

    @classmethod
    def create(
        cls,
        batch_dim: int,
        action_size: int,
        comm_dim: int,
        u_range: float | Sequence[float],
        u_multiplier: float | Sequence[float],
        u_noise: float | Sequence[float],
    ):
        u = jnp.zeros((batch_dim, action_size))
        c = jnp.zeros((batch_dim, comm_dim))

        # control range
        _u_range = u_range
        # agent action is a force multiplied by this amount
        _u_multiplier = u_multiplier
        # physical motor noise amount
        _u_noise = u_noise
        # Number of actions
        action_size = action_size

        action = cls(
            batch_dim,
            u,
            c,
            _u_range,
            _u_multiplier,
            _u_noise,
            action_size,
            None,
            None,
            None,
        )
        action._check_action_init()
        return action

    def _check_action_init(self):
        for attr in (self.u_multiplier, self.u_range, self.u_noise):
            if isinstance(attr, list):
                assert len(attr) == self.action_size, (
                    "Action attributes u_... must be either a float or a list of floats"
                    " (one per action) all with same length"
                )

    @property
    def u_range_jax_array(self):
        ret = self._u_range_jax_array
        if ret is None:
            ret = self._to_jax_array(self.u_range)
        return ret

    @property
    def u_multiplier_jax_array(self):
        ret = self._u_multiplier_jax_array
        if ret is None:
            ret = self._to_jax_array(self.u_multiplier)
        return ret

    @property
    def u_noise_jax_array(self):
        ret = self._u_noise_jax_array
        if ret is None:
            ret = self._to_jax_array(self.u_noise)
        return ret

    def _to_jax_array(self, value):
        return jnp.array(
            value if isinstance(value, Sequence) else [value] * self.action_size,
        )

    def _reset(self, env_index: None | int):
        for attr_name in ["u", "c"]:
            attr = getattr(self, attr_name)
            if attr is not None:
                if env_index is None:
                    self = self.replace(**{attr_name: jnp.zeros_like(attr)})
                else:
                    self = self.replace(
                        **{
                            attr_name: JaxUtils.where_from_index(
                                env_index, jnp.zeros_like(attr), attr
                            )
                        }
                    )
        return self


class Entity(JaxVectorizedObject, Observable):
    state: EntityState

    gravity: Array

    name: str
    movable: bool
    rotatable: bool
    collide: bool
    density: float
    mass: float
    shape: Shape
    v_range: float | None
    max_speed: float | None
    color: Color
    is_joint: bool
    drag: float | None
    linear_friction: float | None
    angular_friction: float | None
    collision_filter: Callable[["Entity"], bool]

    goal: Array | None
    _render: Array

    @classmethod
    def create(
        cls,
        batch_dim: int,
        name: str,
        movable: bool = False,
        rotatable: bool = False,
        collide: bool = True,
        density: float = 25.0,  # Unused for now
        mass: float = 1.0,
        shape: Shape | None = None,
        v_range: float | None = None,
        max_speed: float | None = None,
        color=Color.GRAY,
        is_joint: bool = False,
        drag: float | None = None,
        linear_friction: float | None = None,
        angular_friction: float | None = None,
        gravity: float | Sequence[float] | None = None,
        collision_filter: Callable[["Entity"], bool] = lambda _: True,
        dim_p: int = 2,
        dim_c: int = 0,
    ):
        if shape is None:
            shape = Sphere()
        # name
        name = name
        # entity can move / be pushed
        movable = movable
        # entity can rotate
        rotatable = rotatable
        # entity collides with others
        collide = collide
        # material density (affects mass)
        density = density
        # mass
        mass = mass
        # max speed
        max_speed = max_speed
        v_range = v_range
        # color
        color = color
        # shape
        shape = shape
        # is joint
        is_joint = is_joint
        # collision filter
        collision_filter = collision_filter
        # drag
        drag = drag
        # friction
        linear_friction = linear_friction
        angular_friction = angular_friction
        # gravity
        if isinstance(gravity, Array):
            gravity = gravity
        else:
            gravity = (
                jnp.array(gravity) if gravity is not None else jnp.zeros((batch_dim, 1))
            )
        # entity goal
        goal = None
        # Render the entity
        _render = jnp.full((batch_dim,), True)

        state = EntityState.create(batch_dim, dim_c, dim_p)

        return cls(
            batch_dim,
            state,
            gravity,
            name,
            movable,
            rotatable,
            collide,
            density,
            mass,
            shape,
            v_range,
            max_speed,
            color,
            is_joint,
            drag,
            linear_friction,
            angular_friction,
            collision_filter,
            goal,
            _render,
        )

    @property
    def is_rendering(self):
        return self._render

    def reset_render(self):
        return self.replace(_render=jnp.full((self.batch_dim,), True))

    def collides(self, entity: "Entity"):
        if not self.collide:
            return False
        return self.collision_filter(entity)

    def _spawn(self, dim_c: int, dim_p: int) -> "Entity":
        return self.replace(state=self.state._spawn(dim_c, dim_p))

    def _reset(self, env_index: int):
        return self.replace(state=self.state._reset(env_index))

    def set_pos(self, pos: Array, batch_index: int):
        return self._set_state_property("pos", pos, batch_index)

    def set_vel(self, vel: Array, batch_index: int):
        return self._set_state_property("vel", vel, batch_index)

    def set_rot(self, rot: Array, batch_index: int):
        return self._set_state_property("rot", rot, batch_index)

    def set_ang_vel(self, ang_vel: Array, batch_index: int):
        return self._set_state_property("ang_vel", ang_vel, batch_index)

    def _set_state_property(self, prop_name: str, new: Array, batch_index: int | None):
        assert (
            self.batch_dim is not None
        ), f"Tried to set property of {self.name} without adding it to the world"
        self._check_batch_index(batch_index)
        new_entity = None
        if batch_index is None:
            if len(new.shape) > 1 and new.shape[0] == self.batch_dim:
                new_entity = self.state.replace(**{prop_name: new})
            else:
                new_entity = self.state.replace(
                    **{prop_name: new.repeat(self.batch_dim, 1)}
                )
        else:
            value = getattr(self.state, prop_name)
            new_entity = self.state.replace(
                **{prop_name: value.at[batch_index].set(new)}
            )
        # there was a notify_observers call in the past, so we need to notify again
        return self.replace(state=new_entity)

    def render(self, env_index: int = 0) -> "list[Geom]":
        from jaxvmas.simulator import rendering

        if not self.is_rendering[env_index]:
            return []
        geom = self.shape.get_geometry()
        xform = rendering.Transform()
        geom.add_attr(xform)

        xform.set_translation(*self.state.pos[env_index])
        xform.set_rotation(self.state.rot[env_index].item())

        color = self.color
        if isinstance(color, Array) and len(color.shape) > 1:
            color = color[env_index]
        geom.set_color(*color.value)

        return [geom]


# properties of landmark entities
class Landmark(Entity):
    def __init__(
        self,
        name: str,
        shape: Shape = None,
        movable: bool = False,
        rotatable: bool = False,
        collide: bool = True,
        density: float = 25.0,  # Unused for now
        mass: float = 1.0,
        v_range: float = None,
        max_speed: float = None,
        color=Color.GRAY,
        is_joint: bool = False,
        drag: float = None,
        linear_friction: float = None,
        angular_friction: float = None,
        gravity: float = None,
        collision_filter: Callable[[Entity], bool] = lambda _: True,
    ):
        super().__init__(
            name,
            movable,
            rotatable,
            collide,
            density,  # Unused for now
            mass,
            shape,
            v_range,
            max_speed,
            color,
            is_joint,
            drag,
            linear_friction,
            angular_friction,
            gravity,
            collision_filter,
        )


class Agent(Entity):
    state: AgentState
    action: Action

    obs_range: float | None
    obs_noise: float | None
    f_range: float | None
    max_f: float | None
    t_range: float | None
    max_t: float | None
    action_script: Callable[["Agent", Action, "World"], None]
    sensors: list[Sensor]
    c_noise: float
    silent: bool
    render_action: bool
    adversary: bool
    alpha: float

    dynamics: Dynamics
    action_size: int
    discrete_action_nvec: list[int]

    @classmethod
    def create(
        cls,
        batch_dim: int,
        name: str,
        dim_c: int,
        dim_p: int,
        shape: Shape | None = None,
        movable: bool = True,
        rotatable: bool = True,
        collide: bool = True,
        density: float = 25.0,  # Unused for now
        mass: float = 1.0,
        f_range: float = None,
        max_f: float = None,
        t_range: float = None,
        max_t: float = None,
        v_range: float = None,
        max_speed: float = None,
        color=Color.BLUE,
        alpha: float = 0.5,
        obs_range: float = None,
        obs_noise: float = None,
        u_noise: float | Sequence[float] = 0.0,
        u_range: float | Sequence[float] = 1.0,
        u_multiplier: float | Sequence[float] = 1.0,
        action_script: Callable[["Agent", Action, "World"], None] = None,
        sensors: list[Sensor] = None,
        c_noise: float = 0.0,
        silent: bool = True,
        adversary: bool = False,
        drag: float = None,
        linear_friction: float = None,
        angular_friction: float = None,
        gravity: float = None,
        collision_filter: Callable[[Entity], bool] = lambda _: True,
        render_action: bool = False,
        dynamics: Dynamics = None,  # Defaults to holonomic
        action_size: int | None = None,  # Defaults to what required by the dynamics
        discrete_action_nvec: (
            list[int] | None
        ) = None,  # Defaults to 3-way discretization if discrete actions are chosen (stay, decrement, increment)
    ):
        entity = Entity.create(
            batch_dim,
            name,
            movable,
            rotatable,
            collide,
            density,  # Unused for now
            mass,
            shape,
            v_range,
            max_speed,
            color,
            is_joint=False,
            drag=drag,
            linear_friction=linear_friction,
            angular_friction=angular_friction,
            gravity=gravity,
            collision_filter=collision_filter,
        )
        if obs_range == 0.0:
            assert sensors is None, f"Blind agent cannot have sensors, got {sensors}"

        if action_size is not None and discrete_action_nvec is not None:
            if action_size != len(discrete_action_nvec):
                raise ValueError(
                    f"action_size {action_size} is inconsistent with discrete_action_nvec {discrete_action_nvec}"
                )
        if discrete_action_nvec is not None:
            if not all(n > 1 for n in discrete_action_nvec):
                raise ValueError(
                    f"All values in discrete_action_nvec must be greater than 1, got {discrete_action_nvec}"
                )
        state = AgentState.create(batch_dim, dim_c, dim_p)
        # cannot observe the world
        obs_range = obs_range
        # observation noise
        obs_noise = obs_noise
        # force constraints
        f_range = f_range
        max_f = max_f
        # torque constraints
        t_range = t_range
        max_t = max_t
        # script behavior to execute
        action_script = action_script
        # agents sensors
        sensors = []
        # non differentiable communication noise
        c_noise = c_noise
        # cannot send communication signals
        silent = silent
        # render the agent action force
        render_action = render_action
        # is adversary
        adversary = adversary
        # Render alpha
        alpha = alpha

        # Dynamics
        dynamics = dynamics if dynamics is not None else Holonomic(agent=None)
        # Action
        if action_size is not None:
            action_size = action_size
        elif discrete_action_nvec is not None:
            action_size = len(discrete_action_nvec)
        else:
            action_size = dynamics.needed_action_size
        if discrete_action_nvec is None:
            discrete_action_nvec = [3] * action_size
        else:
            discrete_action_nvec = discrete_action_nvec
        action = Action.create(
            batch_dim=batch_dim,
            u_range=u_range,
            u_multiplier=u_multiplier,
            u_noise=u_noise,
            action_size=action_size,
            comm_dim=dim_c,
        )

        agent = cls(
            **(
                asdict(entity)
                | {
                    "state": state,
                    "action": action,
                    "obs_range": obs_range,
                    "obs_noise": obs_noise,
                    "f_range": f_range,
                    "max_f": max_f,
                    "t_range": t_range,
                    "max_t": max_t,
                    "action_script": action_script,
                    "sensors": sensors,
                    "c_noise": c_noise,
                    "silent": silent,
                    "render_action": render_action,
                    "adversary": adversary,
                    "alpha": alpha,
                    "dynamics": dynamics,
                    "action_size": action_size,
                    "discrete_action_nvec": discrete_action_nvec,
                }
            )
        )

        dynamics = dynamics.replace(agent=agent)
        agent = agent.replace(dynamics=dynamics)

        if sensors is not None:
            sensors = [sensor.replace(agent=agent) for sensor in sensors]
            agent = agent.replace(sensors=sensors)

        return agent

    def action_callback(self, world: "World"):
        self._action_script(self, world)
        if self._silent or world.dim_c == 0:
            assert (
                self.action.c is None
            ), f"Agent {self.name} should not communicate but action script communicates"
        assert (
            self.action.u is not None
        ), f"Action script of {self.name} should set u action"
        assert (
            self._action.u.shape[1] == self.action_size
        ), f"Scripted action of agent {self.name} has wrong shape"

        assert (
            (self.action.u / self.action.u_multiplier_jax_array).abs()
            <= self.action.u_range_jax_array
        ).all(), f"Scripted physical action of {self.name} is out of range"

    def _spawn(self, dim_c: int, dim_p: int) -> "Agent":
        if dim_c == 0:
            assert (
                self.silent
            ), f"Agent {self.name} must be silent when world has no communication"
        if self.silent:
            dim_c = 0
        return super(Agent, self)._spawn(dim_c, dim_p)

    def _reset(self, env_index: int) -> "Agent":
        self = self.replace(action=self.action._reset(env_index))
        self = self.replace(dynamics=self.dynamics.reset(env_index))
        return super(Agent, self)._reset(env_index)

    def render(self, env_index: int = 0) -> "list[Geom]":
        from jaxvmas.simulator import rendering

        geoms = super(Agent, self).render(env_index)
        if len(geoms) == 0:
            return geoms
        for geom in geoms:
            geom.set_color(*self.color.value, alpha=self.alpha)
        if self.sensors is not None:
            for sensor in self.sensors:
                geoms += sensor.render(env_index=env_index)
        if self.render_action and self.state.force is not None:
            velocity = rendering.Line(
                self.state.pos[env_index],
                self.state.pos[env_index]
                + self.state.force[env_index] * 10 * self.shape.circumscribed_radius(),
                width=2,
            )
            velocity.set_color(*self.color.value)
            geoms.append(velocity)

        return geoms


class WorldDynamicState:
    pass


# Multi-agent world
class World(JaxVectorizedObject):

    _agents: list[Agent]
    _landmarks: list[Landmark]
    _x_semidim: float
    _y_semidim: float
    _dim_p: int
    _dim_c: int
    _dt: float
    _substeps: int
    _sub_dt: float
    _drag: float
    _gravity: Array
    _linear_friction: float
    _angular_friction: float
    _collision_force: float
    _joint_force: float
    _torque_constraint_force: float
    _contact_margin: float
    _torque_constraint_force: float
    _joints: dict[frozenset[str], JointConstraint]
    _collidable_pairs: list[tuple[Shape, Shape]]
    _entity_index_map: dict[Entity, int]

    @classmethod
    def __init__(
        cls,
        batch_dim: int,
        dt: float = 0.1,
        substeps: int = 1,  # if you use joints, higher this value to gain simulation stability
        drag: float = DRAG,
        linear_friction: float = LINEAR_FRICTION,
        angular_friction: float = ANGULAR_FRICTION,
        x_semidim: float = None,
        y_semidim: float = None,
        dim_c: int = 0,
        collision_force: float = COLLISION_FORCE,
        joint_force: float = JOINT_FORCE,
        torque_constraint_force: float = TORQUE_CONSTRAINT_FORCE,
        contact_margin: float = 1e-3,
        gravity: tuple[float, float] = (0.0, 0.0),
    ):
        assert batch_dim > 0, f"Batch dim must be greater than 0, got {batch_dim}"
        # list of agents and entities static params(can change at execution-time!)
        _agents = []
        _landmarks = []

        # world dims: no boundaries if none
        _x_semidim = x_semidim
        _y_semidim = y_semidim
        # position dimensionality
        _dim_p = 2
        # communication channel dimensionality
        _dim_c = dim_c
        # simulation timestep
        _dt = dt
        _substeps = substeps
        _sub_dt = _dt / _substeps
        # drag coefficient
        _drag = drag
        # gravity
        _gravity = jnp.asarray(gravity, dtype=jnp.float32)
        # friction coefficients
        _linear_friction = linear_friction
        _angular_friction = angular_friction
        # constraint response parameters
        _collision_force = collision_force
        _joint_force = joint_force
        _contact_margin = contact_margin
        _torque_constraint_force = torque_constraint_force
        # joints
        _joints = {}
        # Pairs of collidable shapes
        _collidable_pairs = [
            {Sphere, Sphere},
            {Sphere, Box},
            {Sphere, Line},
            {Line, Line},
            {Line, Box},
            {Box, Box},
        ]
        # Map to save entity indexes
        entity_index_map = {}

        return cls(
            batch_dim,
            _agents,
            _landmarks,
            _x_semidim,
            _y_semidim,
            _dim_p,
            _dim_c,
            _dt,
            _substeps,
            _sub_dt,
            _drag,
            _gravity,
            _linear_friction,
            _angular_friction,
            _collision_force,
            _joint_force,
            _torque_constraint_force,
            _contact_margin,
            _joints,
            _collidable_pairs,
            entity_index_map,
        )

    def add_agent(
        self,
        agent: Agent,
    ):
        """Only way to add agents to the world"""
        agent.batch_dim = self._batch_dim
        agent = agent._spawn(dim_c=self._dim_c, dim_p=self.dim_p)

        self = self.replace(_agents=self._agents + [agent])
        return self

    def add_landmark(
        self,
        landmark: Landmark,
    ):
        """Only way to add landmarks to the world"""
        landmark.batch_dim = self._batch_dim
        landmark = landmark._spawn(dim_c=self.dim_c, dim_p=self.dim_p)
        self = self.replace(_landmarks=self._landmarks + [landmark])
        return self

    def add_joint(self, joint: Joint):
        assert self._substeps > 1, "For joints, world substeps needs to be more than 1"
        if joint.landmark is not None:
            self = self.add_landmark(joint.landmark)
        for constraint in joint.joint_constraints:
            self = self.replace(
                _joints=self._joints
                | {
                    frozenset(
                        {constraint.entity_a.name, constraint.entity_b.name}
                    ): constraint
                }
            )
        return self

    def reset(self, env_index: int):
        for e in self.entities:
            self = e._reset(env_index)

        return self

    @property
    def agents(self) -> list[Agent]:
        return self._agents

    @property
    def landmarks(self) -> list[Landmark]:
        return self._landmarks

    @property
    def x_semidim(self):
        return self._x_semidim

    @property
    def dt(self):
        return self._dt

    @property
    def y_semidim(self):
        return self._y_semidim

    @property
    def dim_p(self):
        return self._dim_p

    @property
    def dim_c(self):
        return self._dim_c

    @property
    def joints(self):
        return self._joints.values()

    # return all entities in the world
    @property
    def entities(self) -> list[Entity]:
        return self._landmarks + self._agents

    # return all agents controllable by external policies
    @property
    def policy_agents(self) -> list[Agent]:
        return [agent for agent in self._agents if agent.action_script is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self) -> list[Agent]:
        return [agent for agent in self._agents if agent.action_script is not None]

    def _cast_ray_to_box(
        self,
        box: Entity,
        ray_origin: Array,
        ray_direction: Array,
        max_range: float,
    ):
        """
        Inspired from https://tavianator.com/2011/ray_box.html
        Computes distance of ray originating from pos at angle to a box and sets distance to
        max_range if there is no intersection.
        """
        assert ray_origin.ndim == 2 and ray_direction.ndim == 1
        assert ray_origin.shape[0] == ray_direction.shape[0]
        assert isinstance(box.shape, Box)

        pos_origin = ray_origin - box.state.pos
        pos_aabb = JaxUtils.rotate_vector(pos_origin, -box.state.rot)
        ray_dir_world = jnp.stack(
            [jnp.cos(ray_direction), jnp.sin(ray_direction)], dim=-1
        )
        ray_dir_aabb = JaxUtils.rotate_vector(ray_dir_world, -box.state.rot)

        tx1 = (-box.shape.length / 2 - pos_aabb[:, X]) / ray_dir_aabb[:, X]
        tx2 = (box.shape.length / 2 - pos_aabb[:, X]) / ray_dir_aabb[:, X]
        tx = jnp.stack([tx1, tx2], axis=-1)
        tmin, _ = jnp.min(tx, axis=-1)
        tmax, _ = jnp.max(tx, axis=-1)

        ty1 = (-box.shape.width / 2 - pos_aabb[:, Y]) / ray_dir_aabb[:, Y]
        ty2 = (box.shape.width / 2 - pos_aabb[:, Y]) / ray_dir_aabb[:, Y]
        ty = jnp.stack([ty1, ty2], axis=-1)
        tymin, _ = jnp.min(ty, axis=-1)
        tymax, _ = jnp.max(ty, axis=-1)
        tmin, _ = jnp.max(jnp.stack([tmin, tymin], axis=-1), axis=-1)
        tmax, _ = jnp.min(jnp.stack([tmax, tymax], axis=-1), axis=-1)

        intersect_aabb = tmin[:, None] * ray_dir_aabb + pos_aabb
        intersect_world = (
            JaxUtils.rotate_vector(intersect_aabb, box.state.rot) + box.state.pos
        )

        collision = (tmax >= tmin) & (tmin > 0.0)
        dist = jnp.linalg.norm(ray_origin - intersect_world, axis=1)
        dist = jnp.where(collision, dist, max_range)
        return dist

    def _cast_rays_to_box(
        self,
        box_pos: Array,
        box_rot: Array,
        box_length: Array,
        box_width: Array,
        ray_origin: Array,
        ray_direction: Array,
        max_range: float,
    ):
        """
        Inspired from https://tavianator.com/2011/ray_box.html
        Computes distance of ray originating from pos at angle to a box and sets distance to
        max_range if there is no intersection.
        """
        batch_size = ray_origin.shape[:-1]
        assert batch_size[0] == self.batch_dim
        assert ray_origin.shape[-1] == 2  # ray_origin is [*batch_size, 2]
        assert (
            ray_direction.shape[:-1] == batch_size
        )  # ray_direction is [*batch_size, n_angles]
        assert box_pos.shape[:-2] == batch_size
        assert box_pos.shape[-1] == 2
        assert box_rot.shape[:-1] == batch_size
        assert box_width.shape[:-1] == batch_size
        assert box_length.shape[:-1] == batch_size

        num_angles = ray_direction.shape[-1]
        n_boxes = box_pos.shape[-2]

        # Expand input to [*batch_size, n_boxes, num_angles, 2]
        ray_origin = jnp.broadcast_to(
            jnp.expand_dims(jnp.expand_dims(ray_origin, -2), -2),
            (*batch_size, n_boxes, num_angles, 2),
        )
        box_pos_expanded = jnp.broadcast_to(
            jnp.expand_dims(box_pos, -2), (*batch_size, n_boxes, num_angles, 2)
        )
        # Expand input to [*batch_size, n_boxes, num_angles]
        ray_direction = jnp.broadcast_to(
            jnp.expand_dims(ray_direction, -2), (*batch_size, n_boxes, num_angles)
        )
        box_rot_expanded = jnp.broadcast_to(
            jnp.expand_dims(box_rot, -1), (*batch_size, n_boxes, num_angles)
        )
        box_width_expanded = jnp.broadcast_to(
            jnp.expand_dims(box_width, -1), (*batch_size, n_boxes, num_angles)
        )
        box_length_expanded = jnp.broadcast_to(
            jnp.expand_dims(box_length, -1), (*batch_size, n_boxes, num_angles)
        )

        # Compute pos_origin and pos_aabb
        pos_origin = ray_origin - box_pos_expanded
        pos_aabb = JaxUtils.rotate_vector(pos_origin, -box_rot_expanded)

        # Calculate ray_dir_world
        ray_dir_world = jnp.stack(
            [jnp.cos(ray_direction), jnp.sin(ray_direction)], dim=-1
        )

        # Calculate ray_dir_aabb
        ray_dir_aabb = JaxUtils.rotate_vector(ray_dir_world, -box_rot_expanded)

        # Calculate tx, ty, tmin, and tmax
        tx1 = (-box_length_expanded / 2 - pos_aabb[..., X]) / ray_dir_aabb[..., X]
        tx2 = (box_length_expanded / 2 - pos_aabb[..., X]) / ray_dir_aabb[..., X]
        tx = jnp.stack([tx1, tx2], dim=-1)
        tmin, _ = jnp.min(tx, dim=-1)
        tmax, _ = jnp.max(tx, dim=-1)

        ty1 = (-box_width_expanded / 2 - pos_aabb[..., Y]) / ray_dir_aabb[..., Y]
        ty2 = (box_width_expanded / 2 - pos_aabb[..., Y]) / ray_dir_aabb[..., Y]
        ty = jnp.stack([ty1, ty2], dim=-1)
        tymin, _ = jnp.min(ty, dim=-1)
        tymax, _ = jnp.max(ty, dim=-1)
        tmin, _ = jnp.max(jnp.stack([tmin, tymin], dim=-1), dim=-1)
        tmax, _ = jnp.min(jnp.stack([tmax, tymax], dim=-1), dim=-1)

        # Compute intersect_aabb and intersect_world
        intersect_aabb = tmin[..., None] * ray_dir_aabb + pos_aabb
        intersect_world = (
            JaxUtils.rotate_vector(intersect_aabb, box_rot_expanded) + box_pos_expanded
        )

        # Calculate collision and distances
        collision = (tmax >= tmin) & (tmin > 0.0)
        dist = jnp.linalg.norm(ray_origin - intersect_world, axis=-1)
        dist = jnp.where(collision, dist, max_range)
        return dist

    def _cast_ray_to_sphere(
        self,
        sphere: Entity,
        ray_origin: Array,
        ray_direction: Array,
        max_range: float,
    ):
        ray_dir_world = jnp.stack(
            [jnp.cos(ray_direction), jnp.sin(ray_direction)], dim=-1
        )
        test_point_pos = sphere.state.pos
        line_rot = ray_direction
        line_length = max_range
        line_pos = ray_origin + ray_dir_world * (line_length / 2)

        closest_point = _get_closest_point_line(
            line_pos,
            line_rot[..., None],
            line_length,
            test_point_pos,
            limit_to_line_length=False,
        )

        d = test_point_pos - closest_point
        d_norm = jnp.linalg.vector_norm(d, axis=1)
        ray_intersects = d_norm < sphere.shape.radius
        a = sphere.shape.radius**2 - d_norm**2
        m = jnp.sqrt(jnp.where(a > 0, a, 1e-8))

        u = test_point_pos - ray_origin
        u1 = closest_point - ray_origin

        # Dot product of u and u1
        u_dot_ray = (u * ray_dir_world).sum(-1)
        sphere_is_in_front = u_dot_ray > 0.0
        dist = jnp.linalg.vector_norm(u1, axis=1) - m
        dist = jnp.where(ray_intersects & sphere_is_in_front, dist, max_range)

        return dist

    def _cast_rays_to_sphere(
        self,
        sphere_pos: Array,
        sphere_radius: Array,
        ray_origin: Array,
        ray_direction: Array,
        max_range: float,
    ):
        batch_size = ray_origin.shape[:-1]
        assert batch_size[0] == self.batch_dim
        assert ray_origin.shape[-1] == 2  # ray_origin is [*batch_size, 2]
        assert (
            ray_direction.shape[:-1] == batch_size
        )  # ray_direction is [*batch_size, n_angles]
        assert sphere_pos.shape[:-2] == batch_size
        assert sphere_pos.shape[-1] == 2
        assert sphere_radius.shape[:-1] == batch_size

        num_angles = ray_direction.shape[-1]
        n_spheres = sphere_pos.shape[-2]

        # Expand input to [*batch_size, n_spheres, num_angles, 2]
        ray_origin = jnp.broadcast_to(
            ray_origin[..., None, None, :], (*batch_size, n_spheres, num_angles, 2)
        )
        sphere_pos_expanded = jnp.broadcast_to(
            sphere_pos[..., None, :], (*batch_size, n_spheres, num_angles, 2)
        )
        # Expand input to [*batch_size, n_spheres, num_angles]
        ray_direction = jnp.broadcast_to(
            ray_direction[..., None, :], (*batch_size, n_spheres, num_angles)
        )
        sphere_radius_expanded = jnp.broadcast_to(
            sphere_radius[..., None], (*batch_size, n_spheres, num_angles)
        )

        # Calculate ray_dir_world
        ray_dir_world = jnp.stack(
            [jnp.cos(ray_direction), jnp.sin(ray_direction)], dim=-1
        )

        line_rot = ray_direction[..., None]

        # line_length remains scalar and will be broadcasted as needed
        line_length = max_range

        # Calculate line_pos
        line_pos = ray_origin + ray_dir_world * (line_length / 2)

        # Call the updated _get_closest_point_line function
        closest_point = _get_closest_point_line(
            line_pos,
            line_rot,
            line_length,
            sphere_pos_expanded,
            limit_to_line_length=False,
        )

        # Calculate distances and other metrics
        d = sphere_pos_expanded - closest_point
        d_norm = jnp.linalg.vector_norm(d, axis=-1)
        ray_intersects = d_norm < sphere_radius_expanded
        a = sphere_radius_expanded**2 - d_norm**2
        m = jnp.sqrt(jnp.where(a > 0, a, 1e-8))

        u = sphere_pos_expanded - ray_origin
        u1 = closest_point - ray_origin

        # Dot product of u and u1
        u_dot_ray = (u * ray_dir_world).sum(-1)
        sphere_is_in_front = u_dot_ray > 0.0
        dist = jnp.linalg.vector_norm(u1, axis=-1) - m
        dist = jnp.where(ray_intersects & sphere_is_in_front, dist, max_range)

        return dist

    def _cast_ray_to_line(
        self,
        line: Entity,
        ray_origin: Array,
        ray_direction: Array,
        max_range: float,
    ):
        """
        Inspired by https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect/565282#565282
        Computes distance of ray originating from pos at angle to a line and sets distance to
        max_range if there is no intersection.
        """
        assert ray_origin.ndim == 2 and ray_direction.ndim == 1
        assert ray_origin.shape[0] == ray_direction.shape[0]
        assert isinstance(line.shape, Line)

        p = line.state.pos
        r = (
            jnp.stack(
                [
                    jnp.cos(line.state.rot),
                    jnp.sin(line.state.rot),
                ],
                dim=-1,
            )
            * line.shape.length
        )

        q = ray_origin
        s = jnp.stack(
            [
                jnp.cos(ray_direction),
                jnp.sin(ray_direction),
            ],
            dim=-1,
        )

        rxs = JaxUtils.cross(r, s)
        t = JaxUtils.cross(q - p, s / rxs)
        u = JaxUtils.cross(q - p, r / rxs)
        d = jnp.linalg.norm(u * s, axis=-1)

        perpendicular = rxs == 0.0
        above_line = t > 0.5
        below_line = t < -0.5
        behind_line = u < 0.0
        d = jnp.where(perpendicular.squeeze(-1), max_range, d)
        d = jnp.where(above_line.squeeze(-1), max_range, d)
        d = jnp.where(below_line.squeeze(-1), max_range, d)
        d = jnp.where(behind_line.squeeze(-1), max_range, d)
        return d

    def _cast_rays_to_line(
        self,
        line_pos: Array,
        line_rot: Array,
        line_length: Array,
        ray_origin: Array,
        ray_direction: Array,
        max_range: float,
    ):
        """
        Inspired by https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect/565282#565282
        Computes distance of ray originating from pos at angle to a line and sets distance to
        max_range if there is no intersection.
        """
        batch_size = ray_origin.shape[:-1]
        assert batch_size[0] == self.batch_dim
        assert ray_origin.shape[-1] == 2  # ray_origin is [*batch_size, 2]
        assert (
            ray_direction.shape[:-1] == batch_size
        )  # ray_direction is [*batch_size, n_angles]
        assert line_pos.shape[:-2] == batch_size
        assert line_pos.shape[-1] == 2
        assert line_rot.shape[:-1] == batch_size
        assert line_length.shape[:-1] == batch_size

        num_angles = ray_direction.shape[-1]
        n_lines = line_pos.shape[-2]

        # Expand input to [*batch_size, n_lines, num_angles, 2]
        ray_origin = ray_origin[..., None, None, :]
        ray_origin = jnp.broadcast_to(ray_origin, (*batch_size, n_lines, num_angles, 2))

        line_pos_expanded = line_pos[..., None, :]
        line_pos_expanded = jnp.broadcast_to(
            line_pos_expanded, (*batch_size, n_lines, num_angles, 2)
        )

        # Expand input to [*batch_size, n_lines, num_angles]
        ray_direction = ray_direction[..., None, :]
        ray_direction = jnp.broadcast_to(
            ray_direction, (*batch_size, n_lines, num_angles)
        )

        line_rot_expanded = line_rot[..., None]
        line_rot_expanded = jnp.broadcast_to(
            line_rot_expanded, (*batch_size, n_lines, num_angles)
        )

        line_length_expanded = line_length[..., None]
        line_length_expanded = jnp.broadcast_to(
            line_length_expanded, (*batch_size, n_lines, num_angles)
        )

        # Expand line state variables
        r = jnp.stack(
            [
                jnp.cos(line_rot_expanded),
                jnp.sin(line_rot_expanded),
            ],
            dim=-1,
        ) * line_length_expanded.unsqueeze(-1)

        # Calculate q and s
        q = ray_origin
        s = jnp.stack(
            [
                jnp.cos(ray_direction),
                jnp.sin(ray_direction),
            ],
            dim=-1,
        )

        # Calculate rxs, t, u, and d
        rxs = JaxUtils.cross(r, s)
        t = JaxUtils.cross(q - line_pos_expanded, s / rxs)
        u = JaxUtils.cross(q - line_pos_expanded, r / rxs)
        d = jnp.linalg.norm(u * s, dim=-1)

        # Handle edge cases
        perpendicular = rxs == 0.0
        above_line = t > 0.5
        below_line = t < -0.5
        behind_line = u < 0.0
        d = jnp.where(perpendicular.squeeze(-1), max_range, d)
        d = jnp.where(above_line.squeeze(-1), max_range, d)
        d = jnp.where(below_line.squeeze(-1), max_range, d)
        d = jnp.where(behind_line.squeeze(-1), max_range, d)
        return d

    def cast_ray(
        self,
        entity: Entity,
        angles: Array,
        max_range: float,
        entity_filter: Callable[[Entity], bool] = lambda _: False,
    ):
        pos = entity.state.pos

        assert pos.ndim == 2 and angles.ndim == 1
        assert pos.shape[0] == angles.shape[0]

        # Initialize with full max_range to avoid dists being empty when all entities are filtered
        dists = [jnp.full((self.batch_dim,), fill_value=max_range)]
        n_agents = len(self.agents)
        for i, e in enumerate(self.entities):

            e_state = None
            if i < n_agents:
                e_state = self.agents[i].state
            else:
                e_state = self.landmarks[i - n_agents].state

            if entity is e or not entity_filter(e):
                continue
            assert e.collides(entity) and entity.collides(
                e
            ), "Rays are only casted among collidables"
            if isinstance(e.shape, Box):
                d = self._cast_ray_to_box(e, e_state, pos, angles, max_range)
            elif isinstance(e.shape, Sphere):
                d = self._cast_ray_to_sphere(e, e_state, pos, angles, max_range)
            elif isinstance(e.shape, Line):
                d = self._cast_ray_to_line(e, e_state, pos, angles, max_range)
            else:
                raise RuntimeError(f"Shape {e.shape} currently not handled by cast_ray")
            dists.append(d)
        dist, _ = jnp.min(jnp.stack(dists, dim=-1), dim=-1)
        return dist

    def cast_rays(
        self,
        entity: Entity,
        angles: Array,
        max_range: float,
        entity_filter: Callable[[Entity], bool] = lambda _: False,
    ):
        pos = entity.state.pos

        # Initialize with full max_range to avoid dists being empty when all entities are filtered
        dists = jnp.full_like(angles, fill_value=max_range)[..., None]
        boxes: list[Box] = []
        boxes_state: list[EntityState] = []
        spheres: list[Sphere] = []
        spheres_state: list[EntityState] = []
        lines: list[Line] = []
        lines_state: list[EntityState] = []
        n_agents = len(self.agents)
        for i, e in enumerate(self.entities):
            e_state = None
            if i < n_agents:
                e_state = self.agents[i].state
            else:
                e_state = self.landmarks[i - n_agents].state
            if entity is e or not entity_filter(e):
                continue
            assert e.collides(entity) and entity.collides(
                e
            ), "Rays are only casted among collidables"
            if isinstance(e.shape, Box):
                boxes.append(e)
                boxes_state.append(e_state)
            elif isinstance(e.shape, Sphere):
                spheres.append(e)
                spheres_state.append(e_state)
            elif isinstance(e.shape, Line):
                lines.append(e)
                lines_state.append(e_state)
            else:
                raise RuntimeError(f"Shape {e.shape} currently not handled by cast_ray")

        # Boxes
        if len(boxes):
            pos_box = []
            rot_box = []
            length_box = []
            width_box = []
            for box, box_state in zip(boxes, boxes_state):
                pos_box.append(box_state.pos)
                rot_box.append(box_state.rot)
                length_box.append(box.shape.length)
                width_box.append(box.shape.width)
            pos_box = jnp.stack(pos_box, dim=-2)
            rot_box = jnp.stack(rot_box, dim=-2)
            length_box = jnp.stack(length_box, dim=-1)
            width_box = jnp.stack(width_box, dim=-1)
            width_box = jnp.stack(width_box, dim=-1)
            width_box = jnp.expand_dims(width_box, 0)
            width_box = jnp.broadcast_to(width_box, (self.batch_dim, -1))
            dist_boxes = self._cast_rays_to_box(
                pos_box,
                jnp.squeeze(rot_box, axis=-1),
                length_box,
                width_box,
                pos,
                angles,
                max_range,
            )
            dists = jnp.concatenate([dists, dist_boxes.transpose(-1, -2)], axis=-1)
        # Spheres
        if len(spheres):
            pos_s = []
            radius_s = []
            for s, s_state in zip(spheres, spheres_state):
                pos_s.append(s_state.pos)
                radius_s.append(s.shape.radius)
            pos_s = jnp.stack(pos_s, dim=-2)
            radius_s = jnp.stack(radius_s, dim=-1)
            radius_s = jnp.expand_dims(radius_s, 0)
            radius_s = jnp.broadcast_to(radius_s, (self.batch_dim, -1))

            dist_spheres = self._cast_rays_to_sphere(
                pos_s,
                radius_s,
                pos,
                angles,
                max_range,
            )
            dists = jnp.concatenate([dists, dist_spheres.transpose(-1, -2)], axis=-1)
        # Lines
        if len(lines):
            pos_l = []
            rot_l = []
            length_l = []
            for line, line_state in zip(lines, lines_state):
                pos_l.append(line_state.pos)
                rot_l.append(line_state.rot)
                length_l.append(line.shape.length)
            pos_l = jnp.stack(pos_l, dim=-2)
            rot_l = jnp.stack(rot_l, dim=-2)
            length_l = jnp.stack(length_l, dim=-1)
            length_l = jnp.expand_dims(length_l, 0)
            length_l = jnp.broadcast_to(length_l, (self.batch_dim, -1))

            dist_lines = self._cast_rays_to_line(
                pos_l,
                jnp.squeeze(rot_l, axis=-1),
                length_l,
                pos,
                angles,
                max_range,
            )
            dists = jnp.concatenate([dists, dist_lines.transpose(-1, -2)], axis=-1)

        dist, _ = jnp.min(dists, axis=-1)
        return dist

    # start from here !!!!
    def get_distance_from_point(
        self,
        entity: Entity,
        test_point_pos: Array,
        env_index: int = None,
    ):
        self._check_batch_index(env_index)

        if isinstance(entity.shape, Sphere):
            delta_pos = entity.state.pos - test_point_pos
            dist = jnp.linalg.vector_norm(delta_pos, axis=-1)
            return_value = dist - entity.shape.radius
        elif isinstance(entity.shape, Box):
            closest_point = _get_closest_point_box(
                entity.state.pos,
                entity.state.rot,
                entity.shape.width,
                entity.shape.length,
                test_point_pos,
            )
            distance = jnp.linalg.vector_norm(test_point_pos - closest_point, axis=-1)
            return_value = distance - LINE_MIN_DIST
        elif isinstance(entity.shape, Line):
            closest_point = _get_closest_point_line(
                entity.state.pos,
                entity.state.rot,
                entity.shape.length,
                test_point_pos,
            )
            distance = jnp.linalg.vector_norm(test_point_pos - closest_point, axis=-1)
            return_value = distance - LINE_MIN_DIST
        else:
            raise RuntimeError("Distance not computable for given entity")
        if env_index is not None:
            return_value = return_value[env_index]
        return return_value

    def get_distance(
        self,
        entity_a: Entity,
        entity_b: Entity,
        env_index: int = None,
    ):
        a_shape = entity_a.shape
        b_shape = entity_b.shape

        if isinstance(a_shape, Sphere) and isinstance(b_shape, Sphere):
            dist = self.get_distance_from_point(entity_a, entity_b.state.pos, env_index)
            return_value = dist - b_shape.radius
        elif (
            isinstance(entity_a.shape, Box)
            and isinstance(entity_b.shape, Sphere)
            or isinstance(entity_b.shape, Box)
            and isinstance(entity_a.shape, Sphere)
        ):
            box, sphere = (
                (entity_a, entity_b)
                if isinstance(entity_b.shape, Sphere)
                else (entity_b, entity_a)
            )
            dist = self.get_distance_from_point(box, entity_b.state.pos, env_index)
            return_value = dist - sphere.shape.radius
            is_overlapping = self.is_overlapping(entity_a, entity_b)
            return_value[is_overlapping] = -1
        elif (
            isinstance(entity_a.shape, Line)
            and isinstance(entity_b.shape, Sphere)
            or isinstance(entity_b.shape, Line)
            and isinstance(entity_a.shape, Sphere)
        ):
            line, sphere = (
                (entity_a, entity_b)
                if isinstance(entity_b.shape, Sphere)
                else (entity_b, entity_a)
            )
            dist = self.get_distance_from_point(line, entity_b.state.pos, env_index)
            return_value = dist - sphere.shape.radius
        elif isinstance(entity_a.shape, Line) and isinstance(entity_b.shape, Line):
            point_a, point_b = _get_closest_points_line_line(
                entity_a.state.pos,
                entity_a.state.rot,
                entity_a.shape.length,
                entity_b.state.pos,
                entity_b.state.rot,
                entity_b.shape.length,
            )
            dist = jnp.linalg.vector_norm(point_a - point_b, axis=-1)
            return_value = dist - LINE_MIN_DIST
        elif (
            isinstance(entity_a.shape, Box)
            and isinstance(entity_b.shape, Line)
            or isinstance(entity_b.shape, Box)
            and isinstance(entity_a.shape, Line)
        ):
            box, box_state, line, line_state = (
                (entity_a, entity_a.state, entity_b, entity_b.state)
                if isinstance(entity_b.shape, Line)
                else (entity_b, entity_b.state, entity_a, entity_a.state)
            )
            point_box, point_line = _get_closest_line_box(
                box_state.pos,
                box_state.rot,
                box.shape.width,
                box.shape.length,
                line_state.pos,
                line_state.rot,
                line.shape.length,
            )
            dist = jnp.linalg.vector_norm(point_box - point_line, dim=1)
            return_value = dist - LINE_MIN_DIST
        elif isinstance(entity_a.shape, Box) and isinstance(entity_b.shape, Box):
            point_a, point_b = _get_closest_box_box(
                entity_a.state.pos,
                entity_a.state.rot,
                entity_a.shape.width,
                entity_a.shape.length,
                entity_b.state.pos,
                entity_b.state.rot,
                entity_b.shape.width,
                entity_b.shape.length,
            )
            dist = jnp.linalg.vector_norm(point_a - point_b, axis=-1)
            return_value = dist - LINE_MIN_DIST
        else:
            raise RuntimeError("Distance not computable for given entities")
        return return_value

    def is_overlapping(
        self,
        entity_a: Entity,
        entity_b: Entity,
        env_index: int = None,
    ):
        a_shape = entity_a.shape
        b_shape = entity_b.shape
        self._check_batch_index(env_index)

        # Sphere sphere, sphere line, line line, line box, box box
        if (
            (isinstance(a_shape, Sphere) and isinstance(b_shape, Sphere))
            or (
                (
                    isinstance(entity_a.shape, Line)
                    and isinstance(entity_b.shape, Sphere)
                    or isinstance(entity_b.shape, Line)
                    and isinstance(entity_a.shape, Sphere)
                )
            )
            or (isinstance(entity_a.shape, Line) and isinstance(entity_b.shape, Line))
            or (
                isinstance(entity_a.shape, Box)
                and isinstance(entity_b.shape, Line)
                or isinstance(entity_b.shape, Box)
                and isinstance(entity_a.shape, Line)
            )
            or (isinstance(entity_a.shape, Box) and isinstance(entity_b.shape, Box))
        ):
            return self.get_distance(entity_a, entity_b, env_index) < 0
        elif (
            isinstance(entity_a.shape, Box)
            and isinstance(entity_b.shape, Sphere)
            or isinstance(entity_b.shape, Box)
            and isinstance(entity_a.shape, Sphere)
        ):
            box, box_state, sphere, sphere_state = (
                (entity_a, entity_a.state, entity_b, entity_b.state)
                if isinstance(entity_b.shape, Sphere)
                else (entity_b, entity_b.state, entity_a, entity_a.state)
            )
            closest_point = _get_closest_point_box(
                box.state.pos,
                box.state.rot,
                box.shape.width,
                box.shape.length,
                sphere.state.pos,
            )

            distance_sphere_closest_point = jnp.linalg.vector_norm(
                sphere.state.pos - closest_point, axis=-1
            )
            distance_sphere_box = jnp.linalg.vector_norm(
                sphere.state.pos - box.state.pos, axis=-1
            )
            distance_closest_point_box = jnp.linalg.vector_norm(
                box.state.pos - closest_point, axis=-1
            )
            dist_min = sphere.shape.radius + LINE_MIN_DIST
            return_value = (distance_sphere_box < distance_closest_point_box) + (
                distance_sphere_closest_point < dist_min
            )
        else:
            raise RuntimeError("Overlap not computable for give entities")
        if env_index is not None:
            return_value = return_value[env_index]
        return return_value

    # update state of the world
    def step(self):
        self.entity_index_map = {e: i for i, e in enumerate(self.entities)}
        for substep in range(self._substeps):
            forces_dict = {
                e: jnp.zeros(
                    self._batch_dim,
                    self._dim_p,
                    dtype=jnp.float32,
                )
                for e in self.entities
            }
            torques_dict = {
                e: jnp.zeros(
                    self._batch_dim,
                    1,
                    dtype=jnp.float32,
                )
                for e in self.entities
            }
            entities = []
            for entity in self.entities:
                if isinstance(entity, Agent):
                    # apply agent force controls
                    entity, forces_dict = self._apply_action_force(entity, forces_dict)
                    # apply agent torque controls
                    entity, torques_dict = self._apply_action_torque(
                        entity, torques_dict
                    )
                # apply friction
                entity, forces_dict, torques_dict = self._apply_friction_force(
                    entity, forces_dict, torques_dict
                )
                # apply gravity
                entity, forces_dict = self._apply_gravity(entity, forces_dict)
                entities.append(entity)

            self = self.replace(entities=entities)

            self._apply_vectorized_enviornment_force()

            entities = []
            for entity in self.entities:
                # integrate physical state
                entity = self._integrate_state(
                    entity, substep, forces_dict, torques_dict
                )
                entities.append(entity)

            self = self.replace(entities=entities)

        # update non-differentiable comm state
        if self._dim_c > 0:
            for agent in self._agents:
                self._update_comm_state(agent)

        return self

    # gather agent action forces
    def _apply_action_force(self, agent: Agent, forces_dict: dict[Agent, Array]):
        forces_dict = {**forces_dict}
        if agent.movable:
            if agent.max_f is not None:
                agent = agent.replace(
                    state=agent.state.replace(
                        force=JaxUtils.clamp_with_norm(agent.state.force, agent.max_f)
                    )
                )
            if agent.f_range is not None:
                agent = agent.replace(
                    state=agent.state.replace(
                        force=jnp.clip(agent.state.force, -agent.f_range, agent.f_range)
                    )
                )
            forces_dict[agent] = forces_dict[agent] + agent.state.force
        return agent, forces_dict

    def _apply_action_torque(self, agent: Agent, torques_dict: dict[Agent, Array]):
        torques_dict = {**torques_dict}
        if agent.rotatable:
            if agent.max_t is not None:
                agent = agent.replace(
                    state=agent.state.replace(
                        torque=JaxUtils.clamp_with_norm(agent.state.torque, agent.max_t)
                    )
                )
            if agent.t_range is not None:
                agent = agent.replace(
                    state=agent.state.replace(
                        torque=jnp.clip(
                            agent.state.torque, -agent.t_range, agent.t_range
                        )
                    )
                )

            torques_dict[agent] = torques_dict[agent] + agent.state.torque
        return agent, torques_dict

    def _apply_gravity(
        self,
        entity: Entity,
        forces_dict: dict[Entity, Array],
    ):
        forces_dict = {**forces_dict}
        if entity.movable:
            if not (self._gravity == 0.0).all():
                forces_dict[entity] = forces_dict[entity] + entity.mass * self._gravity
            if entity.gravity is not None:
                forces_dict[entity] = forces_dict[entity] + entity.mass * entity.gravity
        return entity, forces_dict

    def _apply_friction_force(
        self,
        entity: Entity,
        forces_dict: dict[Entity, Array],
        torques_dict: dict[Entity, Array],
    ):
        def get_friction_force(vel, coeff, force, mass):
            speed = jnp.linalg.vector_norm(vel, axis=-1)
            static = speed == 0
            static_exp = jnp.broadcast_to(static[..., None], vel.shape)

            if not isinstance(coeff, Array):
                coeff = jnp.full_like(force, coeff)
            coeff = jnp.broadcast_to(coeff, force.shape)

            friction_force_constant = coeff * mass

            friction_force = -(
                vel / jnp.broadcast_to(jnp.where(static, 1e-8, speed), vel.shape)
            ) * jnp.minimum(
                friction_force_constant, (jnp.abs(vel) / self._sub_dt) * mass
            )
            friction_force = jnp.where(static_exp, 0.0, friction_force)

            return friction_force

        forces_dict = {**forces_dict}
        torques_dict = {**torques_dict}
        if entity.linear_friction is not None:
            forces_dict[entity] = forces_dict[entity] + get_friction_force(
                entity.state.vel,
                entity.linear_friction,
                forces_dict[entity],
                entity.mass,
            )
        elif self._linear_friction > 0:
            forces_dict[entity] = forces_dict[entity] + get_friction_force(
                entity.state.vel,
                self._linear_friction,
                forces_dict[entity],
                entity.mass,
            )
        if entity.angular_friction is not None:
            torques_dict[entity] = torques_dict[entity] + get_friction_force(
                entity.state.ang_vel,
                entity.angular_friction,
                torques_dict[entity],
                entity.moment_of_inertia,
            )
        elif self._angular_friction > 0:
            torques_dict[entity] = torques_dict[entity] + get_friction_force(
                entity.state.ang_vel,
                self._angular_friction,
                torques_dict[entity],
                entity.moment_of_inertia,
            )

        return entity, forces_dict, torques_dict

    def _apply_vectorized_enviornment_force(self):
        s_s = []
        l_s = []
        b_s = []
        l_l = []
        b_l = []
        b_b = []
        joints = []
        for a, entity_a in enumerate(self.entities):
            for b, entity_b in enumerate(self.entities):
                if b <= a:
                    continue
                joint = self._joints.get(
                    frozenset({entity_a.name, entity_b.name}), None
                )
                if joint is not None:
                    joints.append(joint)
                    if joint.dist == 0:
                        continue
                if not self.collides(entity_a, entity_b):
                    continue
                if isinstance(entity_a.shape, Sphere) and isinstance(
                    entity_b.shape, Sphere
                ):
                    s_s.append((entity_a, entity_b))
                elif (
                    isinstance(entity_a.shape, Line)
                    and isinstance(entity_b.shape, Sphere)
                    or isinstance(entity_b.shape, Line)
                    and isinstance(entity_a.shape, Sphere)
                ):
                    line, sphere = (
                        (entity_a, entity_b)
                        if isinstance(entity_b.shape, Sphere)
                        else (entity_b, entity_a)
                    )
                    l_s.append((line, sphere))
                elif isinstance(entity_a.shape, Line) and isinstance(
                    entity_b.shape, Line
                ):
                    l_l.append((entity_a, entity_b))
                elif (
                    isinstance(entity_a.shape, Box)
                    and isinstance(entity_b.shape, Sphere)
                    or isinstance(entity_b.shape, Box)
                    and isinstance(entity_a.shape, Sphere)
                ):
                    box, sphere = (
                        (entity_a, entity_b)
                        if isinstance(entity_b.shape, Sphere)
                        else (entity_b, entity_a)
                    )
                    b_s.append((box, sphere))
                elif (
                    isinstance(entity_a.shape, Box)
                    and isinstance(entity_b.shape, Line)
                    or isinstance(entity_b.shape, Box)
                    and isinstance(entity_a.shape, Line)
                ):
                    box, line = (
                        (entity_a, entity_b)
                        if isinstance(entity_b.shape, Line)
                        else (entity_b, entity_a)
                    )
                    b_l.append((box, line))
                elif isinstance(entity_a.shape, Box) and isinstance(
                    entity_b.shape, Box
                ):
                    b_b.append((entity_a, entity_b))
                else:
                    raise AssertionError()
        # Joints
        self._vectorized_joint_constraints(joints)

        # Sphere and sphere
        self._sphere_sphere_vectorized_collision(s_s)
        # Line and sphere
        self._sphere_line_vectorized_collision(l_s)
        # Line and line
        self._line_line_vectorized_collision(l_l)
        # Box and sphere
        self._box_sphere_vectorized_collision(b_s)
        # Box and line
        self._box_line_vectorized_collision(b_l)
        # Box and box
        self._box_box_vectorized_collision(b_b)

    def update_env_forces(self, entity_a, f_a, t_a, entity_b, f_b, t_b):
        if entity_a.movable:
            self.forces_dict[entity_a] = self.forces_dict[entity_a] + f_a
        if entity_a.rotatable:
            self.torques_dict[entity_a] = self.torques_dict[entity_a] + t_a
        if entity_b.movable:
            self.forces_dict[entity_b] = self.forces_dict[entity_b] + f_b
        if entity_b.rotatable:
            self.torques_dict[entity_b] = self.torques_dict[entity_b] + t_b

    def _vectorized_joint_constraints(self, joints):
        if len(joints):
            pos_a = []
            pos_b = []
            pos_joint_a = []
            pos_joint_b = []
            dist = []
            rotate = []
            rot_a = []
            rot_b = []
            joint_rot = []
            for joint in joints:
                entity_a = joint.entity_a
                entity_b = joint.entity_b
                pos_joint_a.append(joint.pos_point(entity_a))
                pos_joint_b.append(joint.pos_point(entity_b))
                pos_a.append(entity_a.state.pos)
                pos_b.append(entity_b.state.pos)
                dist.append(joint.dist)
                rotate.append(joint.rotate)
                rot_a.append(entity_a.state.rot)
                rot_b.append(entity_b.state.rot)
                joint_rot.append(
                    jnp.broadcast_to(
                        jnp.asarray(joint.fixed_rotation)[..., None],
                        (self.batch_dim, 1),
                    )
                    if isinstance(joint.fixed_rotation, float)
                    else joint.fixed_rotation
                )
            pos_a = jnp.stack(pos_a, axis=-2)
            pos_b = jnp.stack(pos_b, axis=-2)
            pos_joint_a = jnp.stack(pos_joint_a, axis=-2)
            pos_joint_b = jnp.stack(pos_joint_b, axis=-2)
            rot_a = jnp.stack(rot_a, axis=-2)
            rot_b = jnp.stack(rot_b, axis=-2)
            dist = jnp.broadcast_to(
                jnp.stack(
                    dist,
                    axis=-1,
                )[None],
                (self.batch_dim, -1),
            )
            rotate_prior = jnp.stack(
                rotate,
                axis=-1,
            )
            rotate = jnp.broadcast_to(
                jnp.expand_dims(rotate_prior, 0),
                (self.batch_dim, -1, 1),
            )
            joint_rot = jnp.stack(joint_rot, axis=-2)

            (
                force_a_attractive,
                force_b_attractive,
            ) = self._get_constraint_forces(
                pos_joint_a,
                pos_joint_b,
                dist_min=dist,
                attractive=True,
                force_multiplier=self._joint_force,
            )
            force_a_repulsive, force_b_repulsive = self._get_constraint_forces(
                pos_joint_a,
                pos_joint_b,
                dist_min=dist,
                attractive=False,
                force_multiplier=self._joint_force,
            )
            force_a = force_a_attractive + force_a_repulsive
            force_b = force_b_attractive + force_b_repulsive
            r_a = pos_joint_a - pos_a
            r_b = pos_joint_b - pos_b

            torque_a_rotate = JaxUtils.compute_torque(force_a, r_a)
            torque_b_rotate = JaxUtils.compute_torque(force_b, r_b)

            torque_a_fixed, torque_b_fixed = self._get_constraint_torques(
                rot_a, rot_b + joint_rot, force_multiplier=self._torque_constraint_force
            )

            torque_a = jnp.where(
                rotate, torque_a_rotate, torque_a_rotate + torque_a_fixed
            )
            torque_b = jnp.where(
                rotate, torque_b_rotate, torque_b_rotate + torque_b_fixed
            )

            for i, joint in enumerate(joints):
                self.update_env_forces(
                    joint.entity_a,
                    force_a[:, i],
                    torque_a[:, i],
                    joint.entity_b,
                    force_b[:, i],
                    torque_b[:, i],
                )

    def _sphere_sphere_vectorized_collision(self, s_s):
        if len(s_s):
            pos_s_a = []
            pos_s_b = []
            radius_s_a = []
            radius_s_b = []
            for s_a, s_b in s_s:
                pos_s_a.append(s_a.state.pos)
                pos_s_b.append(s_b.state.pos)
                radius_s_a.append(s_a.shape.radius)
                radius_s_b.append(s_b.shape.radius)

            pos_s_a = jnp.stack(pos_s_a, axis=-2)
            pos_s_b = jnp.stack(pos_s_b, axis=-2)
            radius_s_a = jnp.broadcast_to(
                jnp.stack(
                    radius_s_a,
                    axis=-1,
                )[None],
                (self.batch_dim, -1),
            )
            radius_s_b = jnp.broadcast_to(
                jnp.stack(
                    radius_s_b,
                    axis=-1,
                )[None],
                (self.batch_dim, -1),
            )
            force_a, force_b = self._get_constraint_forces(
                pos_s_a,
                pos_s_b,
                dist_min=radius_s_a + radius_s_b,
                force_multiplier=self._collision_force,
            )

            for i, (entity_a, entity_b) in enumerate(s_s):
                self.update_env_forces(
                    entity_a,
                    force_a[:, i],
                    0,
                    entity_b,
                    force_b[:, i],
                    0,
                )

    def _sphere_line_vectorized_collision(self, l_s):
        if len(l_s):
            pos_l = []
            pos_s = []
            rot_l = []
            radius_s = []
            length_l = []
            for line, sphere in l_s:
                pos_l.append(line.state.pos)
                pos_s.append(sphere.state.pos)
                rot_l.append(line.state.rot)
                radius_s.append(sphere.shape.radius)
                length_l.append(line.shape.length)
            pos_l = jnp.stack(pos_l, axis=-2)
            pos_s = jnp.stack(pos_s, axis=-2)
            rot_l = jnp.stack(rot_l, axis=-2)
            radius_s = jnp.broadcast_to(
                jnp.stack(
                    radius_s,
                    dim=-1,
                )[None],
                (self.batch_dim, -1),
            )
            length_l = jnp.broadcast_to(
                jnp.stack(
                    length_l,
                    dim=-1,
                )[None],
                (self.batch_dim, -1),
            )

            closest_point = _get_closest_point_line(pos_l, rot_l, length_l, pos_s)
            force_sphere, force_line = self._get_constraint_forces(
                pos_s,
                closest_point,
                dist_min=radius_s + LINE_MIN_DIST,
                force_multiplier=self._collision_force,
            )
            r = closest_point - pos_l
            torque_line = JaxUtils.compute_torque(force_line, r)

            for i, (entity_a, entity_b) in enumerate(l_s):
                self.update_env_forces(
                    entity_a,
                    force_line[:, i],
                    torque_line[:, i],
                    entity_b,
                    force_sphere[:, i],
                    0,
                )

    def _line_line_vectorized_collision(self, l_l):
        if len(l_l):
            pos_l_a = []
            pos_l_b = []
            rot_l_a = []
            rot_l_b = []
            length_l_a = []
            length_l_b = []
            for l_a, l_b in l_l:
                pos_l_a.append(l_a.state.pos)
                pos_l_b.append(l_b.state.pos)
                rot_l_a.append(l_a.state.rot)
                rot_l_b.append(l_b.state.rot)
                length_l_a.append(l_a.shape.length)
                length_l_b.append(l_b.shape.length)
            pos_l_a = jnp.stack(pos_l_a, axis=-2)
            pos_l_b = jnp.stack(pos_l_b, axis=-2)
            rot_l_a = jnp.stack(rot_l_a, axis=-2)
            rot_l_b = jnp.stack(rot_l_b, axis=-2)
            length_l_a = jnp.broadcast_to(
                jnp.stack(
                    length_l_a,
                    dim=-1,
                )[None],
                (self.batch_dim, -1),
            )
            length_l_b = jnp.broadcast_to(
                jnp.stack(
                    length_l_b,
                    axis=-1,
                )[None],
                (self.batch_dim, -1),
            )

            point_a, point_b = _get_closest_points_line_line(
                pos_l_a,
                rot_l_a,
                length_l_a,
                pos_l_b,
                rot_l_b,
                length_l_b,
            )
            force_a, force_b = self._get_constraint_forces(
                point_a,
                point_b,
                dist_min=LINE_MIN_DIST,
                force_multiplier=self._collision_force,
            )
            r_a = point_a - pos_l_a
            r_b = point_b - pos_l_b

            torque_a = JaxUtils.compute_torque(force_a, r_a)
            torque_b = JaxUtils.compute_torque(force_b, r_b)
            for i, (entity_a, entity_b) in enumerate(l_l):
                self.update_env_forces(
                    entity_a,
                    force_a[:, i],
                    torque_a[:, i],
                    entity_b,
                    force_b[:, i],
                    torque_b[:, i],
                )

    def _box_sphere_vectorized_collision(self, b_s):
        if len(b_s):
            pos_box = []
            pos_sphere = []
            rot_box = []
            length_box = []
            width_box = []
            not_hollow_box = []
            radius_sphere = []
            for box, sphere in b_s:
                pos_box.append(box.state.pos)
                pos_sphere.append(sphere.state.pos)
                rot_box.append(box.state.rot)
                length_box.append(box.shape.length)
                width_box.append(box.shape.width)
                not_hollow_box.append(not box.shape.hollow)
                radius_sphere.append(sphere.shape.radius)
            pos_box = jnp.stack(pos_box, axis=-2)
            pos_sphere = jnp.stack(pos_sphere, axis=-2)
            rot_box = jnp.stack(rot_box, axis=-2)
            length_box = jnp.broadcast_to(
                jnp.stack(
                    length_box,
                    axis=-1,
                )[None],
                (self.batch_dim, -1),
            )
            width_box = jnp.broadcast_to(
                jnp.stack(
                    width_box,
                    axis=-1,
                )[None],
                (self.batch_dim, -1),
            )
            not_hollow_box_prior = jnp.stack(
                not_hollow_box,
                axis=-1,
            )
            not_hollow_box = jnp.broadcast_to(
                not_hollow_box_prior[None],
                (self.batch_dim, -1),
            )
            radius_sphere = jnp.broadcast_to(
                jnp.stack(
                    radius_sphere,
                    axis=-1,
                )[None],
                (self.batch_dim, -1),
            )

            closest_point_box = _get_closest_point_box(
                pos_box,
                rot_box,
                width_box,
                length_box,
                pos_sphere,
            )

            inner_point_box = closest_point_box
            d = jnp.zeros_like(radius_sphere)
            if not_hollow_box_prior.any():
                inner_point_box_hollow, d_hollow = _get_inner_point_box(
                    pos_sphere, closest_point_box, pos_box
                )
                cond = jnp.broadcast_to(
                    not_hollow_box[..., None],
                    inner_point_box.shape,
                )
                inner_point_box = jnp.where(
                    cond, inner_point_box_hollow, inner_point_box
                )
                d = jnp.where(not_hollow_box, d_hollow, d)

            force_sphere, force_box = self._get_constraint_forces(
                pos_sphere,
                inner_point_box,
                dist_min=radius_sphere + LINE_MIN_DIST + d,
                force_multiplier=self._collision_force,
            )
            r = closest_point_box - pos_box
            torque_box = JaxUtils.compute_torque(force_box, r)

            for i, (entity_a, entity_b) in enumerate(b_s):
                self.update_env_forces(
                    entity_a,
                    force_box[:, i],
                    torque_box[:, i],
                    entity_b,
                    force_sphere[:, i],
                    0,
                )

    def _box_line_vectorized_collision(self, b_l):
        if len(b_l):
            pos_box = []
            pos_line = []
            rot_box = []
            rot_line = []
            length_box = []
            width_box = []
            not_hollow_box = []
            length_line = []
            for box, line in b_l:
                pos_box.append(box.state.pos)
                pos_line.append(line.state.pos)
                rot_box.append(box.state.rot)
                rot_line.append(line.state.rot)
                length_box.append(box.shape.length)
                width_box.append(box.shape.width)
                not_hollow_box.append(not box.shape.hollow)
                length_line.append(line.shape.length)
            pos_box = jnp.stack(pos_box, axis=-2)
            pos_line = jnp.stack(pos_line, axis=-2)
            rot_box = jnp.stack(rot_box, axis=-2)
            rot_line = jnp.stack(rot_line, axis=-2)
            length_box = jnp.broadcast_to(
                jnp.stack(
                    length_box,
                    axis=-1,
                )[None],
                (self.batch_dim, -1),
            )
            width_box = jnp.broadcast_to(
                jnp.stack(
                    width_box,
                    axis=-1,
                )[None],
                (self.batch_dim, -1),
            )
            not_hollow_box_prior = jnp.stack(
                not_hollow_box,
                axis=-1,
            )
            not_hollow_box = jnp.broadcast_to(
                not_hollow_box_prior[None],
                (self.batch_dim, -1),
            )
            length_line = jnp.broadcast_to(
                jnp.stack(
                    length_line,
                    axis=-1,
                )[None],
                (self.batch_dim, -1),
            )

            point_box, point_line = _get_closest_line_box(
                pos_box,
                rot_box,
                width_box,
                length_box,
                pos_line,
                rot_line,
                length_line,
            )

            inner_point_box = point_box
            d = jnp.zeros_like(length_line)
            if not_hollow_box_prior.any():
                inner_point_box_hollow, d_hollow = _get_inner_point_box(
                    point_line, point_box, pos_box
                )
                cond = jnp.broadcast_to(
                    not_hollow_box[..., None],
                    inner_point_box.shape,
                )
                inner_point_box = jnp.where(
                    cond, inner_point_box_hollow, inner_point_box
                )
                d = jnp.where(not_hollow_box, d_hollow, d)

            force_box, force_line = self._get_constraint_forces(
                inner_point_box,
                point_line,
                dist_min=LINE_MIN_DIST + d,
                force_multiplier=self._collision_force,
            )
            r_box = point_box - pos_box
            r_line = point_line - pos_line

            torque_box = JaxUtils.compute_torque(force_box, r_box)
            torque_line = JaxUtils.compute_torque(force_line, r_line)

            for i, (entity_a, entity_b) in enumerate(b_l):
                self.update_env_forces(
                    entity_a,
                    force_box[:, i],
                    torque_box[:, i],
                    entity_b,
                    force_line[:, i],
                    torque_line[:, i],
                )

    def _box_box_vectorized_collision(self, b_b):
        if len(b_b):
            pos_box = []
            pos_box2 = []
            rot_box = []
            rot_box2 = []
            length_box = []
            width_box = []
            not_hollow_box = []
            length_box2 = []
            width_box2 = []
            not_hollow_box2 = []
            for box, box2 in b_b:
                pos_box.append(box.state.pos)
                rot_box.append(box.state.rot)
                length_box.append(box.shape.length)
                width_box.append(box.shape.width)
                not_hollow_box.append(not box.shape.hollow)
                pos_box2.append(box2.state.pos)
                rot_box2.append(box2.state.rot)
                length_box2.append(box2.shape.length)
                width_box2.append(box2.shape.width)
                not_hollow_box2.append(not box2.shape.hollow)

            pos_box = jnp.stack(pos_box, axis=-2)
            rot_box = jnp.stack(rot_box, axis=-2)
            length_box = jnp.broadcast_to(
                jnp.stack(
                    length_box,
                    axis=-1,
                )[None],
                (self.batch_dim, -1),
            )
            width_box = jnp.broadcast_to(
                jnp.stack(
                    width_box,
                    axis=-1,
                )[None],
                (self.batch_dim, -1),
            )
            not_hollow_box_prior = jnp.stack(
                not_hollow_box,
                axis=-1,
            )
            not_hollow_box = not_hollow_box_prior.unsqueeze(0).expand(
                self.batch_dim, -1
            )
            pos_box2 = jnp.stack(pos_box2, axis=-2)
            rot_box2 = jnp.stack(rot_box2, axis=-2)
            length_box2 = jnp.broadcast_to(
                jnp.stack(
                    length_box2,
                    dim=-1,
                )[None],
                (self.batch_dim, -1),
            )
            width_box2 = jnp.broadcast_to(
                jnp.stack(
                    width_box2,
                    dim=-1,
                )[None],
                (self.batch_dim, -1),
            )
            not_hollow_box2_prior = jnp.stack(
                not_hollow_box2,
                axis=-1,
            )
            not_hollow_box2 = not_hollow_box2_prior.unsqueeze(0).expand(
                self.batch_dim, -1
            )

            point_a, point_b = _get_closest_box_box(
                pos_box,
                rot_box,
                width_box,
                length_box,
                pos_box2,
                rot_box2,
                width_box2,
                length_box2,
            )

            inner_point_a = point_a
            d_a = jnp.zeros_like(length_box)
            if not_hollow_box_prior.any():
                inner_point_box_hollow, d_hollow = _get_inner_point_box(
                    point_b, point_a, pos_box
                )
                cond = jnp.broadcast_to(
                    not_hollow_box[..., None],
                    inner_point_a.shape,
                )
                inner_point_a = jnp.where(cond, inner_point_box_hollow, inner_point_a)
                d_a = jnp.where(not_hollow_box, d_hollow, d_a)

            inner_point_b = point_b
            d_b = jnp.zeros_like(length_box2)
            if not_hollow_box2_prior.any():
                inner_point_box2_hollow, d_hollow2 = _get_inner_point_box(
                    point_a, point_b, pos_box2
                )
                cond = jnp.broadcast_to(
                    not_hollow_box2[..., None],
                    inner_point_b.shape,
                )
                inner_point_b = jnp.where(cond, inner_point_box2_hollow, inner_point_b)
                d_b = jnp.where(not_hollow_box2, d_hollow2, d_b)

            force_a, force_b = self._get_constraint_forces(
                inner_point_a,
                inner_point_b,
                dist_min=d_a + d_b + LINE_MIN_DIST,
                force_multiplier=self._collision_force,
            )
            r_a = point_a - pos_box
            r_b = point_b - pos_box2
            torque_a = JaxUtils.compute_torque(force_a, r_a)
            torque_b = JaxUtils.compute_torque(force_b, r_b)

            for i, (entity_a, entity_b) in enumerate(b_b):
                self.update_env_forces(
                    entity_a,
                    force_a[:, i],
                    torque_a[:, i],
                    entity_b,
                    force_b[:, i],
                    torque_b[:, i],
                )

    def collides(self, a: Entity, b: Entity) -> bool:
        if (not a.collides(b)) or (not b.collides(a)) or a is b:
            return False
        a_shape = a.shape
        b_shape = b.shape
        if not a.movable and not a.rotatable and not b.movable and not b.rotatable:
            return False
        if {a_shape.__class__, b_shape.__class__} not in self._collidable_pairs:
            return False
        if not (
            jnp.linalg.vector_norm(a.state.pos - b.state.pos, axis=-1)
            <= a.shape.circumscribed_radius() + b.shape.circumscribed_radius()
        ).any():
            return False

        return True

    def _get_constraint_forces(
        self,
        pos_a: Array,
        pos_b: Array,
        dist_min: float,
        force_multiplier: float,
        attractive: bool = False,
    ) -> tuple[Array, Array]:
        min_dist = 1e-6
        delta_pos = pos_a - pos_b
        dist = jnp.linalg.vector_norm(delta_pos, axis=-1)
        sign = -1 if attractive else 1

        # softmax penetration
        k = self._contact_margin
        penetration = (
            jnp.logaddexp(
                jnp.array(0.0, dtype=jnp.float32),
                (dist_min - dist) * sign / k,
            )
            * k
        )
        force = (
            sign
            * force_multiplier
            * delta_pos
            / jnp.where(dist > 0, dist, 1e-8)[..., None]
            * penetration[..., None]
        )
        force = jnp.where((dist < min_dist)[..., None], 0.0, force)
        if not attractive:
            force = jnp.where((dist > dist_min)[..., None], 0.0, force)
        else:
            force = jnp.where((dist < dist_min)[..., None], 0.0, force)
        return force, -force

    def _get_constraint_torques(
        self,
        rot_a: Array,
        rot_b: Array,
        force_multiplier: float = TORQUE_CONSTRAINT_FORCE,
    ) -> tuple[Array, Array]:
        min_delta_rot = 1e-9
        delta_rot = rot_a - rot_b
        abs_delta_rot = jnp.linalg.vector_norm(delta_rot, axis=-1)[..., None]

        # softmax penetration
        k = 1
        penetration = k * (jnp.exp(abs_delta_rot / k) - 1)

        torque = force_multiplier * delta_rot.sign() * penetration
        torque = jnp.where((abs_delta_rot < min_delta_rot), 0.0, torque)

        return -torque, torque

    # integrate physical state
    # uses semi-implicit euler with sub-stepping
    def _integrate_state(
        self,
        entity: Entity,
        substep: int,
        forces_dict: dict[Entity, Array],
        torques_dict: dict[Entity, Array],
    ):
        if entity.movable:
            # Compute translation
            if substep == 0:
                if entity.drag is not None:
                    entity = entity.replace(
                        state=entity.state.replace(
                            vel=entity.state.vel * (1 - entity.drag)
                        )
                    )
                else:
                    entity = entity.replace(
                        state=entity.state.replace(
                            vel=entity.state.vel * (1 - self._drag)
                        )
                    )
            accel = forces_dict[entity] / entity.mass
            entity = entity.replace(
                state=entity.state.replace(vel=entity.state.vel + accel * self._sub_dt)
            )
            if entity.max_speed is not None:
                entity = entity.replace(
                    state=entity.state.replace(
                        vel=JaxUtils.clamp_with_norm(entity.state.vel, entity.max_speed)
                    )
                )
            if entity.v_range is not None:
                entity = entity.replace(
                    state=entity.state.replace(
                        vel=jnp.clip(entity.state.vel, -entity.v_range, entity.v_range)
                    )
                )
            new_pos = entity.state.pos + entity.state.vel * self._sub_dt
            entity = entity.replace(
                state=entity.state.replace(
                    pos=jnp.stack(
                        [
                            (
                                new_pos[..., X].clip(-self._x_semidim, self._x_semidim)
                                if self._x_semidim is not None
                                else new_pos[..., X]
                            ),
                            (
                                new_pos[..., Y].clip(-self._y_semidim, self._y_semidim)
                                if self._y_semidim is not None
                                else new_pos[..., Y]
                            ),
                        ],
                        axis=-1,
                    )
                )
            )

        if entity.rotatable:
            # Compute rotation
            if substep == 0:
                if entity.drag is not None:
                    entity = entity.replace(
                        state=entity.state.replace(
                            ang_vel=entity.state.ang_vel * (1 - entity.drag)
                        )
                    )
                else:
                    entity = entity.replace(
                        state=entity.state.replace(
                            ang_vel=entity.state.ang_vel * (1 - self._drag)
                        )
                    )
            entity = entity.replace(
                state=entity.state.replace(
                    ang_vel=entity.state.ang_vel
                    + (torques_dict[entity] / entity.moment_of_inertia) * self._sub_dt
                )
            )
            entity = entity.replace(
                state=entity.state.replace(
                    rot=entity.state.rot + entity.state.ang_vel * self._sub_dt
                )
            )

        return entity

    def _update_comm_state(self, agent: "Agent"):
        # set communication state (directly for now)
        if not agent.silent:
            agent = agent.replace(c=agent.action.c)
        return agent
