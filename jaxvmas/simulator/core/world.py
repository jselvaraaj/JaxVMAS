import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Callable
from jaxtyping import Array, Bool, jaxtyped

from jaxvmas.simulator.core.agent import Agent
from jaxvmas.simulator.core.entity import Entity
from jaxvmas.simulator.core.jax_vectorized_object import JaxVectorizedObject
from jaxvmas.simulator.core.landmark import Landmark
from jaxvmas.simulator.core.ray_physics import (
    cast_ray_to_box,
    cast_ray_to_line,
    cast_ray_to_sphere,
    cast_rays_to_box,
    cast_rays_to_line,
    cast_rays_to_sphere,
)
from jaxvmas.simulator.core.shapes import Box, Line, Shape, Sphere
from jaxvmas.simulator.core.states import EntityState
from jaxvmas.simulator.joints import Joint, JointConstraint
from jaxvmas.simulator.physics import (
    _get_closest_box_box,
    _get_closest_line_box,
    _get_closest_point_box,
    _get_closest_point_line,
    _get_closest_points_line_line,
    _get_inner_point_box,
)
from jaxvmas.simulator.utils import (
    ANGULAR_FRICTION,
    COLLISION_FORCE,
    DRAG,
    JOINT_FORCE,
    LINE_MIN_DIST,
    LINEAR_FRICTION,
    TORQUE_CONSTRAINT_FORCE,
    JaxUtils,
    X,
    Y,
)


# Multi-agent world
@jaxtyped(typechecker=beartype)
class World(JaxVectorizedObject):

    agents: list[Agent]
    landmarks: list[Landmark]
    x_semidim: float
    y_semidim: float
    dim_p: int
    dim_c: int
    dt: float
    substeps: int
    sub_dt: float
    drag: float
    gravity: Array
    linear_friction: float
    angular_friction: float
    collision_force: float
    joint_force: float
    torque_constraint_force: float
    contact_margin: float
    torque_constraint_force: float
    _joints: dict[frozenset[str], JointConstraint]
    collidable_pairs: list[tuple[type[Shape], type[Shape]]]
    entity_index_map: dict[str, int]

    force_dict: dict[str, Array]
    torque_dict: dict[str, Array]

    @classmethod
    def create(
        cls,
        batch_dim: int,
        dt: float = 0.1,
        substeps: int = 1,  # if you use joints, higher this value to gain simulation stability
        drag: float = DRAG,
        linear_friction: float = LINEAR_FRICTION,
        angular_friction: float = ANGULAR_FRICTION,
        x_semidim: float = jnp.nan,
        y_semidim: float = jnp.nan,
        dim_c: int = 0,
        collision_force: float = COLLISION_FORCE,
        joint_force: float = JOINT_FORCE,
        torque_constraint_force: float = TORQUE_CONSTRAINT_FORCE,
        contact_margin: float = 1e-3,
        gravity: tuple[float, float] = (0.0, 0.0),
        **kwargs,
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
            (Sphere, Sphere),
            (Sphere, Box),
            (Sphere, Line),
            (Line, Line),
            (Line, Box),
            (Box, Box),
        ]
        # Map to save entity indexes
        entity_index_map = {}
        _force_dict = {}
        _torque_dict = {}

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
            _force_dict,
            _torque_dict,
        )

    @jaxtyped(typechecker=beartype)
    def add_agent(
        self,
        agent: Agent,
    ):
        """Only way to add agents to the world"""
        agent = agent.replace(batch_dim=self.batch_dim)
        agent = agent._spawn(dim_c=self.dim_c, dim_p=self.dim_p)

        self = self.replace(agents=self.agents + [agent])
        return self

    @jaxtyped(typechecker=beartype)
    def add_landmark(
        self,
        landmark: Landmark,
    ):
        """Only way to add landmarks to the world"""
        landmark = landmark.replace(batch_dim=self.batch_dim)
        landmark = landmark._spawn(dim_p=self.dim_p)
        self = self.replace(landmarks=self.landmarks + [landmark])
        return self

    @jaxtyped(typechecker=beartype)
    def add_joint(self, joint: Joint):
        assert self.substeps > 1, "For joints, world substeps needs to be more than 1"
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

    @jaxtyped(typechecker=beartype)
    def reset(self, env_index: int | float = jnp.nan):
        entities = []
        for e in self.entities:
            entities.append(e._reset(env_index))

        self = self.replace(entities=entities)

        _entity_index_map = {e.name: i for i, e in enumerate(self.entities)}
        self = self.replace(entity_index_map=_entity_index_map)

        forces_dict = {
            e.name: jnp.zeros((self.batch_dim, self.dim_p)) for e in self.entities
        }
        torques_dict = {e.name: jnp.zeros((self.batch_dim, 1)) for e in self.entities}
        self = self.replace(force_dict=forces_dict, torque_dict=torques_dict)

        return self

    @property
    def joints(self):
        return self._joints.values()

    # return all entities in the world
    @property
    def entities(self) -> list[Entity]:
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self) -> list[Agent]:
        return [agent for agent in self.agents if not agent.is_scripted_agent]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self) -> list[Agent]:
        return [agent for agent in self.agents if agent.is_scripted_agent]

    @jaxtyped(typechecker=beartype)
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
                d = cast_ray_to_box(e, e_state, pos, angles, max_range)
            elif isinstance(e.shape, Sphere):
                d = cast_ray_to_sphere(e, e_state, pos, angles, max_range)
            elif isinstance(e.shape, Line):
                d = cast_ray_to_line(e, e_state, pos, angles, max_range)
            else:
                raise RuntimeError(f"Shape {e.shape} currently not handled by cast_ray")
            dists.append(d)
        dist, _ = jnp.min(jnp.stack(dists, dim=-1), dim=-1)
        return dist

    @jaxtyped(typechecker=beartype)
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
            dist_boxes = cast_rays_to_box(
                self.batch_dim,
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

            dist_spheres = cast_rays_to_sphere(
                self.batch_dim,
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

            dist_lines = cast_rays_to_line(
                self.batch_dim,
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

    @jaxtyped(typechecker=beartype)
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

    @jaxtyped(typechecker=beartype)
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

    @jaxtyped(typechecker=beartype)
    def is_overlapping(
        self,
        entity_a: Entity,
        entity_b: Entity,
        env_index: int = None,
    ):
        a_shape = entity_a.shap
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
            box, sphere = (
                (entity_a, entity_b)
                if isinstance(entity_b.shape, Sphere)
                else (entity_b, entity_a)
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

    def replace(self, **kwargs):
        if "entities" in kwargs:
            num_agents = len(self.agents)
            entities = kwargs.pop("entities")
            kwargs["agents"] = entities[:num_agents]
            kwargs["landmarks"] = entities[num_agents:]
        elif "policy_agents" in kwargs:
            policy_agents = kwargs.pop("policy_agents")
            scripted_agents = self.scripted_agents
            kwargs["_agents"] = policy_agents + scripted_agents
        return super().replace(**kwargs)

    # update state of the world
    @eqx.filter_jit
    def step(self):
        print("starting step")
        for substep in range(self.substeps):
            # Initialize force and torque dictionaries
            forces_dict = {
                e.name: jnp.zeros((self.batch_dim, self.dim_p)) for e in self.entities
            }
            torques_dict = {
                e.name: jnp.zeros((self.batch_dim, 1)) for e in self.entities
            }
            self = self.replace(force_dict=forces_dict, torque_dict=torques_dict)

            print("starting apply_action_force")

            # Process agents
            if len(self.agents) > 0:
                agent_carry = (self, 0)
                agent_dynamic_carry, agent_static_carry = eqx.partition(
                    agent_carry, eqx.is_array
                )

                # Process agents first
                def _process_agent(dynamic_carry, unused):
                    carry: tuple[World, int] = eqx.combine(
                        agent_static_carry, dynamic_carry
                    )
                    world, i = carry

                    agent = world.agents[i]
                    # apply agent force controls
                    agent, world = world._apply_action_force(agent)
                    # apply agent torque controls
                    agent, world = world._apply_action_torque(agent)
                    # apply friction
                    agent, world = world._apply_friction_force(agent)
                    # apply gravity
                    agent, world = world._apply_gravity(agent)

                    world = world.replace(
                        agents=world.agents[:i] + [agent] + world.agents[i + 1 :]
                    )

                    _agent_carry = (world, i + 1)
                    _agent_dynamic_carry, _ = eqx.partition(_agent_carry, eqx.is_array)

                    return _agent_dynamic_carry, None

                _agent_dynamic_carry, _ = jax.lax.scan(
                    _process_agent, agent_dynamic_carry, None, length=len(self.agents)
                )
                self, _ = eqx.combine(agent_static_carry, _agent_dynamic_carry)
                self = self

            # Process landmarks
            if len(self.landmarks) > 0:
                landmark_carry = (self, 0)
                landmark_dynamic_carry, landmark_static_carry = eqx.partition(
                    landmark_carry, eqx.is_array
                )

                # Process landmarks separately
                def _process_landmark(dynamic_carry, unused):
                    carry: tuple[World, int] = eqx.combine(
                        landmark_static_carry, dynamic_carry
                    )
                    world, i = carry
                    landmark = world.landmarks[i]
                    # apply friction
                    landmark, world = world._apply_friction_force(landmark)
                    # apply gravity
                    landmark, world = world._apply_gravity(landmark)
                    world = world.replace(
                        landmarks=world.landmarks[:i]
                        + [landmark]
                        + world.landmarks[i + 1 :]
                    )
                    _landmark_carry = (world, i + 1)
                    _landmark_dynamic_carry, _ = eqx.partition(
                        _landmark_carry, eqx.is_array
                    )

                    return _landmark_dynamic_carry, None

                _landmark_dynamic_carry, _ = jax.lax.scan(
                    _process_landmark,
                    landmark_dynamic_carry,
                    None,
                    length=len(self.landmarks),
                )
                self, _ = eqx.combine(landmark_static_carry, _landmark_dynamic_carry)
                self = self

            self = self._apply_vectorized_enviornment_force()
            print("ending apply_vectorized_enviornment_force")

            entities = []
            print("starting integrate_state")
            for entity in self.entities:
                # integrate physical state
                entity = self._integrate_state(entity, substep)
                entities.append(entity)
            print("ending integrate_state")
            self = self.replace(entities=entities)

            # Update joint states after entity states have been updated
            print("starting update_joints")
            new_joints = {}
            for joint_key, joint in self._joints.items():
                entity_a = self.entities[self.entity_index_map[joint.entity_a.name]]
                entity_b = self.entities[self.entity_index_map[joint.entity_b.name]]
                updated_joint = joint.update_joint_state(entity_a, entity_b)
                new_joints[joint_key] = updated_joint
            self = self.replace(_joints=new_joints)
            print("ending update_joints")
        # update non-differentiable comm state
        if self.dim_c > 0:
            print("starting update_comm_state")
            agents = []
            for agent in self.agents:
                agents.append(self._update_comm_state(agent))
            self = self.replace(agents=agents)
            print("ending update_comm_state")

        print("ending step")
        return self

    # gather agent action forces
    @jaxtyped(typechecker=beartype)
    def _apply_action_force(self, agent: Agent):
        forces_dict = {**self.force_dict}
        if agent.movable:
            force = agent.state.force
            force = jnp.where(
                ~jnp.isnan(agent.max_f),
                JaxUtils.clamp_with_norm(force, agent.max_f),
                force,
            )
            force = jnp.where(
                ~jnp.isnan(agent.f_range),
                jnp.clip(force, -agent.f_range, agent.f_range),
                force,
            )
            forces_dict[agent.name] = forces_dict[agent.name] + force
            agent = agent.replace(state=agent.state.replace(force=force))
        return agent, self.replace(force_dict=forces_dict)

    @jaxtyped(typechecker=beartype)
    def _apply_action_torque(self, agent: Agent):
        torques_dict = {**self.torque_dict}
        if agent.rotatable:
            torque = agent.state.torque
            torque = jnp.where(
                ~jnp.isnan(agent.max_t),
                JaxUtils.clamp_with_norm(torque, agent.max_t),
                torque,
            )
            torque = jnp.where(
                ~jnp.isnan(agent.t_range),
                jnp.clip(torque, -agent.t_range, agent.t_range),
                torque,
            )

            torques_dict[agent.name] = torques_dict[agent.name] + torque
            agent = agent.replace(state=agent.state.replace(torque=torque))
        return agent, self.replace(torque_dict=torques_dict)

    @jaxtyped(typechecker=beartype)
    def _apply_gravity(
        self,
        entity: Entity,
    ):
        forces_dict = {**self.force_dict}
        if entity.movable:
            gravity_force = entity.mass * self.gravity
            forces_dict[entity.name] = forces_dict[entity.name] + gravity_force
            gravity_force = jnp.where(
                ~jnp.isnan(entity.gravity),
                entity.mass * entity.gravity,
                jnp.zeros_like(entity.gravity),
            )
            forces_dict[entity.name] = forces_dict[entity.name] + gravity_force
        return entity, self.replace(force_dict=forces_dict)

    @jaxtyped(typechecker=beartype)
    def _apply_friction_force(
        self,
        entity: Entity,
    ):
        @jaxtyped(typechecker=beartype)
        def get_friction_force(
            vel: Array, coeff: float | Array, force: Array, mass: float
        ):
            speed = jnp.linalg.vector_norm(vel, axis=-1)
            static = speed == 0
            static_exp = jnp.broadcast_to(static[..., None], vel.shape)

            coeff = jnp.where(jnp.isscalar(coeff), jnp.full_like(force, coeff), coeff)
            coeff = jnp.broadcast_to(coeff, force.shape)

            friction_force_constant = coeff * mass

            friction_force = -(
                vel
                / jnp.broadcast_to(jnp.where(static, 1e-8, speed)[..., None], vel.shape)
            ) * jnp.minimum(
                friction_force_constant, (jnp.abs(vel) / self.sub_dt) * mass
            )
            friction_force = jnp.where(static_exp, 0.0, friction_force)

            return friction_force

        forces_dict = {**self.force_dict}
        torques_dict = {**self.torque_dict}

        friction_force = jnp.where(
            ~jnp.isnan(entity.linear_friction),
            get_friction_force(
                entity.state.vel,
                entity.linear_friction,
                forces_dict[entity.name],
                entity.mass,
            ),
            0.0,
        )
        forces_dict[entity.name] = forces_dict[entity.name] + friction_force

        friction_torque = jnp.where(
            ~jnp.isnan(entity.angular_friction),
            get_friction_force(
                entity.state.ang_vel,
                entity.angular_friction,
                torques_dict[entity.name],
                entity.moment_of_inertia,
            ),
            0.0,
        )
        torques_dict[entity.name] = torques_dict[entity.name] + friction_torque

        return entity, self.replace(force_dict=forces_dict, torque_dict=torques_dict)

    @jaxtyped(typechecker=beartype)
    def _apply_vectorized_enviornment_force(self):
        s_s = []
        l_s = []
        b_s = []
        l_l = []
        b_l = []
        b_b = []
        joints = []
        collision_mask_s_s = []
        collision_mask_l_s = []
        collision_mask_l_l = []
        collision_mask_b_s = []
        collision_mask_b_l = []
        collision_mask_b_b = []
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
                _collision = self.collides(entity_a, entity_b)
                if isinstance(entity_a.shape, Sphere) and isinstance(
                    entity_b.shape, Sphere
                ):
                    s_s.append((entity_a, entity_b))
                    collision_mask_s_s.append(_collision)
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
                    collision_mask_l_s.append(_collision)
                elif isinstance(entity_a.shape, Line) and isinstance(
                    entity_b.shape, Line
                ):
                    l_l.append((entity_a, entity_b))
                    collision_mask_l_l.append(_collision)
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
                    collision_mask_b_s.append(_collision)
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
                    collision_mask_b_l.append(_collision)
                elif isinstance(entity_a.shape, Box) and isinstance(
                    entity_b.shape, Box
                ):
                    b_b.append((entity_a, entity_b))
                    collision_mask_b_b.append(_collision)
                else:
                    raise AssertionError()

        collision_mask_s_s = jnp.asarray(collision_mask_s_s)
        collision_mask_l_s = jnp.asarray(collision_mask_l_s)
        collision_mask_l_l = jnp.asarray(collision_mask_l_l)
        collision_mask_b_s = jnp.asarray(collision_mask_b_s)
        collision_mask_b_l = jnp.asarray(collision_mask_b_l)
        collision_mask_b_b = jnp.asarray(collision_mask_b_b)
        # Joints
        self = self._vectorized_joint_constraints(joints)

        # Sphere and sphere
        self = self._sphere_sphere_vectorized_collision(s_s, collision_mask_s_s)
        # Line and sphere
        self = self._sphere_line_vectorized_collision(l_s, collision_mask_l_s)
        # Line and line
        self = self._line_line_vectorized_collision(l_l, collision_mask_l_l)
        # Box and sphere
        self = self._box_sphere_vectorized_collision(b_s, collision_mask_b_s)
        # Box and line
        self = self._box_line_vectorized_collision(b_l, collision_mask_b_l)
        # Box and box
        self = self._box_box_vectorized_collision(b_b, collision_mask_b_b)

        return self

    @jaxtyped(typechecker=beartype)
    def update_env_forces(
        self,
        entity_a: Entity,
        f_a: Array,
        t_a: Array,
        entity_b: Entity,
        f_b: Array,
        t_b: Array,
    ):
        new_forces_dict = dict(self.force_dict)
        new_torques_dict = dict(self.torque_dict)

        if entity_a.movable:
            new_forces_dict[entity_a.name] = self.force_dict[entity_a.name] + f_a
        if entity_a.rotatable:
            new_torques_dict[entity_a.name] = self.torque_dict[entity_a.name] + t_a
        if entity_b.movable:
            new_forces_dict[entity_b.name] = self.force_dict[entity_b.name] + f_b
        if entity_b.rotatable:
            new_torques_dict[entity_b.name] = self.torque_dict[entity_b.name] + t_b

        return self.replace(force_dict=new_forces_dict, torque_dict=new_torques_dict)

    @jaxtyped(typechecker=beartype)
    def _vectorized_joint_constraints(
        self,
        joints: list[JointConstraint],
    ):
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
            dist = jnp.stack(
                dist,
                axis=-1,
            )[None]
            dist = jnp.broadcast_to(
                dist,
                (self.batch_dim, dist.shape[-1]),
            )
            rotate_prior = jnp.stack(
                rotate,
                axis=-1,
            )[None]
            rotate = jnp.broadcast_to(
                rotate_prior,
                (self.batch_dim, rotate_prior.shape[-1]),
            )[..., None]
            joint_rot = jnp.stack(joint_rot, axis=-2)

            (
                force_a_attractive,
                force_b_attractive,
            ) = self._get_constraint_forces(
                pos_joint_a,
                pos_joint_b,
                dist_min=dist,
                attractive=True,
                force_multiplier=self.joint_force,
            )
            force_a_repulsive, force_b_repulsive = self._get_constraint_forces(
                pos_joint_a,
                pos_joint_b,
                dist_min=dist,
                attractive=False,
                force_multiplier=self.joint_force,
            )
            force_a = force_a_attractive + force_a_repulsive
            force_b = force_b_attractive + force_b_repulsive
            r_a = pos_joint_a - pos_a
            r_b = pos_joint_b - pos_b

            torque_a_rotate = JaxUtils.compute_torque(force_a, r_a)
            torque_b_rotate = JaxUtils.compute_torque(force_b, r_b)

            torque_a_fixed, torque_b_fixed = self._get_constraint_torques(
                rot_a,
                rot_b + joint_rot,
                force_multiplier=self.torque_constraint_force,
            )

            torque_a = jnp.where(
                rotate, torque_a_rotate, torque_a_rotate + torque_a_fixed
            )
            torque_b = jnp.where(
                rotate, torque_b_rotate, torque_b_rotate + torque_b_fixed
            )

            for i, joint in enumerate(joints):
                self = self.update_env_forces(
                    joint.entity_a,
                    force_a[:, i],
                    torque_a[:, i],
                    joint.entity_b,
                    force_b[:, i],
                    torque_b[:, i],
                )

        return self

    @jaxtyped(typechecker=beartype)
    def _sphere_sphere_vectorized_collision(
        self,
        s_s: list[tuple[Entity, Entity]],
        collision_mask: Array,
    ):
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

            pos_s_a = jnp.stack(pos_s_a, axis=-2)  # [batch_dim, n_pairs, 2]
            pos_s_b = jnp.stack(pos_s_b, axis=-2)  # [batch_dim, n_pairs, 2]
            collision_mask = collision_mask[None, :, None]
            radius_s_a = jnp.broadcast_to(
                jnp.expand_dims(jnp.asarray(radius_s_a), 0), (self.batch_dim, len(s_s))
            )  # [batch_dim, n_pairs]
            radius_s_b = jnp.broadcast_to(
                jnp.expand_dims(jnp.asarray(radius_s_b), 0), (self.batch_dim, len(s_s))
            )  # [batch_dim, n_pairs]

            force_multiplier = jnp.where(collision_mask, self.collision_force, 0)
            force_a, force_b = self._get_constraint_forces(
                pos_s_a,
                pos_s_b,
                dist_min=radius_s_a + radius_s_b,
                force_multiplier=force_multiplier,
            )

            for i, (entity_a, entity_b) in enumerate(s_s):
                self = self.update_env_forces(
                    entity_a,
                    force_a[:, i],
                    jnp.asarray(0.0),
                    entity_b,
                    force_b[:, i],
                    jnp.asarray(0.0),
                )

        return self

    @jaxtyped(typechecker=beartype)
    def _sphere_line_vectorized_collision(
        self,
        l_s: list[tuple[Entity, Entity]],
        collision_mask: Array,
    ):
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
            collision_mask = collision_mask[None, :, None]

            radius_s = jnp.broadcast_to(
                jnp.stack(
                    radius_s,
                    axis=-1,
                )[None],
                (self.batch_dim, len(l_s)),
            )
            length_l = jnp.broadcast_to(
                jnp.stack(
                    length_l,
                    axis=-1,
                )[None],
                (self.batch_dim, len(l_s)),
            )

            closest_point = _get_closest_point_line(pos_l, rot_l, length_l, pos_s)
            force_multiplier = jnp.where(collision_mask, self.collision_force, 0)
            force_sphere, force_line = self._get_constraint_forces(
                pos_s,
                closest_point,
                dist_min=radius_s + LINE_MIN_DIST,
                force_multiplier=force_multiplier,
            )
            r = closest_point - pos_l
            torque_line = JaxUtils.compute_torque(force_line, r)

            for i, (entity_a, entity_b) in enumerate(l_s):
                self = self.update_env_forces(
                    entity_a,
                    force_line[:, i],
                    torque_line[:, i],
                    entity_b,
                    force_sphere[:, i],
                    jnp.asarray(0.0),
                )

        return self

    @jaxtyped(typechecker=beartype)
    def _line_line_vectorized_collision(
        self,
        l_l: list[tuple[Entity, Entity]],
        collision_mask: Array,
    ):
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
            collision_mask = collision_mask[None, :, None]

            length_l_a = jnp.broadcast_to(
                jnp.stack(
                    length_l_a,
                    axis=-1,
                )[None],
                (self.batch_dim, len(l_l)),
            )
            length_l_b = jnp.broadcast_to(
                jnp.stack(
                    length_l_b,
                    axis=-1,
                )[None],
                (self.batch_dim, len(l_l)),
            )

            point_a, point_b = _get_closest_points_line_line(
                pos_l_a,
                rot_l_a,
                length_l_a,
                pos_l_b,
                rot_l_b,
                length_l_b,
            )
            force_multiplier = jnp.where(collision_mask, self.collision_force, 0)
            force_a, force_b = self._get_constraint_forces(
                point_a,
                point_b,
                dist_min=LINE_MIN_DIST,
                force_multiplier=force_multiplier,
            )
            r_a = point_a - pos_l_a
            r_b = point_b - pos_l_b

            torque_a = JaxUtils.compute_torque(force_a, r_a)
            torque_b = JaxUtils.compute_torque(force_b, r_b)
            for i, (entity_a, entity_b) in enumerate(l_l):
                self = self.update_env_forces(
                    entity_a,
                    force_a[:, i],
                    torque_a[:, i],
                    entity_b,
                    force_b[:, i],
                    torque_b[:, i],
                )

        return self

    @jaxtyped(typechecker=beartype)
    def _box_sphere_vectorized_collision(
        self,
        b_s: list[tuple[Entity, Entity]],
        collision_mask: Array,
    ):
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
            collision_mask = collision_mask[None, :, None]

            length_box = jnp.broadcast_to(
                jnp.stack(
                    length_box,
                    axis=-1,
                )[None],
                (self.batch_dim, len(b_s)),
            )
            width_box = jnp.broadcast_to(
                jnp.stack(
                    width_box,
                    axis=-1,
                )[None],
                (self.batch_dim, len(b_s)),
            )
            not_hollow_box_prior = jnp.stack(
                not_hollow_box,
                axis=-1,
            )
            not_hollow_box = jnp.broadcast_to(
                not_hollow_box_prior[None],
                (self.batch_dim, len(b_s)),
            )
            radius_sphere = jnp.broadcast_to(
                jnp.stack(
                    radius_sphere,
                    axis=-1,
                )[None],
                (self.batch_dim, len(b_s)),
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

            force_multiplier = jnp.where(collision_mask, self.collision_force, 0)
            force_sphere, force_box = self._get_constraint_forces(
                pos_sphere,
                inner_point_box,
                dist_min=radius_sphere + LINE_MIN_DIST + d,
                force_multiplier=force_multiplier,
            )
            r = closest_point_box - pos_box
            torque_box = JaxUtils.compute_torque(force_box, r)

            for i, (entity_a, entity_b) in enumerate(b_s):
                self = self.update_env_forces(
                    entity_a,
                    force_box[:, i],
                    torque_box[:, i],
                    entity_b,
                    force_sphere[:, i],
                    jnp.asarray(0.0),
                )

        return self

    @jaxtyped(typechecker=beartype)
    def _box_line_vectorized_collision(
        self,
        b_l: list[tuple[Entity, Entity]],
        collision_mask: Array,
    ):
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
            collision_mask = collision_mask[None, :, None]

            length_box = jnp.broadcast_to(
                jnp.stack(
                    length_box,
                    axis=-1,
                )[None],
                (self.batch_dim, len(b_l)),
            )
            width_box = jnp.broadcast_to(
                jnp.stack(
                    width_box,
                    axis=-1,
                )[None],
                (self.batch_dim, len(b_l)),
            )
            not_hollow_box_prior = jnp.stack(
                not_hollow_box,
                axis=-1,
            )
            not_hollow_box = jnp.broadcast_to(
                not_hollow_box_prior[None],
                (self.batch_dim, len(b_l)),
            )
            length_line = jnp.broadcast_to(
                jnp.stack(
                    length_line,
                    axis=-1,
                )[None],
                (self.batch_dim, len(b_l)),
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
            force_multiplier = jnp.where(collision_mask, self.collision_force, 0)
            force_box, force_line = self._get_constraint_forces(
                inner_point_box,
                point_line,
                dist_min=LINE_MIN_DIST + d,
                force_multiplier=force_multiplier,
            )
            r_box = point_box - pos_box
            r_line = point_line - pos_line

            torque_box = JaxUtils.compute_torque(force_box, r_box)
            torque_line = JaxUtils.compute_torque(force_line, r_line)

            for i, (entity_a, entity_b) in enumerate(b_l):
                self = self.update_env_forces(
                    entity_a,
                    force_box[:, i],
                    torque_box[:, i],
                    entity_b,
                    force_line[:, i],
                    torque_line[:, i],
                )

        return self

    @jaxtyped(typechecker=beartype)
    def _box_box_vectorized_collision(
        self,
        b_b: list[tuple[Entity, Entity]],
        collision_mask: Array,
    ):
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
            collision_mask = collision_mask[None, :, None]
            length_box = jnp.broadcast_to(
                jnp.stack(
                    length_box,
                    axis=-1,
                )[None],
                (self.batch_dim, len(b_b)),
            )
            width_box = jnp.broadcast_to(
                jnp.stack(
                    width_box,
                    axis=-1,
                )[None],
                (self.batch_dim, len(b_b)),
            )
            not_hollow_box_prior = jnp.stack(
                not_hollow_box,
                axis=-1,
            )
            not_hollow_box = jnp.broadcast_to(
                jnp.expand_dims(not_hollow_box_prior, 0),
                (self.batch_dim, len(b_b)),
            )
            pos_box2 = jnp.stack(pos_box2, axis=-2)
            rot_box2 = jnp.stack(rot_box2, axis=-2)
            length_box2 = jnp.broadcast_to(
                jnp.stack(
                    length_box2,
                    axis=-1,
                )[None],
                (self.batch_dim, len(b_b)),
            )
            width_box2 = jnp.broadcast_to(
                jnp.stack(
                    width_box2,
                    axis=-1,
                )[None],
                (self.batch_dim, len(b_b)),
            )
            not_hollow_box2_prior = jnp.stack(
                not_hollow_box2,
                axis=-1,
            )
            not_hollow_box2 = jnp.broadcast_to(
                jnp.expand_dims(not_hollow_box2_prior, 0),
                (self.batch_dim, len(b_b)),
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

            force_multiplier = jnp.where(collision_mask, self.collision_force, 0)
            force_a, force_b = self._get_constraint_forces(
                inner_point_a,
                inner_point_b,
                dist_min=d_a + d_b + LINE_MIN_DIST,
                force_multiplier=force_multiplier,
            )
            r_a = point_a - pos_box
            r_b = point_b - pos_box2
            torque_a = JaxUtils.compute_torque(force_a, r_a)
            torque_b = JaxUtils.compute_torque(force_b, r_b)

            for i, (entity_a, entity_b) in enumerate(b_b):
                self = self.update_env_forces(
                    entity_a,
                    force_a[:, i],
                    torque_a[:, i],
                    entity_b,
                    force_b[:, i],
                    torque_b[:, i],
                )

        return self

    @jaxtyped(typechecker=beartype)
    def collides(self, a: Entity, b: Entity) -> Bool[Array, ""]:
        # Early exit conditions
        collides_check = jnp.logical_and(a.collides(b), b.collides(a))
        same_entity = jnp.asarray(a.name == b.name)
        movable_check = jnp.logical_or(
            jnp.logical_or(a.movable, a.rotatable),
            jnp.logical_or(b.movable, b.rotatable),
        )

        # Shape collision check
        shape_pair = (a.shape.__class__, b.shape.__class__)
        shape_check = jnp.any(
            jnp.asarray([p == shape_pair for p in self.collidable_pairs])
        )

        # Distance check
        dist = jnp.linalg.norm(a.state.pos - b.state.pos, axis=-1)
        radius_sum = a.shape.circumscribed_radius() + b.shape.circumscribed_radius()
        dist_check = jnp.any(dist <= radius_sum)

        # Convert list of conditions to JAX array before reducing
        conditions = jnp.asarray(
            [
                collides_check,
                ~same_entity,
                movable_check,
                shape_check,
                dist_check,
            ]
        )

        # Combine all checks
        return jnp.all(conditions)  # Using all instead of logical_and.reduce

    @jaxtyped(typechecker=beartype)
    def _get_constraint_forces(
        self,
        pos_a: Array,
        pos_b: Array,
        dist_min: Array,
        force_multiplier: Array | float,
        attractive: bool = False,
    ) -> tuple[Array, Array]:
        min_dist = 1e-6
        delta_pos = pos_a - pos_b
        dist = jnp.linalg.vector_norm(delta_pos, axis=-1)
        sign = -1 if attractive else 1

        k = self.contact_margin
        # Calculate penetration using safe distance
        penetration = (
            jnp.logaddexp(
                jnp.array(0.0),
                (dist_min - dist) * sign / k,
            )
            * k
        )
        # Handle zero-distance cases
        safe_delta = delta_pos / jnp.where(dist > 0, dist, 1e-8)[..., None]

        # Calculate force using safe direction vector
        force = sign * force_multiplier * safe_delta * penetration[..., None]

        force = jnp.where(
            (dist < min_dist)[..., None],
            jnp.zeros_like(force),
            force,
        )

        # Apply force only when needed
        force = jnp.where(
            (
                (dist > dist_min)[..., None]
                if not attractive
                else (dist < dist_min)[..., None]
            ),
            jnp.zeros_like(force),
            force,
        )

        return force, -force

    @jaxtyped(typechecker=beartype)
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

        torque = force_multiplier * jnp.sign(delta_rot) * penetration
        torque = jnp.where((abs_delta_rot < min_delta_rot), 0.0, torque)

        return -torque, torque

    # integrate physical state
    # uses semi-implicit euler with sub-stepping
    @jaxtyped(typechecker=beartype)
    def _integrate_state(
        self,
        entity: Entity,
        substep: int,
    ):
        if entity.movable:
            vel = entity.state.vel
            # Compute translation
            if substep == 0:
                vel = jnp.where(
                    ~jnp.isnan(entity.drag),
                    vel * (1 - entity.drag),
                    vel * (1 - self.drag),
                )
            accel = self.force_dict[entity.name] / entity.mass
            vel = vel + accel * self.sub_dt
            vel = jnp.where(
                ~jnp.isnan(entity.max_speed),
                JaxUtils.clamp_with_norm(vel, entity.max_speed),
                vel,
            )
            vel = jnp.where(
                ~jnp.isnan(entity.v_range),
                jnp.clip(vel, -entity.v_range, entity.v_range),
                vel,
            )
            new_pos = entity.state.pos + vel * self.sub_dt
            # Apply boundary conditions
            new_pos_x = jnp.where(
                ~jnp.isnan(self.x_semidim),
                jnp.clip(new_pos[..., X], -self.x_semidim, self.x_semidim),
                new_pos[..., X],
            )
            new_pos_y = jnp.where(
                ~jnp.isnan(self.y_semidim),
                jnp.clip(new_pos[..., Y], -self.y_semidim, self.y_semidim),
                new_pos[..., Y],
            )
            new_pos = jnp.where(
                jnp.logical_or(~jnp.isnan(self.x_semidim), ~jnp.isnan(self.y_semidim)),
                jnp.stack([new_pos_x, new_pos_y], axis=-1),
                new_pos,
            )
            entity = entity.replace(state=entity.state.replace(pos=new_pos, vel=vel))

        if entity.rotatable:
            ang_vel = entity.state.ang_vel
            # Compute rotation
            if substep == 0:
                ang_vel = jnp.where(
                    ~jnp.isnan(entity.drag),
                    ang_vel * (1 - entity.drag),
                    ang_vel * (1 - self.drag),
                )
            ang_accel = self.torque_dict[entity.name] / entity.moment_of_inertia
            ang_vel = ang_vel + ang_accel * self.sub_dt
            new_rot = entity.state.rot + ang_vel * self.sub_dt
            entity = entity.replace(
                state=entity.state.replace(rot=new_rot, ang_vel=ang_vel)
            )

        return entity

    @jaxtyped(typechecker=beartype)
    def _update_comm_state(self, agent: Agent):
        # set communication state (directly for now)
        if not agent.silent:
            agent = agent.replace(state=agent.state.replace(c=agent.action.c))
        return agent
