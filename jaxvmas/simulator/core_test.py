from unittest.mock import Mock

import equinox as eqx
import jax.numpy as jnp
import pytest
from jaxtyping import Array

from jaxvmas.simulator.core import (
    DRAG,
    Action,
    Agent,
    AgentState,
    Entity,
    EntityState,
    Landmark,
    Sphere,
    World,
)


class TestEntityState:
    @pytest.fixture
    def entity_state(self):
        # Initialize EntityState with batch_dim=2, dim_c=3, dim_p=4
        return EntityState.create(batch_dim=2, dim_c=3, dim_p=4)

    def test_reset(self, entity_state: EntityState):
        # Test reset without env_index
        reset_state = entity_state._reset()

        assert jnp.array_equal(reset_state.pos, jnp.zeros((2, 4)))
        assert jnp.array_equal(reset_state.vel, jnp.zeros((2, 4)))
        assert jnp.array_equal(reset_state.rot, jnp.zeros((2, 1)))
        assert jnp.array_equal(reset_state.ang_vel, jnp.zeros((2, 1)))

        # Test reset with env_index
        reset_state_index = entity_state._reset(env_index=0)
        assert jnp.array_equal(reset_state_index.pos, jnp.zeros((2, 4)))
        assert jnp.array_equal(reset_state_index.vel, jnp.zeros((2, 4)))
        assert jnp.array_equal(reset_state_index.rot, jnp.zeros((2, 1)))
        assert jnp.array_equal(reset_state_index.ang_vel, jnp.zeros((2, 1)))

    def test_spawn(self, entity_state: EntityState):
        # Test spawn
        spawned_state = entity_state._spawn(dim_c=3, dim_p=4)
        assert jnp.array_equal(spawned_state.pos, jnp.zeros((2, 4)))
        assert jnp.array_equal(spawned_state.vel, jnp.zeros((2, 4)))
        assert jnp.array_equal(spawned_state.rot, jnp.zeros((2, 1)))
        assert jnp.array_equal(spawned_state.ang_vel, jnp.zeros((2, 1)))

    def test_is_jittable(self, entity_state: EntityState):
        @eqx.filter_jit
        def f(e_s):
            return e_s._spawn(dim_c=3, dim_p=4)

        f(entity_state)

        @eqx.filter_jit
        def f2(e_s):
            return e_s._reset()

        f2(entity_state)


class TestAgentState:
    @pytest.fixture
    def agent_state(self):
        # Initialize AgentState with batch_dim=2, dim_c=3, dim_p=4
        return AgentState.create(batch_dim=2, dim_c=3, dim_p=4)

    def test_init(self, agent_state: AgentState):
        assert jnp.array_equal(agent_state.c, jnp.zeros((2, 3)))
        assert jnp.array_equal(agent_state.force, jnp.zeros((2, 4)))
        assert jnp.array_equal(agent_state.torque, jnp.zeros((2, 1)))
        # Test inheritance
        assert jnp.array_equal(agent_state.pos, jnp.zeros((2, 4)))
        assert jnp.array_equal(agent_state.vel, jnp.zeros((2, 4)))
        assert jnp.array_equal(agent_state.rot, jnp.zeros((2, 1)))
        assert jnp.array_equal(agent_state.ang_vel, jnp.zeros((2, 1)))

    def test_reset(self, agent_state: AgentState):
        # Modify state to non-zero values
        agent_state = agent_state.replace(
            c=jnp.ones((2, 3)), force=jnp.ones((2, 4)), torque=jnp.ones((2, 1))
        )

        # Test reset without env_index
        reset_state = agent_state._reset()
        assert jnp.array_equal(reset_state.c, jnp.zeros((2, 3)))
        assert jnp.array_equal(reset_state.force, jnp.zeros((2, 4)))
        assert jnp.array_equal(reset_state.torque, jnp.zeros((2, 1)))

        # Test reset with env_index
        reset_state_index = agent_state._reset(env_index=0)
        assert jnp.array_equal(reset_state_index.c[0], jnp.zeros(3))
        assert jnp.array_equal(reset_state_index.c[1], jnp.ones(3))
        assert jnp.array_equal(reset_state_index.force[0], jnp.zeros(4))
        assert jnp.array_equal(reset_state_index.force[1], jnp.ones(4))
        assert jnp.array_equal(reset_state_index.torque[0], jnp.zeros(1))
        assert jnp.array_equal(reset_state_index.torque[1], jnp.ones(1))

    def test_spawn(self, agent_state: AgentState):
        # Test spawn with non-zero dim_c
        spawned_state = agent_state._spawn(dim_c=2, dim_p=4)
        assert spawned_state.c.shape == (2, 2)
        assert spawned_state.force.shape == (2, 4)
        assert spawned_state.torque.shape == (2, 1)

        # Test spawn with zero dim_c
        spawned_state_zero_c = agent_state._spawn(dim_c=0, dim_p=4)
        assert spawned_state_zero_c.c is None
        assert spawned_state_zero_c.force.shape == (2, 4)
        assert spawned_state_zero_c.torque.shape == (2, 1)

    def test_is_jittable(self, agent_state: AgentState):
        @eqx.filter_jit
        def f(a_s):
            return a_s._spawn(dim_c=3, dim_p=4)

        f(agent_state)

        @eqx.filter_jit
        def f2(a_s):
            return a_s._reset()

        f2(agent_state)

        @eqx.filter_jit
        def f3(a_s):
            return a_s._reset(env_index=0)

        f3(agent_state)


class TestAction:
    @pytest.fixture
    def basic_action(self):
        # Create basic action with float inputs
        return Action.create(
            batch_dim=2,
            action_size=3,
            comm_dim=2,
            u_range=1.0,
            u_multiplier=1.0,
            u_noise=0.1,
        )

    def test_create_with_float_inputs(self, basic_action):
        # Test creation with float inputs
        assert basic_action.u.shape == (2, 3)
        assert basic_action.c.shape == (2, 2)
        assert basic_action.u_range == 1.0
        assert basic_action.u_multiplier == 1.0
        assert basic_action.u_noise == 0.1

    def test_create_with_sequence_inputs(self):
        # Test creation with sequence inputs
        action = Action.create(
            batch_dim=2,
            action_size=3,
            comm_dim=2,
            u_range=[1.0, 2.0, 3.0],
            u_multiplier=[0.5, 1.0, 1.5],
            u_noise=[0.1, 0.2, 0.3],
        )
        assert jnp.array_equal(action.u_range_jax_array, jnp.array([1.0, 2.0, 3.0]))
        assert jnp.array_equal(
            action.u_multiplier_jax_array, jnp.array([0.5, 1.0, 1.5])
        )
        assert jnp.array_equal(action.u_noise_jax_array, jnp.array([0.1, 0.2, 0.3]))

    def test_jax_array_properties(self, basic_action):
        # Test jax array property conversion
        assert jnp.array_equal(
            basic_action.u_range_jax_array, jnp.array([1.0, 1.0, 1.0])
        )
        assert jnp.array_equal(
            basic_action.u_multiplier_jax_array, jnp.array([1.0, 1.0, 1.0])
        )
        assert jnp.array_equal(
            basic_action.u_noise_jax_array, jnp.array([0.1, 0.1, 0.1])
        )

    def test_reset_without_env_index(self, basic_action):
        # Set non-zero values
        action = basic_action.replace(u=jnp.ones((2, 3)), c=jnp.ones((2, 2)))
        # Test reset
        reset_action = action._reset(None)
        assert jnp.all(reset_action.u == 0)
        assert jnp.all(reset_action.c == 0)

    def test_reset_with_env_index(self, basic_action):
        # Set non-zero values
        action = basic_action.replace(u=jnp.ones((2, 3)), c=jnp.ones((2, 2)))
        # Test reset for specific environment
        reset_action = action._reset(0)
        assert jnp.all(reset_action.u[0] == 0)
        assert jnp.all(reset_action.u[1] == 1)
        assert jnp.all(reset_action.c[0] == 0)
        assert jnp.all(reset_action.c[1] == 1)

    def test_invalid_sequence_length(self):
        # Test validation of sequence length
        with pytest.raises(AssertionError):
            Action.create(
                batch_dim=2,
                action_size=3,
                comm_dim=2,
                u_range=[1.0, 2.0],  # Wrong length
                u_multiplier=1.0,
                u_noise=0.1,
            )


class TestEntity:
    @pytest.fixture
    def basic_entity(self):
        return Entity.create(
            batch_dim=2,
            name="test_entity",
            movable=True,
            rotatable=True,
        )

    def test_create(self, basic_entity):
        # Test basic properties
        assert basic_entity.name == "test_entity"
        assert basic_entity.movable is True
        assert basic_entity.rotatable is True
        assert basic_entity.collide is True
        assert basic_entity.mass == 1.0
        assert basic_entity.density == 25.0
        assert jnp.array_equal(basic_entity.gravity, jnp.zeros((2, 1)))

        # Test creation with custom parameters
        custom_entity = Entity.create(
            batch_dim=2,
            name="custom",
            movable=False,
            gravity=jnp.array([0.0, -9.81]),
            mass=2.0,
            v_range=5.0,
            max_speed=10.0,
        )
        assert custom_entity.mass == 2.0
        assert custom_entity.v_range == 5.0
        assert custom_entity.max_speed == 10.0
        assert jnp.array_equal(custom_entity.gravity, jnp.array([0.0, -9.81]))

    def test_property_setters(self, basic_entity):
        # Test setting position
        new_pos = jnp.array([1.0, 2.0])
        updated_entity = basic_entity.set_pos(new_pos, batch_index=0)
        assert jnp.array_equal(updated_entity.state.pos[0], new_pos)

        # Test setting velocity
        new_vel = jnp.array([0.5, 0.5])
        updated_entity = basic_entity.set_vel(new_vel, batch_index=0)
        assert jnp.array_equal(updated_entity.state.vel[0], new_vel)

        # Test setting rotation
        new_rot = jnp.array([1.5])
        updated_entity = basic_entity.set_rot(new_rot, batch_index=0)
        assert jnp.array_equal(updated_entity.state.rot[0], new_rot)

    def test_collision_filter(self, basic_entity):
        # Test default collision filter
        other_entity = Entity.create(batch_dim=2, name="other")
        assert basic_entity.collides(other_entity) is True

        # Test custom collision filter
        custom_filter_entity = Entity.create(
            batch_dim=2, name="filtered", collision_filter=lambda e: e.name == "target"
        )
        assert custom_filter_entity.collides(other_entity) is False
        target_entity = Entity.create(batch_dim=2, name="target")
        assert custom_filter_entity.collides(target_entity) is True

    def test_render_flags(self, basic_entity):
        # Test initial render state
        assert jnp.all(basic_entity._render == True)

        # Test reset render
        basic_entity = basic_entity.reset_render()
        assert jnp.all(basic_entity._render == True)

    def test_spawn_and_reset(self, basic_entity):
        # Test spawn
        spawned_entity = basic_entity._spawn(dim_c=3, dim_p=2)
        assert spawned_entity.state.pos.shape == (2, 2)
        assert spawned_entity.state.vel.shape == (2, 2)
        assert spawned_entity.state.rot.shape == (2, 1)

        # Test reset
        reset_entity = basic_entity._reset(env_index=0)
        assert jnp.all(reset_entity.state.pos[0] == 0)
        assert jnp.all(reset_entity.state.vel[0] == 0)
        assert jnp.all(reset_entity.state.rot[0] == 0)

    def test_is_jittable(self, basic_entity):
        @eqx.filter_jit
        def f(entity, pos):
            return entity.set_pos(pos, batch_index=0)

        test_pos = jnp.array([1.0, 2.0])
        f(basic_entity, test_pos)


class TestAgent:
    @pytest.fixture
    def basic_agent(self):
        # Create basic agent with minimal configuration
        return Agent.create(
            batch_dim=2,
            name="test_agent",
            dim_c=3,
            dim_p=2,
            movable=True,
            rotatable=True,
        )

    def test_create_basic(self, basic_agent):
        # Test basic properties
        assert basic_agent.name == "test_agent"
        assert basic_agent.movable is True
        assert basic_agent.rotatable is True
        assert basic_agent.silent is True
        assert basic_agent.adversary is False
        assert basic_agent.alpha == 0.5
        assert basic_agent.action_size == 2  # Default for Holonomic dynamics
        assert basic_agent.discrete_action_nvec == [
            3,
            3,
        ]  # Default 3-way discretization

    def test_create_with_custom_config(self):
        # Test creation with custom configuration
        agent = Agent.create(
            batch_dim=2,
            name="custom_agent",
            dim_c=4,
            dim_p=2,
            obs_range=10.0,
            obs_noise=0.1,
            f_range=5.0,
            max_f=10.0,
            silent=False,
            action_size=3,
            discrete_action_nvec=[3, 4, 5],
        )

        assert agent.obs_range == 10.0
        assert agent.obs_noise == 0.1
        assert agent.f_range == 5.0
        assert agent.max_f == 10.0
        assert agent.silent is False
        assert agent.action_size == 3
        assert agent.discrete_action_nvec == [3, 4, 5]

    def test_invalid_configurations(self):
        # Test invalid obs_range with sensors
        with pytest.raises(AssertionError):
            Agent.create(
                batch_dim=2,
                name="invalid",
                dim_c=3,
                dim_p=2,
                obs_range=0.0,
                sensors=[Mock()],
            )

        # Test inconsistent action_size and discrete_action_nvec
        with pytest.raises(ValueError):
            Agent.create(
                batch_dim=2,
                name="invalid",
                dim_c=3,
                dim_p=2,
                action_size=2,
                discrete_action_nvec=[3, 3, 3],
            )

    def test_reset_and_spawn(self, basic_agent):
        # Test spawn
        spawned_agent = basic_agent._spawn(dim_c=4, dim_p=2)
        assert spawned_agent.state.pos.shape == (2, 2)
        assert spawned_agent.state.vel.shape == (2, 2)

        # Test spawn with silent agent
        silent_agent = basic_agent.replace(silent=True)
        spawned_silent = silent_agent._spawn(dim_c=4, dim_p=2)
        assert spawned_silent.state.c is None  # No communication dimension

        # Test reset
        reset_agent = basic_agent._reset(env_index=0)
        assert jnp.all(reset_agent.state.pos[0] == 0)
        assert jnp.all(reset_agent.state.vel[0] == 0)
        assert jnp.all(reset_agent.action.u[0] == 0)

    def test_render(self, basic_agent):
        # Test basic rendering
        geoms = basic_agent.render(env_index=0)
        assert len(geoms) > 0  # Should have at least one geometry

        # Test rendering with action visualization
        agent_with_render = basic_agent.replace(render_action=True)
        agent_with_render = agent_with_render.replace(
            state=agent_with_render.state.replace(
                force=jnp.ones((2, 2))  # Non-zero force
            )
        )
        geoms_with_action = agent_with_render.render(env_index=0)
        assert len(geoms_with_action) > len(
            geoms
        )  # Should have additional line for force

    def test_is_jittable(self, basic_agent):
        @eqx.filter_jit
        def f(agent):
            return agent._spawn(dim_c=3, dim_p=2)

        f(basic_agent)

        @eqx.filter_jit
        def f2(agent):
            return agent._reset(env_index=0)

        f2(basic_agent)


class TestWorld:
    @pytest.fixture
    def basic_world(self):
        # Create a basic world with minimal configuration
        return World.create(batch_dim=2)

    @pytest.fixture
    def agent(self):
        return Agent.create(
            batch_dim=2,
            name="test_agent",
            dim_c=3,
            dim_p=2,
            movable=True,
            rotatable=True,
        )

    @pytest.fixture
    def landmark(self):
        return Landmark.create(batch_dim=2, name="test_landmark")

    @pytest.fixture
    def world_with_agent(self, basic_world, agent):
        return basic_world.add_agent(agent)

    def test_create(self, basic_world):
        # Test basic properties
        assert basic_world.batch_dim == 2
        assert basic_world._dt == 0.1
        assert basic_world._substeps == 1
        assert basic_world._drag == DRAG
        assert len(basic_world._agents) == 0
        assert len(basic_world._landmarks) == 0
        assert isinstance(basic_world._joints, dict)

    def test_add_agent(self, basic_world, agent):
        world = basic_world.add_agent(agent)
        assert len(world._agents) == 1
        assert world._agents[0].name == "test_agent"
        assert world._agents[0].batch_dim == 2

    def test_add_landmark(self, basic_world, landmark):
        world = basic_world.add_landmark(landmark)
        assert len(world._landmarks) == 1
        assert world._landmarks[0].name == "test_landmark"
        assert world._landmarks[0].batch_dim == 2

    def test_reset(self, world_with_agent):
        # Modify agent state
        agent = world_with_agent._agents[0]
        agent = agent.replace(state=agent.state.replace(pos=jnp.ones((2, 2))))
        world = world_with_agent.replace(_agents=[agent])

        # Test reset
        world = world.reset(env_index=0)
        assert jnp.all(world._agents[0].state.pos[0] == 0)
        assert jnp.all(world._agents[0].state.pos[1] == 1)

    def test_step(self, world_with_agent):
        # Test basic stepping without forces
        world = world_with_agent.step()
        assert isinstance(world, World)

        # Test stepping with forces
        agent = world._agents[0]
        agent = agent.replace(state=agent.state.replace(force=jnp.ones((2, 2))))
        world = world.replace(_agents=[agent])
        world = world.step()
        assert isinstance(world, World)
        # Velocity should have changed due to force
        assert not jnp.all(world._agents[0].state.vel == 0)

    def test_collisions(self, basic_world: World):
        # Create two overlapping spherical agents
        # Create two overlapping spherical agents
        agent1 = Agent.create(
            batch_dim=2,
            name="agent1",
            dim_c=0,
            dim_p=2,
            movable=True,
        )
        agent2 = Agent.create(
            batch_dim=2,
            name="agent2",
            dim_c=0,
            dim_p=2,
            movable=True,
        )

        world = basic_world.add_agent(agent1).add_agent(agent2)

        # Test collision detection
        assert world.collides(agent1, agent2)

        # Test no collision with non-collidable entity
        landmark = Landmark.create(batch_dim=2, name="landmark", collide=False)
        world = world.add_landmark(landmark)
        assert not world.collides(agent1, landmark)

    def test_communication(self, basic_world):
        # Create world with communication
        world = World.create(batch_dim=2, dim_c=3)
        agent = Agent.create(batch_dim=2, name="agent", dim_c=3, dim_p=2, silent=False)
        world = world.add_agent(agent)

        # Set communication action
        agent = world._agents[0]
        agent = agent.replace(action=agent.action.replace(c=jnp.ones((2, 3))))
        world = world.replace(_agents=[agent])

        # Step world and check communication state
        world = world.step()
        assert jnp.all(world._agents[0].state.c == 1)

    def test_joints(self, basic_world):
        from jaxvmas.simulator.joints import Joint, JointConstraint

        # Create world with higher substeps for joint stability
        world = World.create(batch_dim=2, substeps=5)

        # Create two agents to be joined
        agent1 = Agent.create(
            batch_dim=2,
            name="agent1",
            dim_c=0,
            dim_p=2,
            movable=True,
        )
        agent2 = Agent.create(
            batch_dim=2,
            name="agent2",
            dim_c=0,
            dim_p=2,
            movable=True,
        )

        world = world.add_agent(agent1).add_agent(agent2)

        # Create and add joint
        joint = Joint.create(
            batch_dim=2,
            entity_a=agent1,
            entity_b=agent2,
            anchor_a=(0, 0),
            anchor_b=(0, 0),
            dist=1.0,
        )
        world = world.add_joint(joint)

        assert len(world._joints) == 2
        assert isinstance(list(world._joints.values())[0], JointConstraint)
        assert isinstance(list(world._joints.values())[1], JointConstraint)

    def test_entity_index_map(self, world_with_agent):
        # Test entity index map is properly updated
        world = world_with_agent.reset()
        assert len(world._entity_index_map) == 1
        assert world._agents[0].name in world._entity_index_map
        assert world._entity_index_map[world._agents[0].name] == 0

    def test_boundary_conditions(self, basic_world):
        # Test world with boundaries
        world = World.create(batch_dim=2, x_semidim=1.0, y_semidim=1.0)
        agent = Agent.create(
            batch_dim=2,
            name="agent",
            dim_c=0,
            dim_p=2,
            movable=True,
        )
        world = world.add_agent(agent)

        # Move agent beyond boundary
        agent = world._agents[0]
        agent = agent.replace(
            state=agent.state.replace(pos=jnp.array([[2.0, 0.0], [0.0, 2.0]]))
        )
        world = world.replace(_agents=[agent])

        # Step world and check if position is constrained
        world = world.step()
        assert jnp.all(jnp.abs(world._agents[0].state.pos) <= 1.0)

    def test_collision_response(self):
        # Create two colliding agents with no initial forces
        agent1 = Agent.create(
            batch_dim=1,
            name="collider1",
            dim_p=2,
            dim_c=0,
            mass=1.0,
        )
        agent1 = agent1.replace(
            state=agent1.state.replace(
                pos=jnp.array([[0.0, 0.0]]),
            ),
            shape=Sphere(radius=0.1),
        )

        agent2 = Agent.create(
            batch_dim=1,
            name="collider2",
            dim_p=2,
            dim_c=0,
            mass=1.0,
        )
        agent2 = agent2.replace(
            state=agent2.state.replace(
                pos=jnp.array([[0.15, 0.0]]),
            ),
            shape=Sphere(radius=0.1),
        )

        world = World.create(batch_dim=1, substeps=10)
        world = world.replace(_agents=[agent1, agent2])
        initial_dist = jnp.linalg.norm(
            world._agents[0].state.pos - world._agents[1].state.pos
        )

        # Step and verify collision response
        world = world.step()
        # Calculate final distance
        final_dist = jnp.linalg.norm(
            world._agents[0].state.pos - world._agents[1].state.pos
        )
        assert (
            final_dist > initial_dist
        ), f"Agents moved closer together (from {initial_dist} to {final_dist})"

    def test_joint_constraint_satisfaction(self):
        # Create joined agents
        agent1 = Agent.create(batch_dim=1, name="joint1", dim_p=2, dim_c=0)
        agent1 = agent1.replace(state=agent1.state.replace(pos=jnp.array([[0.0, 0.0]])))

        agent2 = Agent.create(batch_dim=1, name="joint2", dim_p=2, dim_c=0)
        agent2 = agent2.replace(state=agent2.state.replace(pos=jnp.array([[0.5, 0.0]])))

        world = World.create(
            batch_dim=1,
            substeps=10,  # Higher substeps for joint stability
            joint_force=500.0,  # Increase joint force for stronger constraint
            linear_friction=0.0,  # Remove friction to allow easier movement
            drag=0.1,  # Reduce drag for smoother motion
        )

        world = world.replace(_agents=[agent1, agent2])
        world = world.reset()

        # Add joint constraint
        from jaxvmas.simulator.joints import JointConstraint

        # Create a direct joint constraint between agents
        constraint = JointConstraint.create(
            entity_a=agent1,
            entity_b=agent2,
            anchor_a=(0, 0),
            anchor_b=(0, 0),
            dist=0.5,  # Target distance
        )
        world = world.replace(
            _joints={frozenset({agent1.name, agent2.name}): constraint}
        )

        # Step and verify joint constraint
        stepped_world = world.step()

        # Distance between agents should be close to joint distance
        final_dist = jnp.linalg.norm(
            stepped_world._agents[0].state.pos - stepped_world._agents[1].state.pos
        )
        print(f"Final distance: {final_dist}, Target distance: 0.5")

        # Check if final distance is reasonably close to desired distance
        assert jnp.abs(final_dist - 0.5) < 0.1, (
            f"Joint constraint not satisfied. Expected distance close to 0.5, "
            f"but got {final_dist:.3f}"
        )

    def test_extreme_boundary_conditions(self):
        # Test negative boundaries
        world = World.create(batch_dim=1, x_semidim=-1.0, y_semidim=-0.5)
        agent = Agent.create(batch_dim=1, name="boundary_test", dim_p=2, dim_c=0)
        agent = agent.replace(state=agent.state.replace(pos=jnp.array([[2.0, 1.0]])))
        world = world.add_agent(agent)

        # Step and verify boundary handling
        stepped_world = world.step()

        # Agent should be constrained by boundaries
        final_pos = stepped_world._agents[0].state.pos[0]
        assert jnp.all(jnp.abs(final_pos) <= jnp.array([1.0, 0.5]))

    def test_is_jittable(self, basic_world: World, agent: Agent, landmark: Landmark):
        # Test jit compatibility of world creation and entity addition
        @eqx.filter_jit
        def create_world_with_entities(world: World, agent: Agent, landmark: Landmark):
            world = world.add_agent(agent)
            world = world.add_landmark(landmark)
            world = world.reset()
            return world

        world = create_world_with_entities(basic_world, agent, landmark)
        assert len(world._agents) == 1
        assert len(world._landmarks) == 1

        # Test jit compatibility of stepping with forces
        @eqx.filter_jit
        def step_with_force(world: World):
            agent = world._agents[0]
            agent = agent.replace(state=agent.state.replace(force=jnp.ones((2, 2))))
            world = world.replace(_agents=[agent])
            return world.step()

        stepped_world = step_with_force(world)
        assert not jnp.all(stepped_world._agents[0].state.vel == 0)

        # Test jit compatibility of reset with specific index
        @eqx.filter_jit
        def reset_env(world: World, env_idx: int):
            return world.reset(env_index=env_idx)

        reset_world = reset_env(world, 0)
        assert jnp.all(reset_world._agents[0].state.pos[0] == 0)

        # Test jit compatibility of collision detection
        @eqx.filter_jit
        def check_collision(world: World, entity1: Entity, entity2: Entity):
            return world.collides(entity1, entity2)

        collides = check_collision(world, world._agents[0], world._landmarks[0])
        assert isinstance(collides, Array)

        # Test jit compatibility with joints
        from jaxvmas.simulator.joints import Joint

        @eqx.filter_jit
        def add_and_step_with_joint(world: World):
            agent2 = Agent.create(
                batch_dim=2,
                name="agent2",
                dim_c=0,
                dim_p=2,
                movable=True,
            )
            world = world.add_agent(agent2)
            world = world.replace(_substeps=2)
            joint = Joint.create(
                batch_dim=2,
                entity_a=world._agents[0],
                entity_b=world._agents[1],
                anchor_a=(0, 0),
                anchor_b=(0, 0),
                dist=1.0,
            )
            world = world.add_joint(joint)

            world = world.reset()

            return world.step()

        joint_world = add_and_step_with_joint(world)
        assert len(joint_world._joints) == 2

        # Test jit compatibility with boundary conditions
        @eqx.filter_jit
        def step_with_boundaries(world: World, pos: Array):
            agent = world._agents[0]
            agent = agent.replace(state=agent.state.replace(pos=pos))
            world = world.replace(_agents=[agent])
            world = world.replace(_x_semidim=1.0, _y_semidim=1.0)
            return world.step()

        boundary_world = step_with_boundaries(
            world, jnp.array([[2.0, 2.0], [0.0, 0.0]])
        )
        assert jnp.all(jnp.abs(boundary_world._agents[0].state.pos[0]) <= 1.0)

        # Test jit compatibility with multiple substeps
        @eqx.filter_jit
        def step_with_substeps(world: World, substeps: int):
            return world.replace(_substeps=substeps).step()

        multi_step_world = step_with_substeps(world, 5)
        assert isinstance(multi_step_world, World)
