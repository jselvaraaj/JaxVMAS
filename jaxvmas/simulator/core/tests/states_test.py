import equinox as eqx
import jax.numpy as jnp
import pytest

from jaxvmas.simulator.core.states import AgentState, EntityState


class TestEntityState:
    @pytest.fixture
    def entity_state(self):
        return EntityState.create(batch_dim=2, dim_p=4)

    def test_reset(self, entity_state: EntityState):
        # Set non-zero initial values
        non_zero_pos = jnp.ones((2, 4))
        non_zero_vel = jnp.ones((2, 4)) * 2
        non_zero_rot = jnp.ones((2, 1)) * 3
        non_zero_ang_vel = jnp.ones((2, 1)) * 4

        entity_state = entity_state.replace(
            pos=non_zero_pos,
            vel=non_zero_vel,
            rot=non_zero_rot,
            ang_vel=non_zero_ang_vel,
        )

        # Test reset without env_index - should reset all values to zero
        reset_state = entity_state._reset()
        assert jnp.array_equal(reset_state.pos, jnp.zeros((2, 4)))
        assert jnp.array_equal(reset_state.vel, jnp.zeros((2, 4)))
        assert jnp.array_equal(reset_state.rot, jnp.zeros((2, 1)))
        assert jnp.array_equal(reset_state.ang_vel, jnp.zeros((2, 1)))

        # Test reset with env_index=0 - should only reset first environment
        reset_state_index = entity_state._reset(env_index=0)

        # First environment should be zero
        assert jnp.array_equal(reset_state_index.pos[0], jnp.zeros(4))
        assert jnp.array_equal(reset_state_index.vel[0], jnp.zeros(4))
        assert jnp.array_equal(reset_state_index.rot[0], jnp.zeros(1))
        assert jnp.array_equal(reset_state_index.ang_vel[0], jnp.zeros(1))

        # Second environment should remain unchanged
        assert jnp.array_equal(reset_state_index.pos[1], non_zero_pos[1])
        assert jnp.array_equal(reset_state_index.vel[1], non_zero_vel[1])
        assert jnp.array_equal(reset_state_index.rot[1], non_zero_rot[1])
        assert jnp.array_equal(reset_state_index.ang_vel[1], non_zero_ang_vel[1])

    def test_spawn(self, entity_state: EntityState):
        # Set non-zero initial values
        non_zero_pos = jnp.ones((2, 4))
        non_zero_vel = jnp.ones((2, 4)) * 2
        non_zero_rot = jnp.ones((2, 1)) * 3
        non_zero_ang_vel = jnp.ones((2, 1)) * 4

        entity_state = entity_state.replace(
            pos=non_zero_pos,
            vel=non_zero_vel,
            rot=non_zero_rot,
            ang_vel=non_zero_ang_vel,
        )

        # Test spawn - should reset all values to zero
        spawned_state = entity_state._spawn(dim_p=4)
        assert jnp.array_equal(spawned_state.pos, jnp.zeros((2, 4)))
        assert jnp.array_equal(spawned_state.vel, jnp.zeros((2, 4)))
        assert jnp.array_equal(spawned_state.rot, jnp.zeros((2, 1)))
        assert jnp.array_equal(spawned_state.ang_vel, jnp.zeros((2, 1)))

    def test_is_jittable(self, entity_state: EntityState):
        @eqx.filter_jit
        def f(e_s: EntityState):
            return e_s._spawn(dim_p=4)

        f(entity_state)

        @eqx.filter_jit
        def f2(e_s: EntityState):
            return e_s._reset()

        f2(entity_state)

    def test_entity_state_replace(self, entity_state: EntityState):
        # Test replacing position
        new_pos = jnp.array([[1.0, 2.0, 3.0, 4.0], [4.0, 5.0, 6.0, 7.0]])
        updated_state = entity_state.replace(pos=new_pos)
        assert jnp.array_equal(updated_state.pos, new_pos)
        assert jnp.array_equal(
            updated_state.vel, entity_state.vel
        )  # Other fields unchanged
        assert jnp.array_equal(updated_state.rot, entity_state.rot)
        assert jnp.array_equal(updated_state.ang_vel, entity_state.ang_vel)

        # Test replacing velocity
        new_vel = jnp.array([[0.1, 0.2, 0.3, 0.4], [0.4, 0.5, 0.6, 0.7]])
        updated_state = entity_state.replace(vel=new_vel)
        assert jnp.array_equal(updated_state.vel, new_vel)
        assert jnp.array_equal(updated_state.pos, entity_state.pos)

        # Test replacing rotation
        new_rot = jnp.array([[0.5], [1.0]])
        updated_state = entity_state.replace(rot=new_rot)
        assert jnp.array_equal(updated_state.rot, new_rot)
        assert jnp.array_equal(updated_state.ang_vel, entity_state.ang_vel)

        # Test replacing angular velocity
        new_ang_vel = jnp.array([[0.7], [0.8]])
        updated_state = entity_state.replace(ang_vel=new_ang_vel)
        assert jnp.array_equal(updated_state.ang_vel, new_ang_vel)
        assert jnp.array_equal(updated_state.rot, entity_state.rot)

        # Test replacing multiple fields
        updated_state = entity_state.replace(
            pos=new_pos, vel=new_vel, rot=new_rot, ang_vel=new_ang_vel
        )
        assert jnp.array_equal(updated_state.pos, new_pos)
        assert jnp.array_equal(updated_state.vel, new_vel)
        assert jnp.array_equal(updated_state.rot, new_rot)
        assert jnp.array_equal(updated_state.ang_vel, new_ang_vel)

        # Test invalid batch dimension
        with pytest.raises(AssertionError):
            entity_state.replace(pos=jnp.zeros((3, 4)))  # Wrong batch dim

        # Test invalid shape
        with pytest.raises(AssertionError):
            entity_state.replace(pos=jnp.zeros((2, 5)))  # Wrong feature dim


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
        assert spawned_state_zero_c.c.shape == (2, 0)
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
