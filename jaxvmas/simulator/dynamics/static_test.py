import equinox as eqx
import jax.numpy as jnp
import pytest
from jaxtyping import Array

from jaxvmas.simulator.core.agent import Agent
from jaxvmas.simulator.dynamics.static import Static


class TestStaticDynamics:
    @pytest.fixture
    def basic_dynamics(self):
        return Static()

    @pytest.fixture
    def basic_agent(self):
        agent = Agent.create(
            name="test_agent",
            action_size=1,  # Any action size is valid since needed_action_size is 0
        )
        agent = agent._spawn(id=jnp.asarray(1), batch_dim=2, dim_c=0, dim_p=2)
        return agent

    def test_create(self, basic_dynamics: Static):
        assert isinstance(basic_dynamics, Static)
        assert basic_dynamics.needed_action_size == 0

    def test_reset(self, basic_dynamics: Static):
        # Test reset without index
        reset_dynamics = basic_dynamics.reset()
        assert isinstance(reset_dynamics, Static)

        # Test reset with index
        reset_dynamics_index = basic_dynamics.reset(index=0)
        assert isinstance(reset_dynamics_index, Static)

        # Test reset with array index
        reset_dynamics_array = basic_dynamics.reset(index=jnp.array([0, 1]))
        assert isinstance(reset_dynamics_array, Static)

    def test_check_and_process_action_valid(
        self, basic_dynamics: Static, basic_agent: Agent
    ):
        # Test valid action processing
        dynamics, agent = basic_dynamics.check_and_process_action(basic_agent)

        assert isinstance(dynamics, Static)
        assert isinstance(agent, Agent)
        # Static dynamics should not modify the agent's state
        assert jnp.array_equal(agent.state.force, basic_agent.state.force)
        assert jnp.array_equal(agent.state.torque, basic_agent.state.torque)

    def test_process_action_shapes(self, basic_dynamics: Static, basic_agent: Agent):
        # Test output shapes from process_action
        dynamics, agent = basic_dynamics.check_and_process_action(basic_agent)

        assert agent.state.force.shape == (2, 2)  # batch_dim x dim_p
        assert agent.state.torque.shape == (2, 1)  # batch_dim x 1

    def test_is_jittable(self, basic_dynamics: Static, basic_agent: Agent):
        # Test jit compatibility of reset
        @eqx.filter_jit
        def reset_dynamics(dynamics: Static):
            return dynamics.reset()

        reset_result = reset_dynamics(basic_dynamics)
        assert isinstance(reset_result, Static)

        # Test jit compatibility of check_and_process_action
        @eqx.filter_jit
        def process_action(dynamics: Static, agent: Agent):
            return dynamics.check_and_process_action(agent)

        dynamics_result, agent_result = process_action(basic_dynamics, basic_agent)
        assert isinstance(dynamics_result, Static)
        assert isinstance(agent_result, Agent)

        # Test jit compatibility with array index
        @eqx.filter_jit
        def reset_with_index(dynamics: Static, index: Array):
            return dynamics.reset(index=index)

        reset_index_result = reset_with_index(basic_dynamics, jnp.array([0]))
        assert isinstance(reset_index_result, Static)

    def test_batch_processing(self, basic_dynamics: Static):
        # Test processing with different batch sizes
        for batch_dim in [1, 2, 4]:
            agent = Agent.create(
                name="test_agent",
                action_size=1,
            )
            agent = agent._spawn(
                id=jnp.asarray(1), batch_dim=batch_dim, dim_c=0, dim_p=2
            )

            dynamics, processed_agent = basic_dynamics.check_and_process_action(agent)
            assert processed_agent.state.force.shape == (batch_dim, 2)
            assert processed_agent.state.torque.shape == (batch_dim, 1)

    def test_state_preservation(self, basic_dynamics: Static):
        # Test that the agent's state remains unchanged after processing
        initial_force = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        initial_torque = jnp.array([[0.5], [1.5]])

        agent = Agent.create(
            name="test_agent",
            action_size=1,
        )
        agent = agent._spawn(id=jnp.asarray(1), batch_dim=2, dim_c=0, dim_p=2)
        agent = agent.replace(
            state=agent.state.replace(force=initial_force, torque=initial_torque)
        )

        dynamics, processed_agent = basic_dynamics.check_and_process_action(agent)

        # Verify state remains unchanged
        assert jnp.array_equal(processed_agent.state.force, initial_force)
        assert jnp.array_equal(processed_agent.state.torque, initial_torque)
