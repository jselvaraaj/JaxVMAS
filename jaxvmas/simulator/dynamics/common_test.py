import equinox as eqx
import jax.numpy as jnp
import pytest
from jaxtyping import Array

from jaxvmas.simulator.core.agent import Agent
from jaxvmas.simulator.dynamics.common import Dynamics


# Create a concrete test class implementing Dynamics
class MockDynamics(Dynamics):
    needed_action_size: int = 2

    def process_action(self, agent: Agent) -> tuple[Dynamics, Agent]:
        # Simple implementation that just sets the force to the action
        agent = agent.replace(
            state=agent.state.replace(
                force=agent.action.u[:, :2],  # Only use first 2 dimensions
                torque=jnp.zeros((agent.batch_dim, 1)),
            )
        )
        return self, agent


class TestDynamicsClass:
    @pytest.fixture
    def basic_dynamics(self):
        return MockDynamics()

    @pytest.fixture
    def basic_agent(self):
        # Create a basic agent for testing
        return Agent.create(
            batch_dim=2,
            name="test_agent",
            dim_c=0,
            dim_p=2,
            action_size=3,  # Larger than needed_action_size for testing
        )

    def test_create(self, basic_dynamics: Dynamics):
        # Test basic creation
        assert isinstance(basic_dynamics, Dynamics)
        assert basic_dynamics.needed_action_size == 2

    def test_reset(self, basic_dynamics: Dynamics):
        # Test reset without index
        reset_dynamics = basic_dynamics.reset()
        assert isinstance(reset_dynamics, Dynamics)

        # Test reset with index
        reset_dynamics_index = basic_dynamics.reset(index=0)
        assert isinstance(reset_dynamics_index, Dynamics)

        # Test reset with array index
        reset_dynamics_array = basic_dynamics.reset(index=jnp.array([0, 1]))
        assert isinstance(reset_dynamics_array, Dynamics)

    def test_check_and_process_action_valid(
        self, basic_dynamics: Dynamics, basic_agent: Agent
    ):
        basic_agent = basic_agent.replace(
            action=basic_agent.action.replace(
                u=jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            )
        )
        # Test valid action processing
        dynamics, agent = basic_dynamics.check_and_process_action(basic_agent)

        assert isinstance(dynamics, Dynamics)
        assert isinstance(agent, Agent)
        assert jnp.array_equal(agent.state.force, basic_agent.action.u[:, :2])
        assert jnp.array_equal(agent.state.torque, jnp.zeros((2, 1)))

    def test_check_and_process_action_invalid(self, basic_dynamics: Dynamics):
        # Create agent with insufficient action size
        agent = Agent.create(
            batch_dim=2,
            name="test_agent",
            dim_c=0,
            dim_p=2,
            action_size=1,  # Less than needed_action_size
        )

        # Test that it raises ValueError for insufficient action size
        with pytest.raises(ValueError):
            basic_dynamics.check_and_process_action(agent)

    def test_process_action_shapes(self, basic_dynamics: Dynamics, basic_agent: Agent):
        # Test output shapes from process_action
        dynamics, agent = basic_dynamics.check_and_process_action(basic_agent)

        assert agent.state.force.shape == (2, 2)  # batch_dim x dim_p
        assert agent.state.torque.shape == (2, 1)  # batch_dim x 1

    def test_is_jittable(self, basic_dynamics: Dynamics, basic_agent: Agent):
        # Test jit compatibility of reset
        @eqx.filter_jit
        def reset_dynamics(dynamics: Dynamics):
            return dynamics.reset()

        reset_result = reset_dynamics(basic_dynamics)
        assert isinstance(reset_result, Dynamics)

        # Test jit compatibility of check_and_process_action
        @eqx.filter_jit
        def process_action(dynamics: Dynamics, agent: Agent):
            return dynamics.check_and_process_action(agent)

        dynamics_result, agent_result = process_action(basic_dynamics, basic_agent)
        assert isinstance(dynamics_result, Dynamics)
        assert isinstance(agent_result, Agent)

        # Test jit compatibility with array index
        @eqx.filter_jit
        def reset_with_index(dynamics: Dynamics, index: Array):
            return dynamics.reset(index=index)

        reset_index_result = reset_with_index(basic_dynamics, jnp.array([0]))
        assert isinstance(reset_index_result, Dynamics)

    def test_batch_processing(self, basic_dynamics: Dynamics):
        # Test processing with different batch sizes
        for batch_dim in [1, 2, 4]:
            agent = Agent.create(
                batch_dim=batch_dim,
                name="test_agent",
                dim_c=0,
                dim_p=2,
                action_size=3,
            )

            dynamics, processed_agent = basic_dynamics.check_and_process_action(agent)
            assert processed_agent.state.force.shape == (batch_dim, 2)
            assert processed_agent.state.torque.shape == (batch_dim, 1)

    def test_action_clipping(self, basic_dynamics: Dynamics):
        # Create agent with large action values
        agent = Agent.create(
            batch_dim=2,
            name="test_agent",
            dim_c=0,
            dim_p=2,
            action_size=3,
        )
        # Set large action values
        agent = agent.replace(
            action=agent.action.replace(
                u=jnp.array([[100.0, 100.0, 100.0], [-100.0, -100.0, -100.0]])
            )
        )

        dynamics, processed_agent = basic_dynamics.check_and_process_action(agent)
        # Verify force is set from action
        assert jnp.array_equal(processed_agent.state.force, agent.action.u[:, :2])
        assert jnp.array_equal(processed_agent.state.torque, jnp.zeros((2, 1)))

    def test_zero_action(self, basic_dynamics: Dynamics):
        # Test processing with zero actions
        agent = Agent.create(
            batch_dim=2,
            name="test_agent",
            dim_c=0,
            dim_p=2,
            action_size=3,
        )
        agent = agent.replace(action=agent.action.replace(u=jnp.zeros((2, 3))))

        dynamics, processed_agent = basic_dynamics.check_and_process_action(agent)
        assert jnp.all(processed_agent.state.force == 0)
        assert jnp.all(processed_agent.state.torque == 0)
