import equinox as eqx
import jax.numpy as jnp
import pytest
from jaxtyping import Array

from jaxvmas.simulator.core.agent import Agent
from jaxvmas.simulator.dynamics.rotation import Rotation


class TestRotationDynamics:
    @pytest.fixture
    def basic_dynamics(self):
        return Rotation()

    @pytest.fixture
    def basic_agent(self):
        agent = Agent.create(
            name="test_agent",
            action_size=2,  # Larger than needed_action_size for testing
        )
        agent = agent._spawn(id=jnp.asarray(1), batch_dim=2, dim_c=0, dim_p=2)
        agent = agent.replace(action=agent.action.replace(u=jnp.zeros((2, 2))))
        return agent

    def test_create(self, basic_dynamics: Rotation):
        assert isinstance(basic_dynamics, Rotation)
        assert basic_dynamics.needed_action_size == 1

    def test_reset(self, basic_dynamics: Rotation):
        # Test reset without index
        reset_dynamics = basic_dynamics.reset()
        assert isinstance(reset_dynamics, Rotation)

        # Test reset with index
        reset_dynamics_index = basic_dynamics.reset(index=0)
        assert isinstance(reset_dynamics_index, Rotation)

        # Test reset with array index
        reset_dynamics_array = basic_dynamics.reset(index=jnp.array([0, 1]))
        assert isinstance(reset_dynamics_array, Rotation)

    def test_check_and_process_action_valid(
        self, basic_dynamics: Rotation, basic_agent: Agent
    ):
        # Test valid action processing
        dynamics, agent = basic_dynamics.check_and_process_action(basic_agent)

        assert isinstance(dynamics, Rotation)
        assert isinstance(agent, Agent)
        assert jnp.array_equal(
            agent.state.torque, basic_agent.action.u[:, 0][..., None]
        )

    def test_check_and_process_action_invalid(self, basic_dynamics: Rotation):
        # Create agent with insufficient action size
        agent = Agent.create(
            name="test_agent",
            action_size=0,  # Less than needed_action_size
        )
        agent = agent._spawn(id=jnp.asarray(1), batch_dim=2, dim_c=0, dim_p=2)

        # Test that it raises ValueError for insufficient action size
        with pytest.raises(ValueError):
            basic_dynamics.check_and_process_action(agent)

    def test_process_action_shapes(self, basic_dynamics: Rotation, basic_agent: Agent):
        # Test output shapes from process_action
        dynamics, agent = basic_dynamics.check_and_process_action(basic_agent)

        assert agent.state.torque.shape == (2, 1)  # batch_dim x 1

    def test_is_jittable(self, basic_dynamics: Rotation, basic_agent: Agent):
        # Test jit compatibility of reset
        @eqx.filter_jit
        def reset_dynamics(dynamics: Rotation):
            return dynamics.reset()

        reset_result = reset_dynamics(basic_dynamics)
        assert isinstance(reset_result, Rotation)

        # Test jit compatibility of check_and_process_action
        @eqx.filter_jit
        def process_action(dynamics: Rotation, agent: Agent):
            return dynamics.check_and_process_action(agent)

        dynamics_result, agent_result = process_action(basic_dynamics, basic_agent)
        assert isinstance(dynamics_result, Rotation)
        assert isinstance(agent_result, Agent)

        # Test jit compatibility with array index
        @eqx.filter_jit
        def reset_with_index(dynamics: Rotation, index: Array):
            return dynamics.reset(index=index)

        reset_index_result = reset_with_index(basic_dynamics, jnp.array([0]))
        assert isinstance(reset_index_result, Rotation)

    def test_batch_processing(self, basic_dynamics: Rotation):
        # Test processing with different batch sizes
        for batch_dim in [1, 2, 4]:
            agent = Agent.create(
                name="test_agent",
                action_size=2,
            )
            agent = agent._spawn(
                id=jnp.asarray(1), batch_dim=batch_dim, dim_c=0, dim_p=2
            )

            dynamics, processed_agent = basic_dynamics.check_and_process_action(agent)
            assert processed_agent.state.torque.shape == (batch_dim, 1)

    def test_zero_action(self, basic_dynamics: Rotation):
        # Test processing with zero actions
        agent = Agent.create(
            name="test_agent",
            action_size=2,
        )
        agent = agent._spawn(id=jnp.asarray(1), batch_dim=2, dim_c=0, dim_p=2)
        agent = agent.replace(action=agent.action.replace(u=jnp.zeros((2, 2))))

        dynamics, processed_agent = basic_dynamics.check_and_process_action(agent)
        assert jnp.all(processed_agent.state.torque == 0)

    def test_positive_rotation(self, basic_dynamics: Rotation):
        # Test with positive rotation action
        agent = Agent.create(
            name="test_agent",
            action_size=1,
        )
        agent = agent._spawn(id=jnp.asarray(1), batch_dim=1, dim_c=0, dim_p=2)
        # Set positive rotation
        agent = agent.replace(action=agent.action.replace(u=jnp.array([[1.0]])))

        dynamics, processed_agent = basic_dynamics.check_and_process_action(agent)

        # Check that torque matches input
        assert jnp.allclose(processed_agent.state.torque, jnp.array([[1.0]]))

    def test_negative_rotation(self, basic_dynamics: Rotation):
        # Test with negative rotation action
        agent = Agent.create(
            name="test_agent",
            action_size=1,
        )
        agent = agent._spawn(id=jnp.asarray(1), batch_dim=1, dim_c=0, dim_p=2)
        # Set negative rotation
        agent = agent.replace(action=agent.action.replace(u=jnp.array([[-1.0]])))

        dynamics, processed_agent = basic_dynamics.check_and_process_action(agent)

        # Check that torque matches input
        assert jnp.allclose(processed_agent.state.torque, jnp.array([[-1.0]]))
