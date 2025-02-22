import equinox as eqx
import jax.numpy as jnp
import pytest
from jaxtyping import Array

from jaxvmas.simulator.core.agent import Agent
from jaxvmas.simulator.dynamics.holonomic import Holonomic


class TestHolonomicDynamics:
    @pytest.fixture
    def basic_dynamics(self):
        return Holonomic()

    @pytest.fixture
    def basic_agent(self):
        agent = Agent.create(
            name="test_agent",
            action_size=3,  # Larger than needed_action_size for testing
        )
        agent = agent._spawn(id=jnp.asarray(1), batch_dim=2, dim_c=0, dim_p=2)
        agent = agent.replace(action=agent.action.replace(u=jnp.zeros((2, 3))))
        return agent

    def test_create(self, basic_dynamics: Holonomic):
        assert isinstance(basic_dynamics, Holonomic)
        assert basic_dynamics.needed_action_size == 2

    def test_reset(self, basic_dynamics: Holonomic):
        # Test reset without index
        reset_dynamics = basic_dynamics.reset()
        assert isinstance(reset_dynamics, Holonomic)

        # Test reset with index
        reset_dynamics_index = basic_dynamics.reset(index=0)
        assert isinstance(reset_dynamics_index, Holonomic)

        # Test reset with array index
        reset_dynamics_array = basic_dynamics.reset(index=jnp.array([0, 1]))
        assert isinstance(reset_dynamics_array, Holonomic)

    def test_check_and_process_action_valid(
        self, basic_dynamics: Holonomic, basic_agent: Agent
    ):
        # Test valid action processing
        dynamics, agent = basic_dynamics.check_and_process_action(basic_agent)

        assert isinstance(dynamics, Holonomic)
        assert isinstance(agent, Agent)
        assert jnp.array_equal(agent.state.force, basic_agent.action.u[:, :2])

    def test_check_and_process_action_invalid(self, basic_dynamics: Holonomic):
        # Create agent with insufficient action size
        agent = Agent.create(
            name="test_agent",
            action_size=1,  # Less than needed_action_size
        )
        agent = agent._spawn(id=jnp.asarray(1), batch_dim=2, dim_c=0, dim_p=2)

        # Test that it raises ValueError for insufficient action size
        with pytest.raises(ValueError):
            basic_dynamics.check_and_process_action(agent)

    def test_process_action_shapes(self, basic_dynamics: Holonomic, basic_agent: Agent):
        # Test output shapes from process_action
        dynamics, agent = basic_dynamics.check_and_process_action(basic_agent)

        assert agent.state.force.shape == (2, 2)  # batch_dim x dim_p

    def test_is_jittable(self, basic_dynamics: Holonomic, basic_agent: Agent):
        # Test jit compatibility of reset
        @eqx.filter_jit
        def reset_dynamics(dynamics: Holonomic):
            return dynamics.reset()

        reset_result = reset_dynamics(basic_dynamics)
        assert isinstance(reset_result, Holonomic)

        # Test jit compatibility of check_and_process_action
        @eqx.filter_jit
        def process_action(dynamics: Holonomic, agent: Agent):
            return dynamics.check_and_process_action(agent)

        dynamics_result, agent_result = process_action(basic_dynamics, basic_agent)
        assert isinstance(dynamics_result, Holonomic)
        assert isinstance(agent_result, Agent)

        # Test jit compatibility with array index
        @eqx.filter_jit
        def reset_with_index(dynamics: Holonomic, index: Array):
            return dynamics.reset(index=index)

        reset_index_result = reset_with_index(basic_dynamics, jnp.array([0]))
        assert isinstance(reset_index_result, Holonomic)

    def test_batch_processing(self, basic_dynamics: Holonomic):
        # Test processing with different batch sizes
        for batch_dim in [1, 2, 4]:
            agent = Agent.create(
                name="test_agent",
                action_size=3,
            )
            agent = agent._spawn(
                id=jnp.asarray(1), batch_dim=batch_dim, dim_c=0, dim_p=2
            )

            dynamics, processed_agent = basic_dynamics.check_and_process_action(agent)
            assert processed_agent.state.force.shape == (batch_dim, 2)

    def test_zero_action(self, basic_dynamics: Holonomic):
        # Test processing with zero actions
        agent = Agent.create(
            name="test_agent",
            action_size=3,
        )
        agent = agent._spawn(id=jnp.asarray(1), batch_dim=2, dim_c=0, dim_p=2)
        agent = agent.replace(action=agent.action.replace(u=jnp.zeros((2, 3))))

        dynamics, processed_agent = basic_dynamics.check_and_process_action(agent)
        assert jnp.all(processed_agent.state.force == 0)
