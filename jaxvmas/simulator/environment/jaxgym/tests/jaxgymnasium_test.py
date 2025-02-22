import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array

from jaxvmas.simulator.core.agent import Agent
from jaxvmas.simulator.core.world import World
from jaxvmas.simulator.environment.environment import Environment
from jaxvmas.simulator.environment.jaxgym.base import EnvData
from jaxvmas.simulator.environment.jaxgym.jaxgymnasium import JaxGymnasiumWrapper
from jaxvmas.simulator.scenario import BaseScenario


class MockWorld(World):
    """Mock world for testing."""

    pass


class MockScenario(BaseScenario):
    """Mock scenario for testing."""

    def make_world(self, batch_dim: int, **kwargs):
        world = MockWorld.create(batch_dim=batch_dim, **kwargs)
        self = self.replace(world=world)
        return self

    def reset_world_at(self, PRNG_key: Array, env_index: int | None) -> "MockScenario":
        return self

    def observation(self, agent: Agent) -> Array:
        return jnp.zeros((self.world.batch_dim, 2))

    def reward(self, agent: Agent) -> Array:
        return jnp.zeros(self.world.batch_dim)


class TestJaxGymnasiumWrapper:
    @pytest.fixture
    def wrapper(self):
        """Create test wrapper with mock environment."""
        PRNG_key = jax.random.PRNGKey(0)
        env = Environment.create(
            scenario=MockScenario.create(),
            num_envs=1,
            terminated_truncated=True,
            PRNG_key=PRNG_key,
            dim_c=2,
            dim_p=2,
        )
        mock_agent_1 = Agent.create(name="agent_0")
        mock_agent_2 = Agent.create(name="agent_1")
        world = env.world
        world = world.add_agent(mock_agent_1)
        world = world.add_agent(mock_agent_2)
        mock_agent_1, mock_agent_2 = world.agents
        env = env.replace(world=world)
        return JaxGymnasiumWrapper.create(env=env)

    def test_create(self, wrapper: JaxGymnasiumWrapper):
        """Test wrapper creation."""
        assert isinstance(wrapper.env, Environment)
        assert wrapper.render_mode == "human"
        assert wrapper.vectorized is False
        assert wrapper.env.num_envs == 1

    def test_create_assertions(self):
        """Test create method assertions."""
        PRNG_key = jax.random.PRNGKey(0)
        # Test num_envs assertion
        env = Environment.create(
            scenario=MockScenario.create(),
            num_envs=2,
            terminated_truncated=True,
            PRNG_key=PRNG_key,
        )
        with pytest.raises(AssertionError):
            JaxGymnasiumWrapper.create(env=env)

        # Test terminated_truncated assertion
        env = Environment.create(
            scenario=MockScenario.create(),
            num_envs=1,
            terminated_truncated=False,
            PRNG_key=PRNG_key,
        )
        with pytest.raises(AssertionError):
            JaxGymnasiumWrapper.create(env=env)

    def test_step_jit(self, wrapper: JaxGymnasiumWrapper):
        """Test jitted step function."""
        PRNG_key = jax.random.PRNGKey(0)
        PRNG_key, key_step_i = jax.random.split(PRNG_key)

        @eqx.filter_jit
        def step(key_step_i: Array, wrapper: JaxGymnasiumWrapper, action):
            return wrapper.step(key_step_i, action)

        action = [jnp.ones((1, 2)), jnp.zeros((1, 2))]
        wrapper, _ = wrapper.reset(PRNG_key=key_step_i)
        PRNG_key, key_step_i = jax.random.split(PRNG_key)
        new_wrapper, env_data = step(key_step_i, wrapper, action)

        assert isinstance(new_wrapper, JaxGymnasiumWrapper)
        assert isinstance(env_data, EnvData)
        assert isinstance(env_data.obs, list)
        assert isinstance(env_data.rews, list)
        assert env_data.terminated.shape == tuple()
        assert env_data.truncated.shape == tuple()
        assert env_data.done.shape == tuple()

    def test_reset_jit(self, wrapper: JaxGymnasiumWrapper):
        """Test jitted reset function."""

        @eqx.filter_jit
        def reset(wrapper: JaxGymnasiumWrapper, PRNG_key: Array):
            return wrapper.reset(PRNG_key=PRNG_key)

        PRNG_key = jax.random.PRNGKey(0)
        wrapper, _ = wrapper.reset(PRNG_key=PRNG_key)
        new_wrapper, (obs, info) = reset(wrapper, PRNG_key)

        assert isinstance(new_wrapper, JaxGymnasiumWrapper)
        assert isinstance(obs, list)
        assert isinstance(info, dict)
        assert len(obs) == 2  # Two agents
        assert obs[0].shape == (2,)  # (batch_dim, obs_dim)

    def test_list_spaces(self):
        """Test wrapper with list spaces."""
        PRNG_key = jax.random.PRNGKey(0)
        env = Environment.create(
            scenario=MockScenario.create(),
            num_envs=1,
            terminated_truncated=True,
            PRNG_key=PRNG_key,
            dim_c=2,
            dim_p=2,
        )
        agent_0 = Agent.create(name="agent_0")
        agent_1 = Agent.create(name="agent_1")
        world = env.world
        world = world.add_agent(agent_0)
        world = world.add_agent(agent_1)
        env = env.replace(world=world)
        wrapper = JaxGymnasiumWrapper.create(env=env)

        @eqx.filter_jit
        def step(key_step_i: Array, wrapper: JaxGymnasiumWrapper, action):
            return wrapper.step(key_step_i, action)

        action = [jnp.ones((1, 2)), jnp.zeros((1, 2))]
        PRNG_key, key_step_i = jax.random.split(PRNG_key)
        new_wrapper, env_data = step(key_step_i, wrapper, action)

        assert isinstance(env_data.obs, list)
        assert isinstance(env_data.rews, list)
        assert env_data.obs[0].shape == (2,)
        assert env_data.rews[0].shape == tuple()
