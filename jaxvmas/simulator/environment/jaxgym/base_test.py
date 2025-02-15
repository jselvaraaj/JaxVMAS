import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Float

from jaxvmas.simulator.core import Agent, World
from jaxvmas.simulator.environment.environment import Environment
from jaxvmas.simulator.environment.jaxgym.base import (
    BaseJaxGymWrapper,
    EnvData,
    batch,
)
from jaxvmas.simulator.scenario import BaseScenario

# Define dimensions for type hints
dim1 = "dim1"
dim2 = "dim2"


class MockWorld(World):
    """Mock world for testing."""

    pass


class MockScenario(BaseScenario):
    """Mock scenario for testing."""

    def make_world(self, batch_dim: int, **kwargs) -> World:
        return MockWorld.create(batch_dim=batch_dim)

    def reset_world_at(self, PRNG_key: Array, env_index: int | None) -> "MockScenario":
        return self

    def observation(self, agent: Agent) -> Array:
        return jnp.zeros((self.world.batch_dim, 2))

    def reward(self, agent: Agent) -> Array:
        return jnp.zeros(self.world.batch_dim)


class MockJaxGymWrapper(BaseJaxGymWrapper):
    """Mock wrapper for testing."""

    def step(self, action):
        if self.dict_spaces:
            obs = {
                agent.name: jnp.ones((self.env.batch_dim, 2))
                for agent in self.env.agents
            }
            rews = {
                agent.name: jnp.zeros(self.env.batch_dim) for agent in self.env.agents
            }
            info = {
                agent.name: {"test": jnp.ones(self.env.batch_dim)}
                for agent in self.env.agents
            }
        else:
            obs = [jnp.ones((self.env.batch_dim, 2)) for _ in self.env.agents]
            rews = [jnp.zeros(self.env.batch_dim) for _ in self.env.agents]
            info = [{"test": jnp.ones(self.env.batch_dim)} for _ in self.env.agents]

        terminated = jnp.zeros(self.env.batch_dim, dtype=bool)
        truncated = jnp.zeros(self.env.batch_dim, dtype=bool)
        done = jnp.zeros(self.env.batch_dim, dtype=bool)

        return self._convert_env_data(obs, rews, info, terminated, truncated, done)

    def reset(self, *, options=None):
        if self.dict_spaces:
            obs = {
                agent.name: jnp.ones((self.env.batch_dim, 2))
                for agent in self.env.agents
            }
            info = {
                agent.name: {"test": jnp.ones(self.env.batch_dim)}
                for agent in self.env.agents
            }
        else:
            obs = [jnp.ones((self.env.batch_dim, 2)) for _ in self.env.agents]
            info = [{"test": jnp.ones(self.env.batch_dim)} for _ in self.env.agents]
        return self._convert_env_data(obs=obs, info=info)

    def render(self, agent_index_focus=None, visualize_when_rgb=False, **kwargs):
        return jnp.zeros((64, 64, 3))


class TestEnvData:
    @pytest.fixture
    def env_data(self):
        """Create test environment data."""
        batch_dim = 2
        obs = {"agent_0": jnp.ones((batch_dim, 2)), "agent_1": jnp.ones((batch_dim, 2))}
        rews = {"agent_0": jnp.zeros(batch_dim), "agent_1": jnp.zeros(batch_dim)}
        terminated = jnp.zeros(batch_dim, dtype=bool)
        truncated = jnp.zeros(batch_dim, dtype=bool)
        done = jnp.zeros(batch_dim, dtype=bool)
        info = {
            "agent_0": {"test": jnp.ones(batch_dim)},
            "agent_1": {"test": jnp.ones(batch_dim)},
        }
        return EnvData(
            obs=obs,
            rews=rews,
            terminated=terminated,
            truncated=truncated,
            done=done,
            info=info,
        )

    def test_create(self, env_data):
        assert isinstance(env_data.obs, dict)
        assert isinstance(env_data.rews, dict)
        assert isinstance(env_data.terminated, Array)
        assert isinstance(env_data.truncated, Array)
        assert isinstance(env_data.done, Array)
        assert isinstance(env_data.info, dict)

    def test_jit_compatibility(self, env_data):
        @eqx.filter_jit
        def process_env_data(data: EnvData) -> Float[Array, f"{batch}"]:
            return data.rews["agent_0"]

        rews = process_env_data(env_data)
        assert rews.shape == (2,)
        assert jnp.all(rews == 0)


class TestBaseJaxGymWrapper:
    @pytest.fixture
    def wrapper(self):
        PRNG_key = jax.random.PRNGKey(0)
        key_step, key_step_i = jax.random.split(PRNG_key)
        env = Environment.create(
            scenario=MockScenario.create(), num_envs=2, PRNG_key=key_step_i
        )
        mock_agent_1 = Agent.create(name="agent_0", batch_dim=2, dim_c=2, dim_p=2)
        mock_agent_2 = Agent.create(name="agent_1", batch_dim=2, dim_c=2, dim_p=2)
        world = env.world
        world = world.add_agent(mock_agent_1)
        world = world.add_agent(mock_agent_2)
        env = env.replace(world=world)
        env, _ = env.reset(PRNG_key=key_step)
        return MockJaxGymWrapper.create(env=env, vectorized=True)

    def test_create(self, wrapper: BaseJaxGymWrapper):
        assert isinstance(wrapper.env, Environment)
        assert wrapper.dict_spaces is False
        assert wrapper.vectorized is True

    def test_convert_output(self, wrapper: BaseJaxGymWrapper):
        data = jnp.ones((2, 2))

        # Test vectorized output
        converted = wrapper._convert_output(data)
        assert converted.shape == (2, 2)

        # Test non-vectorized output
        wrapper = wrapper.replace(vectorized=False)
        converted = wrapper._convert_output(data)
        assert converted.shape == (2,)

    def test_compress_infos(self, wrapper: BaseJaxGymWrapper):
        # Test dict input
        dict_info = {"agent_0": 1, "agent_1": 2}
        compressed = wrapper._compress_infos(dict_info)
        assert compressed == dict_info

        # Test list input
        list_info = [1, 2]
        compressed = wrapper._compress_infos(list_info)
        assert compressed == {"agent_0": 1, "agent_1": 2}

        # Test invalid input
        with pytest.raises(ValueError):
            wrapper._compress_infos(42)

    def test_action_list_to_array(self, wrapper: BaseJaxGymWrapper):
        actions = [jnp.ones((2, 2)), jnp.zeros((2, 2))]
        converted = wrapper._action_list_to_array(actions)
        assert len(converted) == 2
        assert converted[0].shape == (2, 2)
        assert converted[1].shape == (2, 2)

        # Test invalid number of actions
        with pytest.raises(AssertionError):
            wrapper._action_list_to_array([jnp.ones(2)])

    def test_step_jit(self, wrapper: BaseJaxGymWrapper):
        @eqx.filter_jit
        def step(wrapper: BaseJaxGymWrapper, action):
            return wrapper.step(action)

        action = [jnp.ones((2, 2)), jnp.zeros((2, 2))]
        result = step(wrapper, action)
        assert isinstance(result, EnvData)
        assert isinstance(result.obs, list)
        assert isinstance(result.rews, list)
        assert result.terminated.shape == (2,)
        assert result.truncated.shape == (2,)
        assert result.done.shape == (2,)

    def test_reset_jit(self, wrapper: BaseJaxGymWrapper):
        @eqx.filter_jit
        def reset(wrapper: BaseJaxGymWrapper):
            return wrapper.reset()

        result = reset(wrapper)
        assert isinstance(result, EnvData)
        assert isinstance(result.obs, list)
        assert isinstance(result.info, dict)

    def test_render_jit(self, wrapper: BaseJaxGymWrapper):
        @eqx.filter_jit
        def render(wrapper: BaseJaxGymWrapper):
            return wrapper.render()

        result = render(wrapper)
        assert isinstance(result, Array)
        assert result.shape == (64, 64, 3)

    def test_convert_env_data_jit(self, wrapper: BaseJaxGymWrapper):
        @eqx.filter_jit
        def convert_data(
            wrapper: BaseJaxGymWrapper,
            obs: dict,
            rews: dict,
            info: dict,
            terminated: Array,
            truncated: Array,
            done: Array,
        ) -> EnvData:
            return wrapper._convert_env_data(
                obs, rews, info, terminated, truncated, done
            )

        wrapper = wrapper.replace(dict_spaces=True)
        obs = {
            "agent_0": jnp.ones((2, 2)),
            "agent_1": jnp.ones((2, 2)),
        }
        rews = {
            "agent_0": jnp.zeros(2),
            "agent_1": jnp.zeros(2),
        }
        info = {
            "agent_0": {"test": jnp.ones(2)},
            "agent_1": {"test": jnp.ones(2)},
        }
        terminated = jnp.zeros(2, dtype=bool)
        truncated = jnp.zeros(2, dtype=bool)
        done = jnp.zeros(2, dtype=bool)

        result = convert_data(wrapper, obs, rews, info, terminated, truncated, done)
        assert isinstance(result, EnvData)
        assert isinstance(result.obs, dict)
        assert isinstance(result.rews, dict)
        assert result.terminated.shape == (2,)
        assert result.truncated.shape == (2,)
        assert result.done.shape == (2,)
