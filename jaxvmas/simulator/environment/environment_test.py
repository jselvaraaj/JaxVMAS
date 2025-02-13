import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Float

from jaxvmas.simulator.core import Agent, World
from jaxvmas.simulator.environment.environment import Environment
from jaxvmas.simulator.scenario import BaseScenario

# Dimension type variables (add near top of file)
batch_dim = "batch"
pos_dim = "dim_p"
dots_dim = "..."


class MockScenario(BaseScenario):
    """Mock scenario for testing"""

    def __init__(self, num_agents: int = 2):
        super().__init__()
        self.num_agents = num_agents
        self.viewer_size = (400, 300)
        self.viewer_zoom = 1.0
        self.render_origin = jnp.zeros(2)

    def env_make_world(self, num_envs: int, **kwargs) -> World:
        world = World.create(batch_dim=num_envs)

        # Add agents
        for i in range(self.num_agents):
            agent = Agent.create(
                batch_dim=num_envs,
                name=f"agent_{i}",
                dim_c=3,
                dim_p=2,
                movable=True,
            )
            world = world.add_agent(agent)

        return world

    def observation(self, agent: Agent) -> Float[Array, f"{batch_dim} {pos_dim}"]:
        # Simple observation of agent position
        return agent.state.pos

    def reward(self, agent: Agent) -> Float[Array, f"{batch_dim}"]:
        # Simple reward of -1
        return -jnp.ones(agent.batch_dim)

    def done(self) -> Float[Array, f"{batch_dim}"]:
        # Never done
        return jnp.zeros(self.world.batch_dim, dtype=bool)

    def info(self, agent: Agent) -> dict[str, Float[Array, f"{batch_dim} {dots_dim}"]]:
        # Empty info
        return {}


class TestEnvironment:
    @pytest.fixture
    def basic_env(self):
        """Basic environment fixture with default settings"""
        scenario = MockScenario()
        return Environment.create(
            scenario=scenario,
            num_envs=32,
            max_steps=100,
            continuous_actions=True,
            seed=0,
        )

    @pytest.fixture
    def discrete_env(self):
        """Environment fixture with discrete actions"""
        scenario = MockScenario()
        return Environment.create(
            scenario=scenario,
            num_envs=32,
            max_steps=100,
            continuous_actions=False,
            seed=0,
        )

    def test_create(self, basic_env):
        """Test environment creation and initialization"""
        assert basic_env.num_envs == 32
        assert basic_env.max_steps == 100
        assert basic_env.continuous_actions is True
        assert basic_env.n_agents == 2
        assert isinstance(basic_env.PRNG_key, jax.Array)
        assert basic_env.steps.shape == (32,)
        assert jnp.all(basic_env.steps == 0)

    def test_reset(self, basic_env):
        """Test environment reset functionality"""
        # Full reset
        obs, env = basic_env.reset()
        assert isinstance(obs, list)
        assert len(obs) == basic_env.n_agents
        assert obs[0].shape == (basic_env.num_envs, 2)  # pos observation
        assert jnp.all(env.steps == 0)

        # Reset with info
        result, env = basic_env.reset(return_info=True)
        obs, info = result
        assert isinstance(info, list)
        assert len(info) == basic_env.n_agents

        # Reset with dones
        result, env = basic_env.reset(return_dones=True)
        obs, dones = result
        assert dones.shape == (basic_env.num_envs,)

    def test_reset_at(self, basic_env):
        """Test resetting specific environment"""
        # Step environment first
        actions = [jnp.ones((32, 2)) for _ in range(basic_env.n_agents)]
        basic_env.step(actions)

        # Reset specific environment
        result, env = basic_env.reset_at(0)
        assert isinstance(result, list)
        assert len(result) == basic_env.n_agents
        assert jnp.all(env.steps[0] == 0)
        assert jnp.all(env.steps[1:] == 1)

        # Test invalid index
        with pytest.raises(AssertionError):
            basic_env.reset_at(32)

    def test_step(self, basic_env):
        """Test environment stepping"""
        # Test with valid actions
        actions = [jnp.ones((32, 2)) for _ in range(basic_env.n_agents)]
        obs, rewards, dones, infos = basic_env.step(actions)

        assert len(obs) == basic_env.n_agents
        assert len(rewards) == basic_env.n_agents
        assert dones.shape == (basic_env.num_envs,)
        assert len(infos) == basic_env.n_agents

        # Test with dict actions
        actions_dict = {f"agent_{i}": act for i, act in enumerate(actions)}
        obs, rewards, dones, infos = basic_env.step(actions_dict)

        assert len(obs) == basic_env.n_agents
        assert len(rewards) == basic_env.n_agents

        # Test invalid action shape
        with pytest.raises(AssertionError):
            invalid_actions = [jnp.ones((31, 2)) for _ in range(basic_env.n_agents)]
            basic_env.step(invalid_actions)

    def test_action_space(self, basic_env, discrete_env):
        """Test action space configuration"""
        # Test continuous action space
        assert basic_env.action_space is not None
        assert len(basic_env.action_space.spaces) == basic_env.n_agents

        # Test discrete action space
        assert discrete_env.action_space is not None
        assert len(discrete_env.action_space.spaces) == discrete_env.n_agents

        # Test with dict spaces
        dict_env = Environment.create(
            scenario=MockScenario(),
            num_envs=32,
            dict_spaces=True,
        )
        assert isinstance(dict_env.action_space.spaces, dict)
        assert len(dict_env.action_space.spaces) == dict_env.n_agents

    def test_observation_space(self, basic_env):
        """Test observation space configuration"""
        assert basic_env.observation_space is not None
        assert len(basic_env.observation_space.spaces) == basic_env.n_agents

        # Test with dict spaces
        dict_env = Environment.create(
            scenario=MockScenario(),
            num_envs=32,
            dict_spaces=True,
        )
        assert isinstance(dict_env.observation_space.spaces, dict)
        assert len(dict_env.observation_space.spaces) == dict_env.n_agents

    def test_random_actions(self, basic_env, discrete_env):
        """Test random action generation"""
        # Test continuous actions
        random_actions = basic_env.get_random_actions()
        assert len(random_actions) == basic_env.n_agents
        assert random_actions[0].shape == (basic_env.num_envs, 2)

        # Test discrete actions
        random_discrete = discrete_env.get_random_actions()
        assert len(random_discrete) == discrete_env.n_agents
        assert random_discrete[0].shape == (discrete_env.num_envs,)

    def test_max_steps(self):
        """Test max steps functionality"""
        env = Environment.create(
            scenario=MockScenario(),
            num_envs=32,
            max_steps=2,
        )

        # Step until max steps
        actions = [jnp.ones((32, 2)) for _ in range(env.n_agents)]
        _, _, dones, _ = env.step(actions)
        assert not jnp.any(dones)

        _, _, dones, _ = env.step(actions)
        assert jnp.all(dones)

    def test_multidiscrete_actions(self):
        """Test multidiscrete action space"""
        env = Environment.create(
            scenario=MockScenario(),
            num_envs=32,
            continuous_actions=False,
            multidiscrete_actions=True,
        )

        # Test action space
        for space in env.action_space.spaces:
            assert space.shape == (2,)  # Default 2D actions

        # Test random actions
        random_actions = env.get_random_actions()
        assert len(random_actions) == env.n_agents
        assert random_actions[0].shape == (env.num_envs, 2)

    def test_terminated_truncated(self):
        """Test terminated/truncated functionality"""
        env = Environment.create(
            scenario=MockScenario(),
            num_envs=32,
            max_steps=2,
            terminated_truncated=True,
        )

        actions = [jnp.ones((32, 2)) for _ in range(env.n_agents)]
        _, _, terminated, truncated, _ = env.step(actions)

        assert terminated.shape == (env.num_envs,)
        assert truncated.shape == (env.num_envs,)
        assert not jnp.any(terminated)
        assert not jnp.any(truncated)

        _, _, terminated, truncated, _ = env.step(actions)
        assert not jnp.any(terminated)
        assert jnp.all(truncated)

    def test_is_jittable(self, basic_env):
        """Test jit compatibility"""

        @jax.jit
        def reset_env(env):
            return env.reset()

        reset_env(basic_env)

        @jax.jit
        def step_env(env, actions):
            return env.step(actions)

        actions = [jnp.ones((32, 2)) for _ in range(basic_env.n_agents)]
        step_env(basic_env, actions)
