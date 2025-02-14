import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Float

from jaxvmas.simulator.core import Agent, World
from jaxvmas.simulator.environment.environment import Environment
from jaxvmas.simulator.environment.jaxgym.spaces import (
    Box,
    Dict,
    Discrete,
    MultiDiscrete,
    Tuple,
)
from jaxvmas.simulator.scenario import BaseScenario

# Dimension type variables
batch_dim = "batch"
pos_dim = "dim_p"
dots_dim = "..."


class MockScenario(BaseScenario):
    """Mock scenario for testing"""

    def make_world(self, batch_dim: int, **kwargs) -> World:
        world = World.create(batch_dim=batch_dim)

        # Add agents with different configurations
        agent1 = Agent.create(
            batch_dim=batch_dim,
            name="agent_1",
            dim_c=3,
            dim_p=2,
            movable=True,
        )
        agent2 = Agent.create(
            batch_dim=batch_dim,
            name="agent_2",
            dim_c=0,  # No communication
            dim_p=2,
            movable=True,
            silent=True,
        )
        world = world.add_agent(agent1)
        world = world.add_agent(agent2)
        return world

    def reset_world_at(self, env_index: int | None) -> "MockScenario":
        # Reset all agents in the world at the specified index
        world = self.world
        if env_index is not None:
            world = world.reset(env_index)
        else:
            # Reset all environments
            world = world.reset(None)
        self = self.replace(world=world)
        return self

    def observation(self, agent: Agent) -> Float[Array, f"{batch_dim} {pos_dim}"]:
        return agent.state.pos

    def reward(self, agent: Agent) -> Float[Array, f"{batch_dim}"]:
        return -jnp.ones(self.world.batch_dim)

    def info(self, agent: Agent) -> dict[str, Float[Array, f"{batch_dim} {dots_dim}"]]:
        return {"test_info": jnp.ones(self.world.batch_dim)}


class TestEnvironment:
    @pytest.fixture
    def basic_env(self):
        """Basic environment fixture with default settings"""
        scenario = MockScenario.create()
        batch_dim = 32
        return Environment.create(
            scenario=scenario,
            num_envs=batch_dim,
            max_steps=100,
            continuous_actions=True,
            seed=0,
        )

    @pytest.fixture
    def discrete_env(self):
        """Environment fixture with discrete actions"""
        scenario = MockScenario.create()
        return Environment.create(
            scenario=scenario,
            num_envs=32,
            max_steps=100,
            continuous_actions=False,
            seed=0,
        )

    def test_create(self, basic_env: Environment):
        """Test environment creation and initialization"""
        assert basic_env.num_envs == 32
        assert basic_env.max_steps == 100
        assert basic_env.continuous_actions is True
        assert basic_env.n_agents == 2
        assert isinstance(basic_env.PRNG_key, jax.Array)
        assert basic_env.steps.shape == (32,)
        assert jnp.all(basic_env.steps == 0)

    def test_reset(self, basic_env: Environment):
        """Test environment reset functionality"""
        # Full reset
        env, obs = basic_env.reset()
        assert isinstance(obs, list)
        assert len(obs) == basic_env.n_agents
        assert obs[0].shape == (basic_env.num_envs, 2)  # pos observation
        assert jnp.all(env.steps == 0)

        # Reset with info
        env, result = basic_env.reset(return_info=True)
        obs, info = result
        assert isinstance(info, list)
        assert len(info) == basic_env.n_agents

        # Reset with dones
        env, result = basic_env.reset(return_dones=True)
        obs, dones = result
        assert dones.shape == (basic_env.num_envs,)

    def test_reset_at(self, basic_env: Environment):
        """Test resetting specific environment"""
        # Step environment first
        actions = [jnp.ones((32, 2)) for _ in range(basic_env.n_agents)]
        basic_env, _ = basic_env.step(actions)

        # Reset specific environment
        env, result = basic_env.reset_at(0)
        assert isinstance(result, list)
        assert len(result) == basic_env.n_agents
        assert jnp.all(env.steps[0] == 0)
        assert jnp.all(env.steps[1:] == 1)

        # Test invalid index
        with pytest.raises(AssertionError):
            basic_env.reset_at(32)

    def test_step(self, basic_env: Environment):
        """Test environment stepping"""
        # Test with valid actions
        actions = [jnp.ones((32, 2)) for _ in range(basic_env.n_agents)]
        env, result = basic_env.step(actions)
        obs, rewards, dones, infos = result

        assert len(obs) == basic_env.n_agents
        assert len(rewards) == basic_env.n_agents
        assert dones.shape == (basic_env.num_envs,)
        assert len(infos) == basic_env.n_agents

        # Test with dict actions
        actions_dict = {f"agent_{i+1}": act for i, act in enumerate(actions)}
        env, result = basic_env.step(actions_dict)
        obs, rewards, dones, infos = result

        assert len(obs) == basic_env.n_agents
        assert len(rewards) == basic_env.n_agents

        # Test invalid action shape
        with pytest.raises(AssertionError):
            invalid_actions = [jnp.ones((31, 2)) for _ in range(basic_env.n_agents)]
            env, result = basic_env.step(invalid_actions)
            obs, rewards, dones, infos = result

    def test_action_space(self, basic_env: Environment, discrete_env: Environment):
        """Test action space configuration"""
        # Test continuous action space
        assert isinstance(basic_env.action_space, Tuple)
        assert len(basic_env.action_space.spaces) == basic_env.n_agents
        assert isinstance(basic_env.action_space.spaces[0], Box)

        # Test discrete action space
        assert isinstance(discrete_env.action_space, Tuple)
        assert len(discrete_env.action_space.spaces) == discrete_env.n_agents
        assert isinstance(discrete_env.action_space.spaces[0], Discrete)

        # Test with dict spaces
        dict_env = Environment.create(
            scenario=MockScenario.create(),
            num_envs=32,
            dict_spaces=True,
        )
        assert isinstance(dict_env.action_space, Dict)
        assert len(dict_env.action_space.spaces) == dict_env.n_agents

    def test_observation_space(self, basic_env: Environment):
        """Test observation space configuration"""
        assert isinstance(basic_env.observation_space, Tuple)
        assert len(basic_env.observation_space.spaces) == basic_env.n_agents

        # Test with dict spaces
        dict_env = Environment.create(
            scenario=MockScenario.create(),
            num_envs=32,
            dict_spaces=True,
        )
        assert isinstance(dict_env.observation_space, Dict)
        assert len(dict_env.observation_space.spaces) == dict_env.n_agents

    def test_random_actions(self, basic_env: Environment, discrete_env: Environment):
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
            scenario=MockScenario.create(),
            num_envs=32,
            max_steps=2,
        )

        # Step until max steps
        actions = [jnp.ones((32, 2)) for _ in range(env.n_agents)]
        env, result = env.step(actions)
        _, _, dones, _ = result
        assert not jnp.any(dones)

        env, result = env.step(actions)
        _, _, dones, _ = result
        assert jnp.all(dones)

    def test_multidiscrete_actions(self):
        """Test multidiscrete action space"""
        env = Environment.create(
            scenario=MockScenario.create(),
            num_envs=32,
            continuous_actions=False,
            multidiscrete_actions=True,
        )

        # Test action space
        assert isinstance(env.action_space, Tuple)
        for space in env.action_space.spaces:
            assert isinstance(space, MultiDiscrete)
            assert space.shape == (2,)  # Default 2D actions

        # Test random actions
        random_actions = env.get_random_actions()
        assert len(random_actions) == env.n_agents
        assert random_actions[0].shape == (env.num_envs, 2)

    def test_terminated_truncated(self):
        """Test terminated/truncated functionality"""
        env = Environment.create(
            scenario=MockScenario.create(),
            num_envs=32,
            max_steps=2,
            terminated_truncated=True,
        )

        actions = [jnp.ones((32, 2)) for _ in range(env.n_agents)]
        env, result = env.step(actions)
        _, _, terminated, truncated, _ = result

        assert terminated.shape == (env.num_envs,)
        assert truncated.shape == (env.num_envs,)
        assert not jnp.any(terminated)
        assert not jnp.any(truncated)

        env, result = env.step(actions)
        _, _, terminated, truncated, _ = result
        assert not jnp.any(terminated)
        assert jnp.all(truncated)

    def test_is_jittable(self, basic_env: Environment):
        """Test jit compatibility of all major functions"""

        # Test jit compatibility of reset
        @eqx.filter_jit
        def reset_env(env: Environment):
            return env.reset()

        env, obs = reset_env(basic_env)
        assert len(obs) == basic_env.n_agents

        # Test jit compatibility of reset_at
        @eqx.filter_jit
        def reset_at_env(env: Environment, index: int):
            return env.reset_at(index)

        env, obs = reset_at_env(basic_env, 0)
        assert len(obs) == basic_env.n_agents

        # Test jit compatibility of step
        @eqx.filter_jit
        def step_env(env: Environment, actions: list):
            return env.step(actions)

        actions = [jnp.ones((32, 2)) for _ in range(basic_env.n_agents)]
        env, result = step_env(basic_env, actions)
        obs, rewards, dones, infos = result
        assert len(obs) == basic_env.n_agents

        # Test jit compatibility of random actions
        @eqx.filter_jit
        def get_random_actions(env: Environment):
            return env.get_random_actions()

        random_actions = get_random_actions(basic_env)
        assert len(random_actions) == basic_env.n_agents

        # Test jit compatibility of done computation
        @eqx.filter_jit
        def compute_done(env: Environment):
            return env.done()

        done = compute_done(basic_env)
        assert done.shape == (basic_env.num_envs,)

        # Test jit compatibility of get_from_scenario
        @eqx.filter_jit
        def get_scenario_data(env: Environment):
            return env.get_from_scenario(
                get_observations=True,
                get_rewards=True,
                get_infos=True,
                get_dones=True,
            )

        result = get_scenario_data(basic_env)
        assert len(result) == 4  # obs, rewards, dones, infos
