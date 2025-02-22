import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Float

from jaxvmas.equinox_utils import equinox_filter_cond_return_pytree_node
from jaxvmas.simulator.core.agent import Agent
from jaxvmas.simulator.core.world import World
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
        world = World.create(batch_dim=batch_dim, dim_c=3, dim_p=2)

        # Add agents with different configurations
        agent1 = Agent.create(
            name="agent_1",
            movable=True,
            silent=False,
        )
        agent1 = agent1._spawn(id=jnp.asarray(1), batch_dim=batch_dim, dim_c=3, dim_p=2)
        agent2 = Agent.create(
            name="agent_2",
            movable=True,
            silent=True,
        )
        agent2 = agent2._spawn(id=jnp.asarray(2), batch_dim=batch_dim, dim_c=3, dim_p=2)
        world = world.add_agent(agent1)
        world = world.add_agent(agent2)
        return world

    def reset_world_at(self, PRNG_key: Array, env_index: int | float) -> "MockScenario":
        # Reset all agents in the world at the specified index
        world = equinox_filter_cond_return_pytree_node(
            jnp.isnan(env_index),
            lambda world: world.reset(env_index=jnp.nan),
            lambda world: world.reset(env_index=env_index),
            self.world,
        )
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
        PRNG_key = jax.random.PRNGKey(0)
        PRNG_key, sub_key = jax.random.split(PRNG_key)
        return (
            Environment.create(
                scenario=scenario,
                num_envs=batch_dim,
                max_steps=100,
                continuous_actions=True,
                PRNG_key=sub_key,
            ),
            PRNG_key,
        )

    @pytest.fixture
    def discrete_env(self):
        """Environment fixture with discrete actions"""
        scenario = MockScenario.create()
        PRNG_key = jax.random.PRNGKey(0)
        PRNG_key, sub_key = jax.random.split(PRNG_key)
        return (
            Environment.create(
                scenario=scenario,
                num_envs=32,
                max_steps=100,
                continuous_actions=False,
                PRNG_key=sub_key,
            ),
            PRNG_key,
        )

    def test_create(self, basic_env: tuple[Environment, Array]):
        """Test environment creation and initialization"""
        env, PRNG_key = basic_env
        assert env.num_envs == 32
        assert env.max_steps == 100
        assert env.continuous_actions is True
        assert env.n_agents == 2
        assert isinstance(PRNG_key, jax.Array)
        assert env.steps.shape == (32,)
        assert jnp.all(env.steps == 0)

    def test_reset(self, basic_env: tuple[Environment, Array]):
        """Test environment reset functionality"""
        # Full reset
        env, PRNG_key = basic_env
        key_step, key_step_i = jax.random.split(PRNG_key)
        env, obs = env.reset(PRNG_key=key_step_i)
        assert isinstance(obs, list)
        assert len(obs) == env.n_agents
        assert obs[0].shape == (env.num_envs, 2)  # pos observation
        assert jnp.all(env.steps == 0)

        # Reset with info
        key_step, key_step_i = jax.random.split(PRNG_key)
        env, result = env.reset(PRNG_key=key_step_i, return_info=True)
        obs, info = result
        assert isinstance(info, list)
        assert len(info) == env.n_agents

        # Reset with dones
        key_step, key_step_i = jax.random.split(PRNG_key)
        env, result = env.reset(PRNG_key=key_step_i, return_dones=True)
        obs, dones = result
        assert dones.shape == (env.num_envs,)

    def test_reset_at(self, basic_env: tuple[Environment, Array]):
        """Test resetting specific environment"""
        env, PRNG_key = basic_env

        # Step environment first
        actions = [jnp.ones((32, 5)), jnp.ones((32, 2))]
        PRNG_key, key_step_i = jax.random.split(PRNG_key)
        env, _ = env.step(PRNG_key=key_step_i, actions=actions)

        # Reset specific environment
        PRNG_key, key_step_i = jax.random.split(PRNG_key)
        env, result = env.reset_at(PRNG_key=key_step_i, index=0)
        assert isinstance(result, list)
        assert len(result) == env.n_agents
        assert jnp.all(env.steps[0] == 0)
        assert jnp.all(env.steps[1:] == 1)
        with jax.disable_jit(True):
            # Test invalid index
            with pytest.raises(AssertionError):
                PRNG_key, key_step_i = jax.random.split(PRNG_key)
                env.reset_at(PRNG_key=key_step_i, index=32)

    def test_step(self, basic_env: tuple[Environment, Array]):
        """Test environment stepping"""
        env, PRNG_key = basic_env
        # Test with valid actions
        actions = [jnp.ones((32, 5)), jnp.ones((32, 2))]
        PRNG_key, key_step_i = jax.random.split(PRNG_key)
        env, result = env.step(PRNG_key=key_step_i, actions=actions)
        obs, rewards, dones, infos = result

        assert len(obs) == env.n_agents
        assert len(rewards) == env.n_agents
        assert dones.shape == (env.num_envs,)
        assert len(infos) == env.n_agents

        # Test with dict actions
        actions_dict = {f"agent_{i+1}": act for i, act in enumerate(actions)}
        PRNG_key, key_step_i = jax.random.split(PRNG_key)
        env, result = env.step(PRNG_key=key_step_i, actions=actions_dict)
        obs, rewards, dones, infos = result

        assert len(obs) == env.n_agents
        assert len(rewards) == env.n_agents

        # Test invalid action shape
        with pytest.raises(AssertionError):
            invalid_actions = [jnp.ones((31, 2)) for _ in range(env.n_agents)]
            PRNG_key, key_step_i = jax.random.split(PRNG_key)
            env, result = env.step(PRNG_key=key_step_i, actions=invalid_actions)
            obs, rewards, dones, infos = result

    def test_action_space(
        self,
        basic_env: tuple[Environment, Array],
        discrete_env: tuple[Environment, Array],
    ):
        """Test action space configuration"""
        env, PRNG_key = basic_env
        discrete_env, PRNG_key = discrete_env
        # Test continuous action space
        assert isinstance(env.action_space, Tuple)
        assert len(env.action_space.spaces) == env.n_agents
        assert isinstance(env.action_space.spaces[0], Box)

        # Test discrete action space
        assert isinstance(discrete_env.action_space, Tuple)
        assert len(discrete_env.action_space.spaces) == discrete_env.n_agents
        assert isinstance(discrete_env.action_space.spaces[0], Discrete)

        # Test with dict spaces
        PRNG_key, sub_key = jax.random.split(PRNG_key)
        dict_env = Environment.create(
            scenario=MockScenario.create(),
            PRNG_key=sub_key,
            num_envs=32,
            dict_spaces=True,
        )
        assert isinstance(dict_env.action_space, Dict)
        assert len(dict_env.action_space.spaces) == dict_env.n_agents

    def test_observation_space(self, basic_env: tuple[Environment, Array]):
        """Test observation space configuration"""
        env, PRNG_key = basic_env
        assert isinstance(env.observation_space, Tuple)
        assert len(env.observation_space.spaces) == env.n_agents

        # Test with dict spaces
        PRNG_key, sub_key = jax.random.split(PRNG_key)
        dict_env = Environment.create(
            scenario=MockScenario.create(),
            PRNG_key=sub_key,
            num_envs=32,
            dict_spaces=True,
        )
        assert isinstance(dict_env.observation_space, Dict)
        assert len(dict_env.observation_space.spaces) == dict_env.n_agents

    def test_random_actions(
        self,
        basic_env: tuple[Environment, Array],
        discrete_env: tuple[Environment, Array],
    ):
        """Test random action generation"""
        # Test continuous actions
        env, PRNG_key = basic_env
        discrete_env, PRNG_key = discrete_env
        PRNG_key, key_step_i = jax.random.split(PRNG_key)
        random_actions = env.get_random_actions(PRNG_key=key_step_i)
        assert len(random_actions) == env.n_agents
        assert random_actions[0].shape == (env.num_envs, 5)
        assert random_actions[1].shape == (env.num_envs, 2)

        # Test discrete actions
        PRNG_key, key_step_i = jax.random.split(PRNG_key)
        random_discrete = discrete_env.get_random_actions(PRNG_key=key_step_i)
        assert len(random_discrete) == discrete_env.n_agents
        assert random_discrete[0].shape == (discrete_env.num_envs,)

    def test_max_steps(self):
        """Test max steps functionality"""
        PRNG_key = jax.random.PRNGKey(0)
        env = Environment.create(
            scenario=MockScenario.create(),
            num_envs=32,
            max_steps=2,
            PRNG_key=PRNG_key,
        )

        # Step until max steps
        actions = [jnp.ones((32, 5)), jnp.ones((32, 2))]
        PRNG_key, key_step_i = jax.random.split(PRNG_key)
        env, result = env.step(PRNG_key=key_step_i, actions=actions)
        _, _, dones, _ = result
        assert not jnp.any(dones)

        PRNG_key, key_step_i = jax.random.split(PRNG_key)
        env, result = env.step(PRNG_key=key_step_i, actions=actions)
        _, _, dones, _ = result
        assert jnp.all(dones)

    def test_multidiscrete_actions(self):
        """Test multidiscrete action space"""
        PRNG_key = jax.random.PRNGKey(0)
        PRNG_key, sub_key = jax.random.split(PRNG_key)
        env = Environment.create(
            scenario=MockScenario.create(),
            num_envs=32,
            continuous_actions=False,
            multidiscrete_actions=True,
            PRNG_key=sub_key,
        )

        # Test action space
        assert isinstance(env.action_space, Tuple)
        assert isinstance(env.action_space.spaces[0], MultiDiscrete)
        assert isinstance(env.action_space.spaces[1], MultiDiscrete)
        assert env.action_space.spaces[0].shape == (
            3,
        )  # Default 2D actions + 1 communication action
        assert env.action_space.spaces[1].shape == (2,)  # Default 2D actions

        # Test random actions
        PRNG_key, sub_key = jax.random.split(PRNG_key)
        random_actions = env.get_random_actions(PRNG_key=sub_key)
        assert len(random_actions) == env.n_agents
        assert random_actions[0].shape == (env.num_envs, 3)
        assert random_actions[1].shape == (env.num_envs, 2)

    def test_terminated_truncated(self):
        """Test terminated/truncated functionality"""
        PRNG_key = jax.random.PRNGKey(0)
        PRNG_key, sub_key = jax.random.split(PRNG_key)
        env = Environment.create(
            scenario=MockScenario.create(),
            num_envs=32,
            max_steps=2,
            terminated_truncated=True,
            PRNG_key=sub_key,
        )

        actions = [jnp.ones((32, 5)), jnp.ones((32, 2))]
        PRNG_key, key_step_i = jax.random.split(PRNG_key)
        env, result = env.step(PRNG_key=key_step_i, actions=actions)
        _, _, terminated, truncated, _ = result

        assert terminated.shape == (env.num_envs,)
        assert truncated.shape == (env.num_envs,)
        assert not jnp.any(terminated)
        assert not jnp.any(truncated)

        PRNG_key, key_step_i = jax.random.split(PRNG_key)
        env, result = env.step(PRNG_key=key_step_i, actions=actions)
        _, _, terminated, truncated, _ = result
        assert not jnp.any(terminated)
        assert jnp.all(truncated)

    def test_is_jittable(self, basic_env: tuple[Environment, Array]):
        """Test jit compatibility of all major functions"""
        env, PRNG_key = basic_env

        # Test jit compatibility of reset
        @eqx.filter_jit
        def reset_env(env: Environment, PRNG_key: Array):
            return env.reset(PRNG_key=PRNG_key)

        PRNG_key, key_step_i = jax.random.split(PRNG_key)
        env, obs = reset_env(env, key_step_i)
        assert len(obs) == env.n_agents

        # Test jit compatibility of reset_at
        @eqx.filter_jit
        def reset_at_env(env: Environment, PRNG_key: Array, index: int):
            return env.reset_at(PRNG_key=PRNG_key, index=index)

        PRNG_key, key_step_i = jax.random.split(PRNG_key)
        env, obs = reset_at_env(env, key_step_i, 0)
        assert len(obs) == env.n_agents

        # Test jit compatibility of step
        @eqx.filter_jit
        def step_env(env: Environment, PRNG_key: Array, actions: list):
            return env.step(PRNG_key=PRNG_key, actions=actions)

        PRNG_key, key_step_i = jax.random.split(PRNG_key)
        actions = [jnp.ones((32, 5)), jnp.ones((32, 2))]
        env, result = step_env(env, key_step_i, actions)
        obs, rewards, dones, infos = result
        assert len(obs) == env.n_agents

        # Test jit compatibility of random actions
        @eqx.filter_jit
        def get_random_actions(PRNG_key: Array, env: Environment):
            PRNG_key, sub_key = jax.random.split(PRNG_key)
            return env.get_random_actions(PRNG_key=sub_key)

        PRNG_key, key_step_i = jax.random.split(PRNG_key)
        random_actions = get_random_actions(key_step_i, env)
        assert len(random_actions) == env.n_agents

        # Test jit compatibility of done computation
        @eqx.filter_jit
        def compute_done(env: Environment):
            return env.done()

        done = compute_done(env)
        assert done.shape == (env.num_envs,)

        # Test jit compatibility of get_from_scenario
        @eqx.filter_jit
        def get_scenario_data(env: Environment):
            return env.get_from_scenario(
                get_observations=True,
                get_rewards=True,
                get_infos=True,
                get_dones=True,
            )

        result = get_scenario_data(env)
        assert len(result) == 4  # obs, rewards, dones, infos
