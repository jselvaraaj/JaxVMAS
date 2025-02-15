import equinox as eqx
import jax.numpy as jnp
import pytest

from jaxvmas.scenario.mpe.simple import Scenario
from jaxvmas.simulator.utils import Color

# Dimension type variables
batch_dim = "batch"
pos_dim = "dim_p"
vel_dim = "dim_p"
dots_dim = "..."


class TestSimpleScenario:
    @pytest.fixture
    def scenario(self) -> Scenario:
        """Basic scenario fixture with default settings"""
        return Scenario.create()

    @pytest.fixture
    def world(self, scenario: Scenario):
        """World fixture with batch dimension of 32"""
        return scenario.make_world(batch_dim=32)

    def test_make_world(self, scenario: Scenario):
        """Test world creation and initialization"""
        batch_dim = 32
        world = scenario.make_world(batch_dim=batch_dim)

        # Test world properties
        assert world.batch_dim == batch_dim
        assert world.dim_p == 2
        assert len(world.agents) == 1
        assert len(world.landmarks) == 1

        # Test agent properties
        agent = world.agents[0]
        assert agent.name == "agent_0"
        assert agent.collide is False
        assert agent.color == Color.GRAY

        # Test landmark properties
        landmark = world.landmarks[0]
        assert landmark.name == "landmark 0"
        assert landmark.collide is False
        assert landmark.color == Color.RED

    @eqx.filter_jit
    def test_reset_world_at_jit(self, scenario: Scenario):
        """Test that reset_world_at is jit-compatible"""
        batch_dim = 32
        world = scenario.make_world(batch_dim=batch_dim)
        scenario = scenario.replace(world=world)

        # Test full reset
        scenario = scenario.reset_world_at(seed=0)
        assert scenario.world.batch_dim == batch_dim

        # Test single environment reset
        scenario = scenario.reset_world_at(seed=0, env_index=0)
        assert scenario.world.batch_dim == batch_dim

    def test_reset_world_at(self, scenario: Scenario):
        """Test reset functionality and state changes"""
        batch_dim = 32
        world = scenario.make_world(batch_dim=batch_dim)
        scenario = scenario.replace(world=world)

        # Test full reset
        scenario = scenario.reset_world_at(seed=0)
        agent_pos = scenario.world.agents[0].state.pos
        landmark_pos = scenario.world.landmarks[0].state.pos

        assert agent_pos.shape == (batch_dim, 2)
        assert landmark_pos.shape == (batch_dim, 2)
        assert jnp.all((agent_pos >= -1.0) & (agent_pos <= 1.0))
        assert jnp.all((landmark_pos >= -1.0) & (landmark_pos <= 1.0))

        # Test single environment reset
        scenario = scenario.reset_world_at(seed=0, env_index=0)
        agent_pos_single = scenario.world.agents[0].state.pos[0]
        landmark_pos_single = scenario.world.landmarks[0].state.pos[0]

        assert agent_pos_single.shape == (2,)
        assert landmark_pos_single.shape == (2,)
        assert jnp.all((agent_pos_single >= -1.0) & (agent_pos_single <= 1.0))
        assert jnp.all((landmark_pos_single >= -1.0) & (landmark_pos_single <= 1.0))

    def test_reward_jit(self, scenario: Scenario):
        """Test that reward computation is jit-compatible"""

        @eqx.filter_jit
        def _reward(scenario: Scenario):
            reward = scenario.reward(scenario.world.agents[0])
            return reward

        batch_dim = 32
        world = scenario.make_world(batch_dim=batch_dim)
        scenario = scenario.replace(world=world)
        scenario = scenario.reset_world_at(seed=0)

        reward = _reward(scenario)
        assert reward.shape == (batch_dim,)

    def test_reward(self, scenario: Scenario):
        """Test reward computation logic"""
        batch_dim = 32
        world = scenario.make_world(batch_dim=batch_dim)
        scenario = scenario.replace(world=world)
        scenario = scenario.reset_world_at(seed=0)

        # Get initial positions
        agent = scenario.world.agents[0]
        landmark = scenario.world.landmarks[0]

        # Calculate expected reward
        dist2 = jnp.sum(jnp.square(agent.state.pos - landmark.state.pos), axis=-1)
        expected_reward = -dist2

        # Get actual reward
        actual_reward = scenario.reward(agent)

        assert jnp.array_equal(actual_reward, expected_reward)
        assert actual_reward.shape == (batch_dim,)

    def test_observation_jit(self, scenario: Scenario):
        """Test that observation computation is jit-compatible"""
        batch_dim = 32
        world = scenario.make_world(batch_dim=batch_dim)
        scenario = scenario.replace(world=world)
        scenario = scenario.reset_world_at(seed=0)

        @eqx.filter_jit
        def _observation(scenario: Scenario):
            obs = scenario.observation(scenario.world.agents[0])
            return obs

        obs = _observation(scenario)
        assert obs.shape == (
            batch_dim,
            4,
        )  # 2 for velocity + 2 for relative landmark position

    def test_observation(self, scenario: Scenario):
        """Test observation computation logic"""
        batch_dim = 32
        world = scenario.make_world(batch_dim=batch_dim)
        scenario = scenario.replace(world=world)
        scenario = scenario.reset_world_at(seed=0)

        agent = scenario.world.agents[0]
        landmark = scenario.world.landmarks[0]

        # Calculate expected observation
        relative_pos = landmark.state.pos - agent.state.pos
        expected_obs = jnp.concatenate([agent.state.vel, relative_pos], axis=-1)

        # Get actual observation
        actual_obs = scenario.observation(agent)

        assert jnp.array_equal(actual_obs, expected_obs)
        assert actual_obs.shape == (
            batch_dim,
            4,
        )  # 2 for velocity + 2 for relative landmark position
