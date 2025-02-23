import jax
import jax.numpy as jnp
import pytest

from jaxvmas.scenario.football import (
    BallAgent,
    FootballAgent,
    FootballWorld,
    Scenario,
    Splines,
)
from jaxvmas.simulator.core.landmark import Landmark
from jaxvmas.simulator.core.shapes import Box, Line, Sphere
from jaxvmas.simulator.dynamics.holonomic import Holonomic
from jaxvmas.simulator.dynamics.holonomic_with_rot import HolonomicWithRotation
from jaxvmas.simulator.utils import Color


@pytest.fixture
def scenario() -> Scenario:
    scenario = Scenario.create()
    scenario = scenario.make_world(batch_dim=2)
    return scenario


def test_make_world(scenario: Scenario):
    # Create world
    scenario = scenario.make_world(batch_dim=2)
    world = scenario.world

    # Test basic world properties
    assert isinstance(world, FootballWorld)
    assert world.batch_dim == 2

    # Test agents initialization
    assert len(world.agents) == scenario.n_red_agents + scenario.n_blue_agents + 1
    assert len(world.red_agents) == scenario.n_red_agents
    assert len(world.blue_agents) == scenario.n_blue_agents

    # Test ball initialization
    assert isinstance(world.ball, BallAgent)
    assert world.ball.name == "Ball"

    # Test walls initialization (4 walls: top, bottom, left, right)
    wall_names = {
        "Left Top Wall",
        "Left Bottom Wall",
        "Right Top Wall",
        "Right Bottom Wall",
    }
    wall_landmarks = {l.name for l in world.landmarks if "Wall" in l.name}
    assert wall_names == wall_landmarks

    # Test goals initialization (6 goal parts + 2 nets)
    goal_names = {
        "Right Goal Back",
        "Left Goal Back",
        "Right Goal Top",
        "Left Goal Top",
        "Right Goal Bottom",
        "Left Goal Bottom",
        "Blue Net",
        "Red Net",
    }
    goal_landmarks = {
        l.name for l in world.landmarks if "Goal" in l.name or "Net" in l.name
    }
    assert goal_names == goal_landmarks

    # Test trajectory points initialization (if AI agents enabled)
    if scenario.ai_red_agents:
        for agent in world.red_agents:
            assert agent.id in world.traj_points["Red"]
            assert len(world.traj_points["Red"][agent.id]) == scenario.n_traj_points

    if scenario.ai_blue_agents:
        for agent in world.blue_agents:
            assert agent.id in world.traj_points["Blue"]
            assert len(world.traj_points["Blue"][agent.id]) == scenario.n_traj_points


def test_init_agents_default(scenario: Scenario):
    """Test default agent initialization without special configurations"""
    scenario = scenario.make_world(batch_dim=2)
    world = scenario.world

    # Test basic properties for blue agents
    for i, agent in enumerate(world.blue_agents):
        assert agent.name == f"agent_blue_{i}"
        assert agent.shape.radius == scenario.agent_size
        assert agent.color == scenario.blue_color
        assert agent.max_speed == scenario.max_speed
        assert agent.alpha == 1
        assert agent.is_scripted_agent is False  # No AI by default

    # Test basic properties for red agents
    for i, agent in enumerate(world.red_agents):
        assert agent.name == f"agent_red_{i}"
        assert agent.shape.radius == scenario.agent_size
        assert agent.color == scenario.red_color
        assert agent.max_speed == scenario.max_speed
        assert agent.alpha == 1
        assert agent.is_scripted_agent is True  # No AI by default


def test_init_agents_physically_different(scenario: Scenario):
    """Test initialization with physically different agents"""
    scenario = scenario.replace(physically_different=True, n_blue_agents=5)
    scenario = scenario.make_world(batch_dim=2)
    world = scenario.world

    # Test attackers (agents 0 and 1)
    for i in range(2):
        agent = world.blue_agents[i]
        assert agent.shape.radius == pytest.approx(scenario.agent_size - 0.005)
        assert agent.max_speed == pytest.approx(scenario.max_speed + 0.05)

    # Test defenders (agents 2 and 3)
    for i in range(2, 4):
        agent = world.blue_agents[i]
        assert agent.shape.radius == scenario.agent_size
        assert agent.max_speed == scenario.max_speed

    # Test goalkeeper (agent 4)
    goalkeeper = world.blue_agents[4]
    assert goalkeeper.shape.radius == pytest.approx(scenario.agent_size + 0.01)
    assert goalkeeper.max_speed == pytest.approx(scenario.max_speed - 0.1)


def test_init_agents_with_shooting(scenario: Scenario):
    """Test agent initialization with shooting enabled"""
    scenario = scenario.replace(enable_shooting=True)
    scenario = scenario.make_world(batch_dim=2)
    world = scenario.world

    for agent in world.red_agents:
        if isinstance(agent, FootballAgent):  # Exclude ball
            assert agent.action_size == 2  # 2 movement + 1 rotation + 1 shooting
            assert isinstance(agent.dynamics, Holonomic)
            assert len(agent.action.u_multiplier) == 2

    for agent in world.blue_agents:
        if isinstance(agent, FootballAgent):  # Exclude ball
            assert agent.action_size == 4  # 2 movement + 1 rotation + 1 shooting
            assert isinstance(agent.dynamics, HolonomicWithRotation)
            assert len(agent.action.u_multiplier) == 4


def test_init_agents_with_ai(scenario: Scenario):
    """Test agent initialization with AI enabled"""
    scenario = scenario.replace(ai_blue_agents=True, ai_red_agents=True)
    scenario = scenario.make_world(batch_dim=2)
    world = scenario.world

    for agent in world.agents:
        if isinstance(agent, FootballAgent):
            assert agent.is_scripted_agent is True


def test_init_agents_invalid_config():
    """Test invalid configuration handling"""
    scenario = Scenario.create()
    scenario = scenario.replace(physically_different=True, n_blue_agents=4)

    with pytest.raises(AssertionError, match="Physical differences only for 5 agents"):
        scenario = scenario.make_world(batch_dim=2)


# Test ball initialization
def test_init_ball(scenario: Scenario):
    world = scenario.world

    # Test ball properties
    ball = world.ball
    assert isinstance(ball, BallAgent)
    assert ball.name == "Ball"
    assert isinstance(ball.shape, Sphere)
    assert ball.shape.radius == scenario.ball_size
    assert ball.max_speed == scenario.ball_max_speed
    assert ball.mass == scenario.ball_mass
    assert ball.alpha == 1
    assert ball.color == Color.BLACK


# Test walls initialization
def test_init_walls(scenario: Scenario):
    # Test number of walls
    world = scenario.world
    wall_landmarks = [l for l in world.landmarks if "Wall" in l.name]
    assert len(wall_landmarks) == 4

    # Test wall properties
    for wall in wall_landmarks:
        assert isinstance(wall, Landmark)
        assert wall.collide is True
        assert wall.movable is False
        assert isinstance(wall.shape, Line)
        assert wall.color == Color.WHITE
        assert wall.shape.length == pytest.approx(
            scenario.pitch_width / 2 - scenario.agent_size - scenario.goal_size / 2
        )


# Test goals initialization
def test_init_goals(scenario: Scenario):
    # Test goal components
    world = scenario.world
    goal_landmarks = [l for l in world.landmarks if "Goal" in l.name or "Net" in l.name]
    assert len(goal_landmarks) == 8

    # Test goal back properties
    goal_backs = [l for l in goal_landmarks if "Back" in l.name]
    for goal in goal_backs:
        assert isinstance(goal.shape, Line)
        assert goal.shape.length == scenario.goal_size

    # Test goal nets
    nets = [l for l in goal_landmarks if "Net" in l.name]
    for net in nets:
        assert isinstance(net.shape, Box)
        assert net.shape.length == scenario.goal_depth
        assert net.shape.width == scenario.goal_size
        assert net.collide is False


# Test trajectory points initialization
def test_init_traj_pts(scenario: Scenario):
    # Test with AI agents enabled
    scenario = scenario.replace(ai_red_agents=True, ai_blue_agents=True)
    scenario = scenario.make_world(batch_dim=2)
    world = scenario.world
    # Verify trajectory points structure
    assert "Red" in world.traj_points
    assert "Blue" in world.traj_points

    # Test red team trajectory points
    for agent in world.red_agents:
        assert agent.id in world.traj_points["Red"]
        points = world.traj_points["Red"][agent.id]
        assert len(points) == scenario.n_traj_points

    # Test blue team trajectory points
    for agent in world.blue_agents:
        assert agent.id in world.traj_points["Blue"]
        points = world.traj_points["Blue"][agent.id]
        assert len(points) == scenario.n_traj_points


# Test Splines functionality
def test_splines():
    splines = Splines()

    # Test hermite interpolation
    p0 = jnp.array([[0.0, 0.0]])  # Shape: [1, 2]
    p1 = jnp.array([[1.0, 1.0]])  # Shape: [1, 2]
    p0dot = jnp.array([[0.0, 0.0]])  # Shape: [1, 2]
    p1dot = jnp.array([[0.0, 0.0]])  # Shape: [1, 2]

    # Test at different points along curve
    for u in [0.0, 0.5, 1.0]:
        _, result = splines.hermite(p0, p1, p0dot, p1dot, u, deriv=0)
        assert result.shape == (1, 2)

    # Test derivatives
    for deriv in [0, 1, 2]:
        _, result = splines.hermite(p0, p1, p0dot, p1dot, 0.5, deriv)
        assert result.shape == (1, 2)


# Test scenario reset and done conditions
def test_scenario_reset_and_done(scenario: Scenario):
    PRNG_key = jax.random.PRNGKey(0)
    batch_dim = 2
    scenario = scenario.env_make_world(batch_dim=batch_dim)
    # Test full reset
    scenario = scenario.reset_world_at(PRNG_key)
    assert not jnp.any(scenario._done)

    # Test partial reset
    env_index = jnp.asarray([0])
    scenario = scenario.reset_world_at(PRNG_key, env_index)
    assert not scenario._done[env_index]

    # Test done condition
    done_state = scenario.done()
    assert done_state.shape == (batch_dim,)


# Test scenario info generation
def test_scenario_info(
    scenario: Scenario,
):
    agent = scenario.world.blue_agents[0]
    info = scenario.info(agent)

    # Verify info contents
    assert "sparse_reward" in info
    assert "ball_goal_pos_rew" in info
    assert "all_agent_ball_pos_rew" in info
    assert "ball_pos" in info
    assert "dist_ball_to_goal" in info

    # Test ball touching detection
    if "touching_ball" in info:
        assert isinstance(info["touching_ball"], bool)


def test_spawn_formation_basic_positioning():
    """
    Test basic formation spawning with fixed positions (no randomization).
    Verifies that agents are placed in correct columns and rows.
    """

    scenario = Scenario.create(
        n_blue_agents=6,
        n_red_agents=6,
        formation_agents_per_column=2,
        randomise_formation_indices=False,
        formation_noise=0.0,
    )
    scenario = scenario.make_world(batch_dim=1)
    agents = [
        FootballAgent.create(
            name=f"agent_{i}",
            shape=Sphere(radius=0.05),
            movable=True,
            rotatable=True,
            collide=True,
            max_speed=1.0,
        )
        for i in range(6)
    ]
    for agent in agents:
        world = scenario.world.add_agent(agent)
        scenario = scenario.replace(world=world)
    agents = scenario.world.agents[-6:]
    # Test blue team formation (left side)
    PRNG_key = jax.random.PRNGKey(0)
    agents = scenario._spawn_formation(
        agents=agents, blue=True, env_index=None, PRNG_key=PRNG_key
    )

    # Check if agents are in correct columns
    positions = jnp.stack([a.state.pos[0] for a in agents])
    unique_x = jnp.unique(positions[:, 0])
    assert len(unique_x) == 3, "Should have 3 columns of agents"

    # Check if all positions are on the left side (blue team)
    assert jnp.all(positions[:, 0] < 0), "Blue team should be on left side"


def test_spawn_formation_with_randomization():
    """
    Test that formation spawning with randomization produces different
    but valid formations.
    """
    scenario = Scenario.create(
        n_blue_agents=6,
        n_red_agents=6,
        formation_agents_per_column=2,
        randomise_formation_indices=True,
        formation_noise=0.0,
    )
    scenario = scenario.make_world(batch_dim=1)
    new_agents = [
        FootballAgent.create(
            name=f"agent_{i}",
            shape=Sphere(radius=0.05),
            movable=True,
            rotatable=True,
            collide=True,
            max_speed=1.0,
        )
        for i in range(6)
    ]
    for agent in new_agents:
        world = scenario.world.add_agent(agent)
        scenario = scenario.replace(world=world)
    world_agents = scenario.world.agents[-6:]
    # Spawn formation twice
    PRNG_key = jax.random.PRNGKey(0)
    PRNG_key, subkey = jax.random.split(PRNG_key)
    agents = scenario._spawn_formation(
        agents=world_agents, blue=True, env_index=None, PRNG_key=subkey
    )
    positions1 = jnp.stack([a.state.pos[0] for a in agents])

    PRNG_key, subkey = jax.random.split(PRNG_key)
    agents = scenario._spawn_formation(
        agents=world_agents, blue=True, env_index=None, PRNG_key=subkey
    )
    positions2 = jnp.stack([a.state.pos[0] for a in agents])

    # Positions should be different due to randomization
    assert not jnp.allclose(positions1, positions2)


def test_spawn_formation_noise():
    """
    Test that formation noise creates variation in positions while
    maintaining overall formation structure.
    """
    noise = 0.5
    scenario = Scenario.create(
        n_blue_agents=6,
        n_red_agents=6,
        formation_agents_per_column=2,
        randomise_formation_indices=False,
        formation_noise=noise,
    )
    scenario = scenario.make_world(batch_dim=1)
    new_agents = [
        FootballAgent.create(
            name=f"agent_{i}",
            shape=Sphere(radius=0.05),
            movable=True,
            rotatable=True,
            collide=True,
            max_speed=1.0,
        )
        for i in range(6)
    ]
    for agent in new_agents:
        world = scenario.world.add_agent(agent)
        scenario = scenario.replace(world=world)
    world_agents = scenario.world.agents[-6:]
    # Spawn formation twice with same configuration
    PRNG_key = jax.random.PRNGKey(0)
    PRNG_key, subkey = jax.random.split(PRNG_key)
    agents = scenario._spawn_formation(
        agents=world_agents, blue=True, env_index=None, PRNG_key=subkey
    )
    positions1 = jnp.stack([a.state.pos[0] for a in agents])

    # second formation
    PRNG_key, subkey = jax.random.split(PRNG_key)
    agents = scenario._spawn_formation(
        agents=world_agents, blue=True, env_index=None, PRNG_key=subkey
    )
    positions2 = jnp.stack([a.state.pos[0] for a in agents])

    # Positions should be different but within noise bounds
    diff = jnp.abs(positions1 - positions2)
    assert jnp.all(diff <= noise), "Position differences exceed noise bound"


def test_spawn_formation_single_env():
    """
    Test formation spawning for a single environment index in a batched world.
    """
    scenario = Scenario.create(
        n_blue_agents=4,
        n_red_agents=4,
        formation_agents_per_column=2,
        randomise_formation_indices=False,
        formation_noise=0.0,
    )
    scenario = scenario.make_world(batch_dim=3)
    agents = [
        FootballAgent.create(
            name=f"agent_{i}",
            shape=Sphere(radius=0.05),
            movable=True,
            rotatable=True,
            collide=True,
            max_speed=1.0,
        )
        for i in range(4)
    ]
    for agent in agents:
        world = scenario.world.add_agent(agent)
        scenario = scenario.replace(world=world)
    agents = scenario.world.agents[-4:]
    env_index = jnp.asarray([1])
    PRNG_key = jax.random.PRNGKey(0)
    agents = scenario._spawn_formation(
        agents=agents, blue=False, env_index=env_index, PRNG_key=PRNG_key
    )

    # Check if positions are set only for specified environment
    for agent in agents:
        assert not jnp.any(
            jnp.isnan(agent.state.pos[env_index])
        ), "Position not set for env_index"


def test_spawn_formation_pitch_boundaries():
    """
    Test that spawned formations respect pitch boundaries.
    """
    scenario = Scenario.create(
        n_blue_agents=6,
        n_red_agents=6,
        formation_agents_per_column=2,
        randomise_formation_indices=False,
        formation_noise=0.0,
    )
    scenario = scenario.make_world(batch_dim=1)
    agents = [
        FootballAgent.create(
            name=f"agent_{i}",
            shape=Sphere(radius=0.05),
            movable=True,
            rotatable=True,
            collide=True,
            max_speed=1.0,
        )
        for i in range(6)
    ]
    for agent in agents:
        world = scenario.world.add_agent(agent)
        scenario = scenario.replace(world=world)
    agents = scenario.world.agents[-6:]
    PRNG_key = jax.random.PRNGKey(0)
    agents = scenario._spawn_formation(
        agents=agents, blue=True, env_index=None, PRNG_key=PRNG_key
    )
    positions = jnp.stack([a.state.pos[0] for a in agents])

    # Check if all positions are within pitch boundaries
    assert jnp.all(
        jnp.abs(positions[:, 1]) <= scenario.pitch_width / 2
    ), "Agents outside pitch width"
    assert jnp.all(
        positions[:, 0] >= -(scenario.pitch_length / 2 + scenario.goal_depth)
    ), "Agents beyond goal line"
