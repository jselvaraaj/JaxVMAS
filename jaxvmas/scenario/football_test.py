import pytest

from jaxvmas.scenario.football import BallAgent, FootballAgent, FootballWorld, Scenario
from jaxvmas.simulator.dynamics.holonomic import Holonomic
from jaxvmas.simulator.dynamics.holonomic_with_rot import HolonomicWithRotation


@pytest.fixture
def scenario() -> Scenario:
    return Scenario.create(batch_dim=2)


def test_make_world(scenario: Scenario):
    # Create world
    world = scenario.make_world()

    # Test basic world properties
    assert isinstance(world, FootballWorld)
    assert world.batch_dim == scenario.batch_dim

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
            assert agent.name in world.traj_points["Red"]
            assert len(world.traj_points["Red"][agent.name]) == scenario.n_traj_points

    if scenario.ai_blue_agents:
        for agent in world.blue_agents:
            assert agent.name in world.traj_points["Blue"]
            assert len(world.traj_points["Blue"][agent.name]) == scenario.n_traj_points


def test_init_agents_default(scenario: Scenario):
    """Test default agent initialization without special configurations"""
    world = scenario.make_world()

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
    world = scenario.make_world()

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
    world = scenario.make_world()

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
    world = scenario.make_world()

    for agent in world.agents:
        if isinstance(agent, FootballAgent):
            assert agent.is_scripted_agent is True


def test_init_agents_invalid_config():
    """Test invalid configuration handling"""
    scenario = Scenario.create(batch_dim=2)
    scenario = scenario.replace(physically_different=True, n_blue_agents=4)

    with pytest.raises(AssertionError, match="Physical differences only for 5 agents"):
        scenario.make_world()
