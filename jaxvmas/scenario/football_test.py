import pytest

from jaxvmas.scenario.football import BallAgent, FootballWorld, Scenario


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
