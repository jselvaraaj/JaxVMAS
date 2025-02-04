#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

"""
JAX-compatible environment creation function.
Separates static configuration from dynamic state and ensures jit compatibility.
"""

from flax import struct

from jaxvmas import scenarios
from jaxvmas.simulator.environment import Environment, EnvironmentState
from jaxvmas.simulator.environment.jaxgym import (
    JaxGymnasiumVecWrapper,
    JaxGymnasiumWrapper,
)
from jaxvmas.simulator.scenario import BaseScenario


@struct.dataclass
class EnvConfig:
    """Static environment configuration."""

    num_envs: int = struct.field(pytree_node=False)
    continuous_actions: bool = struct.field(pytree_node=False, default=True)
    dict_spaces: bool = struct.field(pytree_node=False, default=False)
    multidiscrete_actions: bool = struct.field(pytree_node=False, default=False)
    clamp_actions: bool = struct.field(pytree_node=False, default=False)
    grad_enabled: bool = struct.field(pytree_node=False, default=False)
    terminated_truncated: bool = struct.field(pytree_node=False, default=False)
    max_steps: int | None = struct.field(pytree_node=False, default=None)


def make_env(
    scenario: str | BaseScenario,
    num_envs: int,
    continuous_actions: bool = True,
    wrapper: str | None = None,
    max_steps: int | None = None,
    seed: int | None = None,
    dict_spaces: bool = False,
    multidiscrete_actions: bool = False,
    clamp_actions: bool = False,
    grad_enabled: bool = False,
    terminated_truncated: bool = False,
    wrapper_kwargs: dict | None = None,
    **kwargs,
) -> tuple[Environment, EnvironmentState]:
    """Create a JAX-compatible vectorized multi-agent environment.

    Args:
        scenario: Scenario name or BaseScenario instance
        num_envs: Number of vectorized environments
        continuous_actions: Whether to use continuous actions
        wrapper: Optional wrapper type ("gymnasium" or "gymnasium_vec")
        max_steps: Maximum steps per episode
        seed: Random seed
        dict_spaces: Whether to use dictionary spaces
        multidiscrete_actions: Whether to use multidiscrete actions
        clamp_actions: Whether to clamp actions to valid range
        grad_enabled: Whether to enable gradients
        terminated_truncated: Whether to use terminated/truncated flags
        wrapper_kwargs: Additional wrapper arguments
        **kwargs: Additional scenario arguments

    Returns:
        Tuple of (Environment instance, initial EnvironmentState)
    """
    # Load scenario from name if needed
    if isinstance(scenario, str):
        if not scenario.endswith(".py"):
            scenario += ".py"
        scenario = scenarios.load(scenario).Scenario()

    # Create static environment configuration
    config = EnvConfig(
        num_envs=num_envs,
        continuous_actions=continuous_actions,
        dict_spaces=dict_spaces,
        multidiscrete_actions=multidiscrete_actions,
        clamp_actions=clamp_actions,
        grad_enabled=grad_enabled,
        terminated_truncated=terminated_truncated,
        max_steps=max_steps,
    )

    # Create environment instance with static config
    env = Environment(
        scenario=scenario,
        num_envs=config.num_envs,
        continuous_actions=config.continuous_actions,
        max_steps=config.max_steps,
        dict_spaces=config.dict_spaces,
        multidiscrete_actions=config.multidiscrete_actions,
        clamp_actions=config.clamp_actions,
        grad_enabled=config.grad_enabled,
        terminated_truncated=config.terminated_truncated,
        **kwargs,
    )

    # Initialize dynamic state
    state = env.init_state(seed=seed or 0)

    # Apply wrapper if specified
    if wrapper is not None:
        wrapper_kwargs = wrapper_kwargs or {}
        if wrapper.lower() == "gymnasium":
            assert num_envs == 1, "Gymnasium wrapper requires num_envs=1"
            env = JaxGymnasiumWrapper(env, **wrapper_kwargs)
        elif wrapper.lower() == "gymnasium_vec":
            env = JaxGymnasiumVecWrapper(env, **wrapper_kwargs)
        else:
            raise ValueError(f"Unsupported wrapper type: {wrapper}")

    return env, state
