# JaxVMAS 🚀

[![PyPI version](https://img.shields.io/pypi/v/jaxvmas)](https://pypi.org/project/jaxvmas/)
[![License](https://img.shields.io/badge/license-GPLv3.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![JAX](https://img.shields.io/badge/Powered%20by-JAX%20%F0%9F%9A%80-yellow)](https://github.com/google/jax)
[![Coverage](./badges/coverage.svg)](https://github.com/jselvaraaj/JaxVMAS/actions)
[![Tests MacOS](https://github.com/jselvaraaj/JaxVMAS/actions/workflows/pytest-macos.yml/badge.svg)](https://github.com/jselvaraaj/JaxVMAS/actions/workflows/pytest-macos.yml)
[![Tests Ubuntu](https://github.com/jselvaraaj/JaxVMAS/actions/workflows/pytest-ubuntu.yml/badge.svg)](https://github.com/jselvaraaj/JaxVMAS/actions/workflows/pytest-ubuntu.yml)

JAX implementation of [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator) - A vectorized differentiable multi-agent simulator for MARL. This project is a direct port of the original PyTorch-based [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator) developed by the [Prorok Lab](https://www.proroklab.org/), rewritten in JAX to provide JIT compilation support and enhanced performance on GPUs/TPUs.

## Features ✨

- **JAX-native** - GPU/TPU support with `jax.jit` compatibility
- **Vectorized** - Batch thousands of environments simultaneously
- **Differentiable** - End-to-end gradients through physics simulations
- **Interactive rendering** - Visualize agent behaviors and interactions

## Installation ⚙️

```bash
pip install jaxvmas
```

For GPU support, first install [JAX with CUDA support](https://github.com/google/jax#installation).

## Quick Start 🏃

```python
import jax
import jax.numpy as jnp
from jaxvmas.make_env import make_env

# Create a random key for initialization
key = jax.random.PRNGKey(0)

# Create vectorized environments
num_envs = 32
env = make_env(
    scenario="football",  # or "simple" from MPE scenarios
    num_envs=num_envs,
    PRNG_key=key,
    continuous_actions=True
)

# Reset environment
env, obs = env.reset(PRNG_key=key)

# Random actions
actions = [jnp.zeros((num_envs, agent_action_space.shape[0])) 
           for agent_action_space in env.action_space]

# Step environment
env, obs, rewards, dones, infos = env.step(actions, key)
```

## Performance 🚀

**Note**: The following benchmarks are preliminary and not comprehensive. Your results may vary depending on hardware and specific scenarios.

### Preliminary Benchmarking on Google Colab GPU

| Configuration | VMAS (original) | JaxVMAS |
|---------------|----------------|---------|
| 1 environment, 200 steps | ~24.63 seconds | ~0.22 seconds simulation time<br>(+~222.87 seconds compile time)<br>Total: ~233 seconds |
| 32 environments, 100,000 steps | 25+ minutes (test stopped) | ~75.20 seconds simulation time<br>(+~302.10 seconds compile time)<br>Total: ~387.05 seconds |

JaxVMAS demonstrates significant speedup in actual simulation time after the initial compilation overhead.

## Supported Scenarios 🎮

- Football
- MPE (Multi-Particle Environment):
  - Simple

## Citing JaxVMAS 📖

If you use JaxVMAS in your research, please cite both JaxVMAS and the original VMAS:

```bibtex
@misc{jaxvmas2024,
  title={JaxVMAS: Multi-Agent Simulation of VMAS scenarios in JAX},
  author={Joseph Selvaraaj},
  year={2024},
  url = {https://github.com/jselvaraaj/JaxVMAS},
}
```

Please also cite the original VMAS paper:

```bibtex
@article{bettini2022vmas,
  title = {VMAS: A Vectorized Multi-Agent Simulator for Collective Robot Learning},
  author = {Bettini, Matteo and Kortvelesy, Ryan and Blumenkamp, Jan and Prorok, Amanda},
  year = {2022},
  journal={The 16th International Symposium on Distributed Autonomous Robotic Systems},
  publisher={Springer}
}
```

---

## Related Projects
- [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator) - Original PyTorch implementation by the Prorok Lab, on which JaxVMAS is based
