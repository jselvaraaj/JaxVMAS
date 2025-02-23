from dataclasses import fields

import chex
from beartype import beartype
from jaxtyping import Array, Int, jaxtyped

from jaxvmas.equinox_utils import (
    PyTreeNode,
)

# Dimension type variables (add near top of file)
batch_axis_dim = "batch_axis_dim"
env_index_dim = "env_index_dim"
pos_dim = "dim_p"
comm_dim = "dim_c"
action_size_dim = "action_size"
angles_dim = "angles"
boxes_dim = "boxes"
spheres_dim = "spheres"
lines_dim = "lines"
dots_dim = "..."


@jaxtyped(typechecker=beartype)
class JaxVectorizedObject(PyTreeNode):
    batch_dim: int | None

    @classmethod
    @chex.assert_max_traces(0)
    def create(cls):
        return cls(*([None] * len(fields(cls))))

    def assert_is_spwaned(self):
        msg = "_spwan first"

        assert self.batch_dim is not None, msg

    @jaxtyped(typechecker=beartype)
    def _check_batch_index(self, batch_index: Int[Array, f"{env_index_dim}"] | None):
        # cannot check dynamic batch index
        pass
        # if isinstance(batch_index, float):
        #     # This mean bathc_index is jnp.nan
        #     # directly checking that with if statment is not allowed with jit compilation.
        #     pass
        # else:
        #     chex.assert_scalar_in(batch_index, 0, self.batch_dim - 1)
