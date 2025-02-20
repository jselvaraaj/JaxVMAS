import chex
from beartype import beartype
from jaxtyping import jaxtyped

from jaxvmas.equinox_utils import (
    PyTreeNode,
)

# Dimension type variables (add near top of file)
batch_dim = "batch"
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
    batch_dim: int

    @classmethod
    def create(cls, batch_dim: int):
        chex.assert_scalar_positive(batch_dim)
        return cls(batch_dim)

    @jaxtyped(typechecker=beartype)
    def _check_batch_index(self, batch_index: int | float):
        if isinstance(batch_index, float):
            pass
        else:
            chex.assert_scalar_in(batch_index, 0, self.batch_dim - 1)
