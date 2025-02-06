from dataclasses import replace
from typing import TypeVar

from equinox import Module as PyTree

TNode = TypeVar("TNode", bound="PyTreeNode")


class PyTreeNode(PyTree):
    def replace(self: TNode, **overrides) -> TNode:
        return replace(self, **overrides)
