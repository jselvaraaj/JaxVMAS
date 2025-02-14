from dataclasses import fields, replace
from typing import TypeVar

from equinox import Module as PyTree

TNode = TypeVar("TNode", bound="PyTreeNode")


class PyTreeNode(PyTree):
    def replace(self: TNode, **overrides) -> TNode:
        return replace(self, **overrides)


def dataclass_to_dict_first_layer(obj):
    return {field.name: getattr(obj, field.name) for field in fields(obj)}
