from dataclasses import fields, replace
from typing import TypeVar

import equinox as eqx
import jax
from equinox import Module as PyTree

TNode = TypeVar("TNode", bound="PyTreeNode")


class PyTreeNode(PyTree):
    def replace(self: TNode, **overrides) -> TNode:
        return replace(self, **overrides)


def dataclass_to_dict_first_layer(obj):
    return {field.name: getattr(obj, field.name) for field in fields(obj)}


def equinox_filter_cond_return_pytree_node(
    pred, on_true, on_false, pytree_node: PyTreeNode, *operands
):

    pytree_node_dynamic, pytree_node_static = eqx.partition(pytree_node, eqx.is_array)

    operands_dynamic, operands_static = eqx.partition(operands, eqx.is_array)

    def on_true_filter(_pytree_node_dynamic, *_operands_dynamic):
        _pytree_node = eqx.combine(_pytree_node_dynamic, pytree_node_static)
        _operands = eqx.combine(operands_static, _operands_dynamic)
        _pytree_node = on_true(_pytree_node, *_operands)
        _pytree_node_dynamic, _ = eqx.partition(_pytree_node, eqx.is_array)
        return _pytree_node_dynamic

    def on_false_filter(_pytree_node_dynamic, *_operands_dynamic):
        _pytree_node = eqx.combine(_pytree_node_dynamic, pytree_node_static)
        _operands = eqx.combine(operands_static, _operands_dynamic)
        _pytree_node = on_false(_pytree_node, *_operands)
        _pytree_node_dynamic, _ = eqx.partition(_pytree_node, eqx.is_array)
        return _pytree_node_dynamic

    pytree_node_dynamic = jax.lax.cond(
        pred, on_true_filter, on_false_filter, pytree_node_dynamic, *operands_dynamic
    )
    return eqx.combine(pytree_node_dynamic, pytree_node_static)


def equinox_filter_cond_return_dynamic(
    pred, on_true, on_false, pytree_node: PyTreeNode, *operands
):

    pytree_node_dynamic, pytree_node_static = eqx.partition(pytree_node, eqx.is_array)

    operands_dynamic, operands_static = eqx.partition(operands, eqx.is_array)

    def on_true_filter(_pytree_node_dynamic, *_operands_dynamic):
        _pytree_node = eqx.combine(_pytree_node_dynamic, pytree_node_static)
        _operands = eqx.combine(operands_static, _operands_dynamic)
        return on_true(_pytree_node, *_operands)

    def on_false_filter(_pytree_node_dynamic, *_operands_dynamic):
        _pytree_node = eqx.combine(_pytree_node_dynamic, pytree_node_static)
        _operands = eqx.combine(operands_static, _operands_dynamic)
        return on_false(_pytree_node, *_operands)

    pytree_node_dynamic = jax.lax.cond(
        pred, on_true_filter, on_false_filter, pytree_node_dynamic, *operands_dynamic
    )
    return pytree_node_dynamic
