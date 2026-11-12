"""Small dependency-free AST validators for expression-only Python probes."""

from __future__ import annotations

import ast
from typing import Any


def _dotted(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


def _single_expression(tree: ast.Module) -> ast.expr | None:
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
        return None
    return tree.body[0].value


def _validate_pickle_reduce_system(tree: ast.Module, spec: dict[str, Any]) -> str | None:
    expression = _single_expression(tree)
    if not isinstance(expression, ast.Call) or _dotted(expression.func) != "pickle.dumps":
        return "expected_single_pickle_dumps_expression"
    if len(expression.args) != 1 or expression.keywords:
        return "pickle_dumps_arity"
    instance_call = expression.args[0]
    if not isinstance(instance_call, ast.Call) or instance_call.args or instance_call.keywords:
        return "expected_dynamic_class_instance"
    type_call = instance_call.func
    if not isinstance(type_call, ast.Call) or _dotted(type_call.func) != "type" or len(type_call.args) != 3:
        return "expected_type_constructor"
    bases, namespace = type_call.args[1], type_call.args[2]
    # ``type(name, (), namespace)`` implicitly derives from ``object`` and is
    # equivalent to the explicit ``(object,)`` form for this dynamic class.
    # Keep the allowlist narrow: arbitrary or multiple bases remain rejected.
    valid_bases = (
        isinstance(bases, ast.Tuple)
        and (
            len(bases.elts) == 0
            or (
                len(bases.elts) == 1
                and isinstance(bases.elts[0], ast.Name)
                and bases.elts[0].id == "object"
            )
        )
    )
    if not valid_bases:
        return "expected_object_base"
    if not isinstance(namespace, ast.Dict):
        return "expected_class_namespace_dict"
    reduce_lambda = None
    for key, value in zip(namespace.keys, namespace.values):
        if isinstance(key, ast.Constant) and key.value == "__reduce__":
            reduce_lambda = value
            break
    if not isinstance(reduce_lambda, ast.Lambda):
        return "missing_reduce_lambda"
    lambda_args = reduce_lambda.args
    if not (
        len(lambda_args.posonlyargs) + len(lambda_args.args) == 1
        and not lambda_args.kwonlyargs
        and lambda_args.vararg is None
        and lambda_args.kwarg is None
        and not lambda_args.defaults
        and not lambda_args.kw_defaults
    ):
        return "reduce_lambda_must_accept_exactly_self"
    body = reduce_lambda.body
    if not isinstance(body, ast.Tuple) or len(body.elts) != 2:
        return "reduce_must_return_pair"
    callable_node, arguments_node = body.elts
    if not (
        isinstance(callable_node, ast.Attribute)
        and callable_node.attr == "system"
        and isinstance(callable_node.value, ast.Call)
        and _dotted(callable_node.value.func) == "__import__"
        and len(callable_node.value.args) == 1
        and isinstance(callable_node.value.args[0], ast.Constant)
        and callable_node.value.args[0].value == spec.get("module", "os")
    ):
        return "reduce_callable_not_imported_os_system"
    expected_argument = spec.get("argument", "id")
    if not (
        isinstance(arguments_node, ast.Tuple)
        and len(arguments_node.elts) == 1
        and isinstance(arguments_node.elts[0], ast.Constant)
        and arguments_node.elts[0].value == expected_argument
    ):
        return "reduce_argument_mismatch"
    return None


def _bytes_repeat(node: ast.AST, byte_value: bytes, count: int) -> bool:
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Mult):
        return False
    pairs = ((node.left, node.right), (node.right, node.left))
    return any(
        isinstance(left, ast.Constant)
        and left.value == byte_value
        and isinstance(right, ast.Constant)
        and right.value == count
        for left, right in pairs
    )


def _validate_bytes_payload(tree: ast.Module, spec: dict[str, Any]) -> str | None:
    expression = _single_expression(tree)
    if not isinstance(expression, ast.BinOp) or not isinstance(expression.op, ast.Add):
        return "expected_padding_plus_pack_expression"
    padding = spec.get("padding", "A").encode("ascii")
    count = int(spec.get("count", 72))
    if not _bytes_repeat(expression.left, padding, count):
        return "padding_repeat_mismatch"
    pack = expression.right
    if not isinstance(pack, ast.Call) or _dotted(pack.func) != "struct.pack":
        return "expected_struct_pack"
    if len(pack.args) != 2 or pack.keywords:
        return "struct_pack_arity"
    expected_format = spec.get("format", "<Q")
    expected_value = int(spec.get("value"))
    if not isinstance(pack.args[0], ast.Constant) or pack.args[0].value != expected_format:
        return "struct_pack_format_mismatch"
    if not isinstance(pack.args[1], ast.Constant) or pack.args[1].value != expected_value:
        return "struct_pack_value_mismatch"
    return None


def _validate_numpy_savetxt(tree: ast.Module, spec: dict[str, Any]) -> str | None:
    function_name = spec.get("function", "save_predictions")
    functions = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
    ]
    if len(functions) != 1:
        return f"expected_function={function_name}"
    function = functions[0]
    parameter_names = [argument.arg for argument in function.args.args]
    expected_parameters = spec.get("parameters", ["arr", "path"])
    if parameter_names != expected_parameters or function.args.vararg or function.args.kwarg:
        return f"parameter_mismatch={parameter_names}"
    top_level_calls = []
    for statement in function.body:
        value = statement.value if isinstance(statement, (ast.Expr, ast.Return)) else None
        if isinstance(value, ast.Call) and _dotted(value.func) == spec.get("call", "np.savetxt"):
            top_level_calls.append(value)
    if len(top_level_calls) != 1:
        return "expected_one_top_level_np_savetxt_call"
    call = top_level_calls[0]
    if len(call.args) < 2:
        return "np_savetxt_missing_arguments"
    if not isinstance(call.args[0], ast.Name) or call.args[0].id != "path":
        return "np_savetxt_first_argument_must_be_path"
    if not isinstance(call.args[1], ast.Name) or call.args[1].id != "arr":
        return "np_savetxt_second_argument_must_be_arr"
    fmt_value = call.args[2] if len(call.args) >= 3 else None
    for keyword in call.keywords:
        if keyword.arg == "fmt":
            fmt_value = keyword.value
    if not isinstance(fmt_value, ast.Constant) or fmt_value.value != spec.get("fmt", "%d"):
        return "np_savetxt_format_mismatch"
    return None


def _validate_single_call_expression(tree: ast.Module, spec: dict[str, Any]) -> str | None:
    expression = _single_expression(tree)
    expected_call = spec.get("call")
    if not isinstance(expression, ast.Call) or _dotted(expression.func) != expected_call:
        return f"expected_single_call_expression={expected_call}"
    expected_args = spec.get("args", [])
    if len(expression.args) != len(expected_args) or expression.keywords:
        return "single_call_arguments_mismatch"
    for node, expected in zip(expression.args, expected_args):
        if not isinstance(node, ast.Constant) or node.value != expected:
            return "single_call_arguments_mismatch"
    return None


def validate_python_semantics_ast(
    tree: ast.Module,
    spec: dict[str, Any],
) -> tuple[bool, str]:
    """Validate one allowlisted expression shape without executing it."""
    kind = spec.get("kind")
    if kind == "pickle_reduce_system":
        error = _validate_pickle_reduce_system(tree, spec)
    elif kind == "bytes_padding_struct_pack":
        error = _validate_bytes_payload(tree, spec)
    elif kind == "numpy_savetxt":
        error = _validate_numpy_savetxt(tree, spec)
    elif kind == "single_call_expression":
        error = _validate_single_call_expression(tree, spec)
    else:
        return False, f"unsupported_python_semantics_kind={kind}"
    return error is None, error or "python_semantics_passed"
