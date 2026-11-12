"""Dependency-free static semantics for small OOXML-authoring functions.

This module never imports openpyxl, python-docx, or python-pptx and never
compiles or executes submitted code.  It interprets a deliberately small,
auditable subset of Python AST into symbolic workbook/document/presentation
objects, then compares the saved symbolic artifact with a declarative spec.
"""

from __future__ import annotations

import ast
import copy
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class _Symbol:
    name: str


@dataclass(frozen=True)
class _Module:
    name: str


@dataclass(frozen=True)
class _Qualified:
    name: str


@dataclass(frozen=True)
class _Builtin:
    name: str


@dataclass(frozen=True)
class _Method:
    owner: Any
    name: str


@dataclass
class _Font:
    bold: Any = None
    italic: Any = None
    underline: Any = None


@dataclass
class _XCell:
    value: Any = None
    font: _Font = field(default_factory=_Font)


@dataclass
class _Sheet:
    title: str = "Sheet"
    cells: dict[str, _XCell] = field(default_factory=dict)
    charts: list[Any] = field(default_factory=list)


@dataclass
class _Workbook:
    sheets: list[_Sheet] = field(default_factory=lambda: [_Sheet()])


@dataclass
class _Reference:
    sheet: _Sheet
    min_col: int | None = None
    min_row: int | None = None
    max_col: int | None = None
    max_row: int | None = None


@dataclass
class _Chart:
    chart_type: str
    data: _Reference | None = None
    categories: _Reference | None = None
    titles_from_data: bool | None = None
    anchor: str | None = None


@dataclass
class _Run:
    text: str = ""
    direct_formatting: bool = False


@dataclass
class _Paragraph:
    text: str = ""
    style: str = "Normal"
    direct_formatting: bool = False
    runs: list[_Run] = field(default_factory=list)


@dataclass(frozen=True)
class _ParagraphFormat:
    paragraph: _Paragraph


@dataclass(frozen=True)
class _RunFont:
    run: _Run


@dataclass
class _DCell:
    text: str = ""


@dataclass
class _Table:
    cells: list[list[_DCell]]


@dataclass
class _Document:
    paragraphs: list[_Paragraph] = field(default_factory=list)
    tables: list[_Table] = field(default_factory=list)


@dataclass(frozen=True)
class _Layout:
    index: int


@dataclass
class _TitleShape:
    text: str = ""


@dataclass
class _Slide:
    layout_index: int
    title_shape: _TitleShape | None
    extra_shapes: int = 0


@dataclass
class _Presentation:
    slides: list[_Slide] = field(default_factory=list)


@dataclass(frozen=True)
class _SlidesProxy:
    presentation: _Presentation


@dataclass(frozen=True)
class _LayoutsProxy:
    presentation: _Presentation


@dataclass(frozen=True)
class _ShapesProxy:
    slide: _Slide


@dataclass(frozen=True)
class _RowsProxy:
    table: _Table


@dataclass(frozen=True)
class _RowProxy:
    table: _Table
    row: int


@dataclass(frozen=True)
class _CellsProxy:
    table: _Table
    row: int


class _StaticInterpreter:
    """Conservative interpreter for literal OOXML construction code."""

    MAX_LOOP_ITEMS = 100

    def __init__(self, tree: ast.Module, function_name: str):
        self.tree = tree
        self.function_name = function_name
        self.env: dict[str, Any] = {}
        self.errors: list[str] = []
        self.artifacts: list[tuple[str, dict[str, Any]]] = []
        self.path_parameter = "path"

    def run(self) -> None:
        function: ast.FunctionDef | ast.AsyncFunctionDef | None = None
        for node in self.tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self._statement(node)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                self._statement(node)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == self.function_name:
                function = node
        if function is None:
            self.errors.append(f"missing_function={self.function_name}")
            return
        if function.args.args:
            self.path_parameter = function.args.args[0].arg
        for argument in function.args.args:
            self.env[argument.arg] = _Symbol(argument.arg)
        self._block(function.body)

    def _block(self, statements: list[ast.stmt]) -> None:
        for statement in statements:
            self._statement(statement)
            if self.errors:
                return

    def _statement(self, node: ast.stmt) -> None:
        if isinstance(node, ast.Import):
            for alias in node.names:
                self.env[alias.asname or alias.name.split(".")[0]] = _Module(alias.name)
            return
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                self.env[alias.asname or alias.name] = _Qualified(f"{module}.{alias.name}".strip("."))
            return
        if isinstance(node, ast.Assign):
            value = self._expression(node.value)
            for target in node.targets:
                self._assign(target, value)
            return
        if isinstance(node, ast.AnnAssign):
            if node.value is not None:
                self._assign(node.target, self._expression(node.value))
            return
        if isinstance(node, ast.Expr):
            self._expression(node.value)
            return
        if isinstance(node, ast.For):
            iterable = self._expression(node.iter)
            values = self._bounded_values(iterable)
            if values is None:
                self.errors.append(f"unsupported_for_iterable={type(iterable).__name__}")
                return
            for value in values:
                self._assign(node.target, value)
                self._block(node.body)
                if self.errors:
                    return
            if node.orelse:
                self._block(node.orelse)
            return
        if isinstance(node, ast.If):
            condition = self._expression(node.test)
            if not isinstance(condition, bool):
                self.errors.append("nonliteral_if_condition")
                return
            self._block(node.body if condition else node.orelse)
            return
        if isinstance(node, (ast.Return, ast.Pass)):
            return
        self.errors.append(f"unsupported_statement={type(node).__name__}")

    def _expression(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in self.env:
                return self.env[node.id]
            if node.id in {"range", "enumerate", "zip", "len"}:
                return _Builtin(node.id)
            self.errors.append(f"unknown_name={node.id}")
            return None
        if isinstance(node, ast.List):
            return [self._expression(item) for item in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(self._expression(item) for item in node.elts)
        if isinstance(node, ast.Dict):
            return {self._expression(k): self._expression(v) for k, v in zip(node.keys, node.values)}
        if isinstance(node, ast.JoinedStr):
            parts: list[str] = []
            for value in node.values:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    parts.append(value.value)
                    continue
                if (isinstance(value, ast.FormattedValue)
                        and value.conversion == -1 and value.format_spec is None):
                    rendered = self._expression(value.value)
                    if isinstance(rendered, (str, int)) and not isinstance(rendered, bool):
                        parts.append(str(rendered))
                        continue
                self.errors.append("unsupported_fstring_component")
                return None
            return "".join(parts)
        if isinstance(node, ast.GeneratorExp):
            return self._generator_values(node)
        if isinstance(node, ast.Attribute):
            return self._get_attribute(self._expression(node.value), node.attr)
        if isinstance(node, ast.Subscript):
            return self._get_item(self._expression(node.value), self._expression(node.slice))
        if isinstance(node, ast.Call):
            callable_value = self._expression(node.func)
            arguments = [self._expression(arg) for arg in node.args]
            keywords = {kw.arg: self._expression(kw.value) for kw in node.keywords if kw.arg is not None}
            return self._call(callable_value, arguments, keywords)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = self._expression(node.operand)
            if not isinstance(value, (int, float)):
                self.errors.append("unsupported_unary_operand")
                return None
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left, right = self._expression(node.left), self._expression(node.right)
            try:
                return left + right
            except Exception:
                self.errors.append("unsupported_addition")
                return None
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1:
            left, right = self._expression(node.left), self._expression(node.comparators[0])
            if isinstance(node.ops[0], ast.Eq):
                return left == right
            if isinstance(node.ops[0], ast.NotEq):
                return left != right
            if isinstance(node.ops[0], ast.Is):
                return left is right
            if isinstance(node.ops[0], ast.IsNot):
                return left is not right
        self.errors.append(f"unsupported_expression={type(node).__name__}")
        return None

    def _bounded_values(self, value: Any) -> list[Any] | None:
        """Materialize only the small, allowlisted iterables used by probes."""
        if isinstance(value, range):
            if len(value) > self.MAX_LOOP_ITEMS:
                self.errors.append("loop_too_large")
                return None
            return list(value)
        if isinstance(value, (list, tuple)):
            if len(value) > self.MAX_LOOP_ITEMS:
                self.errors.append("loop_too_large")
                return None
            return list(value)
        if isinstance(value, _RowsProxy):
            rows = len(value.table.cells)
            if rows > self.MAX_LOOP_ITEMS:
                self.errors.append("loop_too_large")
                return None
            return [_RowProxy(value.table, row) for row in range(rows)]
        if isinstance(value, _CellsProxy):
            try:
                cells = value.table.cells[value.row]
            except IndexError:
                self.errors.append(f"table_row_index={value.row}")
                return None
            if len(cells) > self.MAX_LOOP_ITEMS:
                self.errors.append("loop_too_large")
                return None
            return list(cells)
        return None

    def _generator_values(self, node: ast.GeneratorExp) -> list[Any] | None:
        """Evaluate a bounded comprehension without executing submitted code.

        Conditions, async iteration, and arbitrary iterators intentionally stay
        unsupported.  The comprehension gets an isolated name environment,
        matching Python 3's non-leaking generator scope.
        """
        if not node.generators or any(
                generator.is_async or generator.ifs
                for generator in node.generators):
            self.errors.append("unsupported_generator_comprehension")
            return None
        original_env = self.env
        self.env = self.env.copy()
        results: list[Any] = []

        def visit(index: int) -> None:
            if self.errors:
                return
            if index == len(node.generators):
                results.append(self._expression(node.elt))
                if len(results) > self.MAX_LOOP_ITEMS:
                    self.errors.append("generator_too_large")
                return
            generator = node.generators[index]
            iterable = self._expression(generator.iter)
            values = self._bounded_values(iterable)
            if values is None:
                if not self.errors:
                    self.errors.append(
                        f"unsupported_generator_iterable={type(iterable).__name__}"
                    )
                return
            for value in values:
                self._assign(generator.target, value)
                visit(index + 1)
                if self.errors:
                    return

        try:
            visit(0)
            return None if self.errors else results
        finally:
            self.env = original_env

    def _assign(self, target: ast.AST, value: Any) -> None:
        if isinstance(target, ast.Name):
            self.env[target.id] = value
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            if not isinstance(value, (tuple, list)) or len(target.elts) != len(value):
                self.errors.append("unpack_mismatch")
                return
            for child, child_value in zip(target.elts, value):
                self._assign(child, child_value)
            return
        if isinstance(target, ast.Attribute):
            self._set_attribute(self._expression(target.value), target.attr, value)
            return
        if isinstance(target, ast.Subscript):
            self._set_item(self._expression(target.value), self._expression(target.slice), value)
            return
        self.errors.append(f"unsupported_assignment={type(target).__name__}")

    def _get_attribute(self, owner: Any, name: str) -> Any:
        if isinstance(owner, (_Module, _Qualified)):
            return _Qualified(f"{owner.name}.{name}")
        if isinstance(owner, _Workbook):
            if name == "active":
                return owner.sheets[0]
            if name in {"create_sheet", "save", "remove"}:
                return _Method(owner, name)
            if name == "worksheets":
                return owner.sheets
        if isinstance(owner, _Sheet):
            if name in {"append", "add_chart", "cell"}:
                return _Method(owner, name)
            if name == "title":
                return owner.title
        if isinstance(owner, _XCell):
            if name == "value":
                return owner.value
            if name == "font":
                return owner.font
        if isinstance(owner, _Document):
            if name in {"add_paragraph", "add_heading", "add_table", "save"}:
                return _Method(owner, name)
            if name == "paragraphs":
                return owner.paragraphs
            if name == "tables":
                return owner.tables
        if isinstance(owner, _Paragraph):
            if name == "add_run":
                return _Method(owner, name)
            if name == "paragraph_format":
                return _ParagraphFormat(owner)
            if name == "runs":
                return owner.runs
            if name in {"text", "style"}:
                return getattr(owner, name)
        if isinstance(owner, _Run):
            if name == "font":
                return _RunFont(owner)
            if name in {"text", "bold", "italic", "underline"}:
                return owner.text if name == "text" else owner.direct_formatting
        if isinstance(owner, _Table):
            if name == "cell":
                return _Method(owner, name)
            if name == "rows":
                return _RowsProxy(owner)
        if isinstance(owner, _RowProxy) and name == "cells":
            return _CellsProxy(owner.table, owner.row)
        if isinstance(owner, _DCell) and name == "text":
            return owner.text
        if isinstance(owner, _Presentation):
            if name == "slides":
                return _SlidesProxy(owner)
            if name == "slide_layouts":
                return _LayoutsProxy(owner)
            if name == "save":
                return _Method(owner, name)
        if isinstance(owner, _SlidesProxy) and name == "add_slide":
            return _Method(owner, name)
        if isinstance(owner, _Slide) and name == "shapes":
            return _ShapesProxy(owner)
        if isinstance(owner, _ShapesProxy):
            if name == "title":
                return owner.slide.title_shape
            if name.startswith("add_"):
                return _Method(owner, name)
        if isinstance(owner, _TitleShape):
            if name == "text":
                return owner.text
            if name == "text_frame":
                return owner
        if isinstance(owner, _Chart) and name in {"add_data", "set_categories"}:
            return _Method(owner, name)
        if isinstance(owner, list) and name == "append":
            return _Method(owner, name)
        self.errors.append(f"unsupported_attribute={type(owner).__name__}.{name}")
        return None

    def _set_attribute(self, owner: Any, name: str, value: Any) -> None:
        if isinstance(owner, _Sheet) and name == "title" and isinstance(value, str):
            owner.title = value
            return
        if isinstance(owner, _XCell):
            if name == "value":
                owner.value = value
                return
            if name == "font" and isinstance(value, _Font):
                owner.font = value
                return
        if isinstance(owner, _Font) and name in {"bold", "italic", "underline"}:
            setattr(owner, name, value)
            return
        if isinstance(owner, _Paragraph):
            if name == "text" and isinstance(value, str):
                owner.text = value
                return
            if name == "style" and isinstance(value, str):
                owner.style = value
                return
            if name == "alignment":
                owner.direct_formatting = True
                return
        if isinstance(owner, _ParagraphFormat):
            owner.paragraph.direct_formatting = True
            return
        if isinstance(owner, _Run):
            if name == "text" and isinstance(value, str):
                owner.text = value
                return
            if name in {"bold", "italic", "underline"}:
                owner.direct_formatting = True
                return
        if isinstance(owner, _RunFont):
            owner.run.direct_formatting = True
            return
        if isinstance(owner, _DCell) and name == "text" and isinstance(value, str):
            owner.text = value
            return
        if isinstance(owner, _TitleShape) and name == "text" and isinstance(value, str):
            owner.text = value
            return
        self.errors.append(f"unsupported_attribute_assignment={type(owner).__name__}.{name}")

    def _get_item(self, owner: Any, key: Any) -> Any:
        if isinstance(owner, _Workbook) and isinstance(key, str):
            for sheet in owner.sheets:
                if sheet.title == key:
                    return sheet
            self.errors.append(f"unknown_sheet={key}")
            return None
        if isinstance(owner, _Sheet) and isinstance(key, str):
            coordinate = key.upper()
            return owner.cells.setdefault(coordinate, _XCell())
        if isinstance(owner, _LayoutsProxy) and isinstance(key, int):
            return _Layout(key)
        if isinstance(owner, _SlidesProxy) and isinstance(key, int):
            try:
                return owner.presentation.slides[key]
            except IndexError:
                self.errors.append(f"slide_index={key}")
                return None
        if isinstance(owner, _RowsProxy) and isinstance(key, int):
            return _RowProxy(owner.table, key)
        if isinstance(owner, _CellsProxy) and isinstance(key, int):
            try:
                return owner.table.cells[owner.row][key]
            except IndexError:
                self.errors.append(f"table_cell_index={owner.row},{key}")
                return None
        if isinstance(owner, (list, tuple)) and isinstance(key, int):
            try:
                return owner[key]
            except IndexError:
                self.errors.append(f"sequence_index={key}")
                return None
        self.errors.append(f"unsupported_subscript={type(owner).__name__}")
        return None

    def _set_item(self, owner: Any, key: Any, value: Any) -> None:
        if isinstance(owner, _Sheet) and isinstance(key, str):
            owner.cells.setdefault(key.upper(), _XCell()).value = value
            return
        if isinstance(owner, list) and isinstance(key, int):
            try:
                owner[key] = value
                return
            except IndexError:
                pass
        self.errors.append(f"unsupported_subscript_assignment={type(owner).__name__}")

    def _call(self, callable_value: Any, args: list[Any], kwargs: dict[str, Any]) -> Any:
        if isinstance(callable_value, _Qualified):
            leaf = callable_value.name.rsplit(".", 1)[-1]
            if leaf == "Workbook":
                return _Workbook()
            if leaf == "Document":
                return _Document()
            if leaf == "Presentation":
                return _Presentation()
            if leaf == "Font":
                return _Font(
                    bold=kwargs.get("bold"), italic=kwargs.get("italic"), underline=kwargs.get("underline")
                )
            if leaf in {"BarChart", "LineChart", "PieChart"}:
                return _Chart(leaf)
            if leaf == "Reference":
                if not args or not isinstance(args[0], _Sheet):
                    self.errors.append("reference_missing_sheet")
                    return None
                return _Reference(
                    args[0], kwargs.get("min_col"), kwargs.get("min_row"),
                    kwargs.get("max_col", kwargs.get("min_col")),
                    kwargs.get("max_row", kwargs.get("min_row")),
                )
        if isinstance(callable_value, _Builtin):
            if callable_value.name == "range":
                if not all(isinstance(value, int) for value in args):
                    self.errors.append("noninteger_range")
                    return []
                return range(*args)
            if callable_value.name == "enumerate":
                values = self._bounded_values(args[0]) if args else None
                start = kwargs.get("start", args[1] if len(args) > 1 else 0)
                if (values is None or not isinstance(start, int) or isinstance(start, bool)
                        or (len(args) > 1 and "start" in kwargs)
                        or len(args) > 2 or set(kwargs) - {"start"}):
                    self.errors.append("unsupported_enumerate")
                    return []
                return list(enumerate(values, start))
            if callable_value.name == "zip":
                values = [self._bounded_values(value) for value in args]
                if (not args or kwargs or any(value is None for value in values)):
                    self.errors.append("unsupported_zip")
                    return []
                zipped = list(zip(*values))
                if len(zipped) > self.MAX_LOOP_ITEMS:
                    self.errors.append("loop_too_large")
                    return []
                return zipped
            if callable_value.name == "len":
                try:
                    return len(args[0])
                except Exception:
                    self.errors.append("unsupported_len")
                    return None
        if isinstance(callable_value, _Method):
            return self._call_method(callable_value.owner, callable_value.name, args, kwargs)
        self.errors.append(f"unsupported_call={type(callable_value).__name__}")
        return None

    def _call_method(self, owner: Any, name: str, args: list[Any], kwargs: dict[str, Any]) -> Any:
        if isinstance(owner, _Workbook):
            if name == "create_sheet":
                title = args[0] if args else kwargs.get("title", "Sheet")
                if not isinstance(title, str):
                    self.errors.append("nonliteral_sheet_title")
                    return None
                sheet = _Sheet(title)
                index = kwargs.get("index")
                if index is None:
                    owner.sheets.append(sheet)
                elif isinstance(index, int):
                    owner.sheets.insert(index, sheet)
                else:
                    self.errors.append("nonliteral_sheet_index")
                return sheet
            if name == "remove" and args and isinstance(args[0], _Sheet):
                owner.sheets.remove(args[0])
                return None
            if name == "save":
                self._save("xlsx", owner, args)
                return None
        if isinstance(owner, _Sheet):
            if name == "append" and args and isinstance(args[0], (list, tuple)):
                rows = [self._coordinate_row(key) for key, cell in owner.cells.items() if cell.value is not None]
                row_number = max(rows, default=0) + 1
                for column_number, value in enumerate(args[0], 1):
                    owner.cells[self._coordinate(column_number, row_number)] = _XCell(value=value)
                return None
            if name == "cell":
                row = kwargs.get("row", args[0] if args else None)
                column = kwargs.get("column", args[1] if len(args) > 1 else None)
                if not isinstance(row, int) or not isinstance(column, int):
                    self.errors.append("nonliteral_cell_coordinate")
                    return None
                cell = owner.cells.setdefault(self._coordinate(column, row), _XCell())
                if "value" in kwargs:
                    cell.value = kwargs["value"]
                elif len(args) > 2:
                    cell.value = args[2]
                return cell
            if name == "add_chart" and args and isinstance(args[0], _Chart):
                chart = args[0]
                chart.anchor = args[1] if len(args) > 1 else kwargs.get("anchor")
                owner.charts.append(chart)
                return None
        if isinstance(owner, _Chart):
            if name == "add_data" and args and isinstance(args[0], _Reference):
                owner.data = args[0]
                owner.titles_from_data = kwargs.get("titles_from_data")
                return None
            if name == "set_categories" and args and isinstance(args[0], _Reference):
                owner.categories = args[0]
                return None
        if isinstance(owner, _Document):
            if name == "add_paragraph":
                text = args[0] if args else kwargs.get("text", "")
                style = kwargs.get("style", args[1] if len(args) > 1 else "Normal")
                if not isinstance(text, str) or not isinstance(style, str):
                    self.errors.append("nonliteral_paragraph")
                    return None
                paragraph = _Paragraph(text=text, style=style)
                owner.paragraphs.append(paragraph)
                return paragraph
            if name == "add_heading":
                text = args[0] if args else kwargs.get("text", "")
                level = kwargs.get("level", args[1] if len(args) > 1 else 1)
                if not isinstance(text, str) or not isinstance(level, int):
                    self.errors.append("nonliteral_heading")
                    return None
                paragraph = _Paragraph(text=text, style=f"Heading {level}")
                owner.paragraphs.append(paragraph)
                return paragraph
            if name == "add_table":
                rows = kwargs.get("rows", args[0] if args else None)
                cols = kwargs.get("cols", args[1] if len(args) > 1 else None)
                if not isinstance(rows, int) or not isinstance(cols, int) or rows < 0 or cols < 0:
                    self.errors.append("nonliteral_table_shape")
                    return None
                table = _Table([[_DCell() for _ in range(cols)] for _ in range(rows)])
                owner.tables.append(table)
                return table
            if name == "save":
                self._save("docx", owner, args)
                return None
        if isinstance(owner, _Paragraph) and name == "add_run":
            text = args[0] if args else kwargs.get("text", "")
            if not isinstance(text, str):
                self.errors.append("nonliteral_run")
                return None
            run = _Run(text=text)
            owner.runs.append(run)
            owner.text += text
            return run
        if isinstance(owner, _Table) and name == "cell":
            if len(args) < 2 or not all(isinstance(value, int) for value in args[:2]):
                self.errors.append("nonliteral_table_cell")
                return None
            try:
                return owner.cells[args[0]][args[1]]
            except IndexError:
                self.errors.append(f"table_cell_index={args[0]},{args[1]}")
                return None
        if isinstance(owner, _SlidesProxy) and name == "add_slide":
            if not args or not isinstance(args[0], _Layout):
                self.errors.append("nonliteral_slide_layout")
                return None
            layout_index = args[0].index
            slide = _Slide(layout_index, None if layout_index == 6 else _TitleShape())
            owner.presentation.slides.append(slide)
            return slide
        if isinstance(owner, _ShapesProxy) and name.startswith("add_"):
            owner.slide.extra_shapes += 1
            return _TitleShape() if name in {"add_textbox", "add_shape"} else _Symbol("shape")
        if isinstance(owner, _Presentation) and name == "save":
            self._save("pptx", owner, args)
            return None
        if isinstance(owner, list) and name == "append":
            if len(args) != 1:
                self.errors.append("list_append_arity")
                return None
            owner.append(args[0])
            return None
        self.errors.append(f"unsupported_method={type(owner).__name__}.{name}")
        return None

    def _save(self, kind: str, owner: Any, args: list[Any]) -> None:
        if not args or not isinstance(args[0], _Symbol) or args[0].name != self.path_parameter:
            self.errors.append("save_does_not_use_path_parameter")
            return
        if kind == "xlsx":
            model = self._serialize_workbook(owner)
        elif kind == "docx":
            model = self._serialize_document(owner)
        else:
            model = self._serialize_presentation(owner)
        self.artifacts.append((kind, copy.deepcopy(model)))

    @staticmethod
    def _coordinate(column: int, row: int) -> str:
        letters = ""
        current = column
        while current:
            current, remainder = divmod(current - 1, 26)
            letters = chr(ord("A") + remainder) + letters
        return f"{letters}{row}"

    @staticmethod
    def _coordinate_row(coordinate: str) -> int:
        match = re.fullmatch(r"[A-Z]+([0-9]+)", coordinate)
        return int(match.group(1)) if match else 0

    @staticmethod
    def _serialize_reference(reference: _Reference | None) -> dict[str, Any] | None:
        if reference is None:
            return None
        return {
            "sheet": reference.sheet.title,
            "min_col": reference.min_col,
            "min_row": reference.min_row,
            "max_col": reference.max_col,
            "max_row": reference.max_row,
        }

    def _serialize_workbook(self, workbook: _Workbook) -> dict[str, Any]:
        sheets: dict[str, Any] = {}
        for sheet in workbook.sheets:
            cells = {
                coordinate: {
                    "value": cell.value,
                    "font": {"bold": cell.font.bold, "italic": cell.font.italic, "underline": cell.font.underline},
                }
                for coordinate, cell in sheet.cells.items()
                if cell.value is not None or any(
                    value is not None for value in (cell.font.bold, cell.font.italic, cell.font.underline)
                )
            }
            charts = [
                {
                    "type": chart.chart_type,
                    "anchor": chart.anchor,
                    "data": self._serialize_reference(chart.data),
                    "categories": self._serialize_reference(chart.categories),
                    "titles_from_data": chart.titles_from_data,
                }
                for chart in sheet.charts
            ]
            sheets[sheet.title] = {"cells": cells, "charts": charts}
        return {"sheet_names": [sheet.title for sheet in workbook.sheets], "sheets": sheets}

    @staticmethod
    def _serialize_document(document: _Document) -> dict[str, Any]:
        return {
            "paragraphs": [
                {
                    "text": paragraph.text,
                    "style": paragraph.style,
                    "direct_formatting": paragraph.direct_formatting
                    or any(run.direct_formatting for run in paragraph.runs),
                }
                for paragraph in document.paragraphs
            ],
            "tables": [
                [[cell.text for cell in row] for row in table.cells]
                for table in document.tables
            ],
        }

    @staticmethod
    def _serialize_presentation(presentation: _Presentation) -> dict[str, Any]:
        return {
            "slides": [
                {
                    "layout_index": slide.layout_index,
                    "layout_has_title": slide.title_shape is not None,
                    "title": slide.title_shape.text if slide.title_shape is not None else None,
                    "extra_shapes": slide.extra_shapes,
                }
                for slide in presentation.slides
            ]
        }


def _subset_match(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _subset_match(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _subset_match(a, e) for a, e in zip(actual, expected)
        )
    return actual == expected


def _validate_xlsx(model: dict[str, Any], spec: dict[str, Any]) -> str | None:
    if "sheet_names" in spec and model["sheet_names"] != spec["sheet_names"]:
        return f"sheet_names={model['sheet_names']}"
    required = spec.get("required_sheet_names", [])
    if not set(required).issubset(model["sheet_names"]):
        return f"missing_sheets={sorted(set(required) - set(model['sheet_names']))}"
    expected_cells = spec.get("cells", {})
    for sheet_name, cells in expected_cells.items():
        sheet = model["sheets"].get(sheet_name)
        if sheet is None:
            return f"missing_sheet={sheet_name}"
        for coordinate, expected in cells.items():
            cell = sheet["cells"].get(coordinate.upper())
            if cell is None or not _subset_match(cell, expected if isinstance(expected, dict) else {"value": expected}):
                return f"cell_mismatch={sheet_name}!{coordinate}"
    if spec.get("exact_nonempty_cells"):
        actual = {
            f"{sheet_name}!{coordinate}"
            for sheet_name, sheet in model["sheets"].items()
            for coordinate, cell in sheet["cells"].items()
            if cell["value"] is not None
        }
        expected = {
            f"{sheet_name}!{coordinate.upper()}"
            for sheet_name, cells in expected_cells.items()
            for coordinate, value in cells.items()
            if (value.get("value") if isinstance(value, dict) else value) is not None
        }
        if actual != expected:
            return f"nonempty_cells={sorted(actual)}"
    for expected_chart in spec.get("charts", []):
        sheet_name = expected_chart.get("sheet")
        sheet = model["sheets"].get(sheet_name)
        if sheet is None:
            return f"chart_missing_sheet={sheet_name}"
        expected_without_sheet = {key: value for key, value in expected_chart.items() if key != "sheet"}
        if not any(_subset_match(chart, expected_without_sheet) for chart in sheet["charts"]):
            return f"missing_matching_chart={sheet_name}"
    return None


def _validate_docx(model: dict[str, Any], spec: dict[str, Any]) -> str | None:
    if "paragraphs" in spec and not _subset_match(model["paragraphs"], spec["paragraphs"]):
        return f"paragraphs={model['paragraphs']}"
    if "nonempty_paragraphs" in spec:
        nonempty = [paragraph for paragraph in model["paragraphs"] if paragraph["text"]]
        if not _subset_match(nonempty, spec["nonempty_paragraphs"]):
            return f"nonempty_paragraphs={nonempty}"
    if "tables" in spec and model["tables"] != spec["tables"]:
        return f"tables={model['tables']}"
    return None


def _validate_pptx(model: dict[str, Any], spec: dict[str, Any]) -> str | None:
    if "slides" in spec and not _subset_match(model["slides"], spec["slides"]):
        return f"slides={model['slides']}"
    return None


def validate_ooxml_ast(
    tree: ast.Module,
    function_name: str,
    spec: dict[str, Any],
) -> tuple[bool, str, dict[str, Any] | None]:
    """Validate one saved symbolic OOXML artifact against ``spec``."""
    kind = spec.get("kind")
    if kind not in {"xlsx", "docx", "pptx"}:
        return False, f"unsupported_ooxml_kind={kind}", None
    interpreter = _StaticInterpreter(tree, function_name)
    interpreter.run()
    if interpreter.errors:
        return False, interpreter.errors[0], None
    artifacts = [model for artifact_kind, model in interpreter.artifacts if artifact_kind == kind]
    if not artifacts:
        return False, f"no_saved_{kind}_artifact", None
    model = artifacts[-1]
    if kind == "xlsx":
        error = _validate_xlsx(model, spec)
    elif kind == "docx":
        error = _validate_docx(model, spec)
    else:
        error = _validate_pptx(model, spec)
    return error is None, error or "ooxml_static_passed", model
