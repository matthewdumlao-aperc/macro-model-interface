"""Execute a Python notebook without requiring Jupyter or IPython.

This lightweight executor supports the output types used by the GDP extremes
notebooks: stdout/stderr, rich HTML tables, plain-text values, and Matplotlib
figures. It is intended for reproducibly persisting the bundled notebook
results with the project's existing Python interpreter.
"""

from __future__ import annotations

import argparse
import ast
import base64
import contextlib
import io
import json
import os
import traceback
from pathlib import Path


os.environ.setdefault("MPLBACKEND", "Agg")


def _execute_source(source: str, namespace: dict, filename: str):
    """Execute a cell and return the value of its final expression, if any."""
    tree = ast.parse(source, filename=filename, mode="exec")
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        prefix = ast.Module(body=tree.body[:-1], type_ignores=[])
        if prefix.body:
            exec(compile(prefix, filename, "exec"), namespace)
        expression = ast.Expression(tree.body[-1].value)
        return eval(compile(expression, filename, "eval"), namespace)
    exec(compile(tree, filename, "exec"), namespace)
    return None


def _value_output(value) -> dict | None:
    """Convert a final expression to a notebook display output."""
    if value is None:
        return None
    data = {"text/plain": repr(value)}
    html_renderer = getattr(value, "_repr_html_", None)
    if callable(html_renderer):
        html = html_renderer()
        if html is not None:
            data["text/html"] = html
    return {"output_type": "display_data", "metadata": {}, "data": data}


def _figure_outputs() -> list[dict]:
    """Capture and close all currently open Matplotlib figures."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    outputs = []
    for figure_number in plt.get_fignums():
        figure = plt.figure(figure_number)
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
        outputs.append(
            {
                "output_type": "display_data",
                "metadata": {},
                "data": {
                    "text/plain": f"<Figure size {figure.get_size_inches()} with {len(figure.axes)} Axes>",
                    "image/png": base64.b64encode(buffer.getvalue()).decode("ascii"),
                },
            }
        )
    plt.close("all")
    return outputs


def execute_notebook(path: Path) -> None:
    """Execute code cells in *path* and persist their outputs in place."""
    path = path.resolve()
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    execution_count = 0

    for cell_index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue

        execution_count += 1
        cell["execution_count"] = execution_count
        cell["outputs"] = []
        source = "".join(cell.get("source", []))
        stdout = io.StringIO()
        stderr = io.StringIO()

        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                value = _execute_source(
                    source,
                    namespace,
                    f"{path.name}:cell-{cell_index}",
                )
        except Exception as exc:
            standard_output = stdout.getvalue()
            standard_error = stderr.getvalue()
            if standard_output:
                cell["outputs"].append(
                    {"output_type": "stream", "name": "stdout", "text": standard_output}
                )
            if standard_error:
                cell["outputs"].append(
                    {"output_type": "stream", "name": "stderr", "text": standard_error}
                )
            cell["outputs"].append(
                {
                    "output_type": "error",
                    "ename": type(exc).__name__,
                    "evalue": str(exc),
                    "traceback": traceback.format_exc().splitlines(),
                }
            )
            path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
            raise

        standard_output = stdout.getvalue()
        standard_error = stderr.getvalue()
        if standard_output:
            cell["outputs"].append(
                {"output_type": "stream", "name": "stdout", "text": standard_output}
            )
        if standard_error:
            cell["outputs"].append(
                {"output_type": "stream", "name": "stderr", "text": standard_error}
            )
        value_output = _value_output(value)
        if value_output is not None:
            cell["outputs"].append(value_output)
        cell["outputs"].extend(_figure_outputs())

    path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(f"Executed {execution_count} code cells and updated {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="+", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    for notebook_path in arguments.notebooks:
        execute_notebook(notebook_path)
