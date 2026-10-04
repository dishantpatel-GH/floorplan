"""Judge criterion 4 (explainability / code clarity): size and shape of each extractor's code, measured, not guessed.

    python scripts/judge_code_metrics.py

Counts per package: files, code lines (no blanks, comments or docstrings), functions, long functions (> 50 lines:
hard to explain in one breath at the defense), the longest function, functions without a docstring, and the number
of tunable parameters. These are proxies; the judge doc pairs them with a reading of the code.
Writes outputs/judge/code_metrics.json.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from judge_common import OUT, ROOT  # noqa: E402

PACKAGES = {"alpha": ROOT / "floorplan/plan/alpha", "beta": ROOT / "floorplan/plan/beta"}
LONG_FUNCTION = 50


def code_lines(src: str, tree: ast.AST) -> int:
    """Lines that hold code: not blank, not a comment, not inside a docstring."""
    doc_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) \
                    and isinstance(first.value.value, str):
                doc_lines.update(range(first.lineno, first.end_lineno + 1))
    return sum(1 for i, line in enumerate(src.splitlines(), 1)
               if line.strip() and not line.strip().startswith("#") and i not in doc_lines)


def package_metrics(pkg: Path) -> dict:
    files, loc, funcs, long_funcs, undocumented, longest = 0, 0, 0, [], 0, ("", 0)
    n_params = 0
    for f in sorted(pkg.glob("*.py")):
        src = f.read_text()
        tree = ast.parse(src)
        files += 1
        loc += code_lines(src, tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                funcs += 1
                n = node.end_lineno - node.lineno + 1
                if n > LONG_FUNCTION:
                    long_funcs.append(f"{f.name}:{node.name} ({n})")
                if n > longest[1]:
                    longest = (f"{f.name}:{node.name}", n)
                if ast.get_docstring(node) is None and not node.name.startswith("_"):
                    undocumented += 1
            if isinstance(node, ast.ClassDef) and node.name.endswith("Params"):
                n_params = sum(isinstance(s, ast.AnnAssign) for s in node.body)
    return dict(files=files, code_lines=loc, functions=funcs, long_functions=long_funcs,
                longest_function=dict(name=longest[0], lines=longest[1]),
                public_functions_without_docstring=undocumented, tunable_parameters=n_params)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    result = {name: package_metrics(path) for name, path in PACKAGES.items()}
    (OUT / "code_metrics.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
