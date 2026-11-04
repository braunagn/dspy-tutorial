r"""Build the notebook companion from VS Code percent-format lesson cells.

Run from the project root: .venv\Scripts\python scripts/build_notebook.py
Rebuilding replaces the notebook, including any learner edits and outputs.
"""
from pathlib import Path
import re
import nbformat

ROOT = Path(__file__).resolve().parents[1]


def build():
    source = (ROOT / "dspy_fundamentals.py").read_text(encoding="utf-8")
    parts = re.split(r"^# %%([^\n]*)\n", source, flags=re.MULTILINE)
    cells = []
    for index in range(1, len(parts), 2):
        label, body = parts[index].strip(), parts[index + 1].rstrip()
        if label == "[markdown]":
            markdown = "\n".join(
                line[2:] if line.startswith("# ") else "" if line == "#" else line
                for line in body.splitlines()
            )
            cell = nbformat.v4.new_markdown_cell(markdown)
        else:
            cell = nbformat.v4.new_code_cell(body, metadata={"tags": [label]})
        cell.id = f"lesson-{len(cells):02d}"
        cells.append(cell)
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13.6"},
    })
    nbformat.validate(notebook)
    destination = ROOT / "dspy_fundamentals.ipynb"
    nbformat.write(notebook, destination)
    print(f"Built {destination.name}: {len(cells)} cells")


if __name__ == "__main__":
    build()
