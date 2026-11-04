"""Execute both lesson paths with scripted responses; never call a provider.

This checks notebook execution and DSPy integration, not language-model quality.
Temporary executed copies and a local kernel specification live under .tmp/.
"""
import copy
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TEMP = ROOT / ".tmp" / "verification"
TEMP.mkdir(parents=True, exist_ok=True)
os.environ["JUPYTER_PATH"] = str(TEMP / "jupyter")
os.environ["JUPYTER_RUNTIME_DIR"] = str(TEMP / "runtime")
os.environ["IPYTHONDIR"] = str(TEMP / "ipython")
os.environ["DSPY_CACHEDIR"] = str(TEMP / "cache")

import nbformat
from nbclient import NotebookClient

KERNEL = TEMP / "jupyter" / "kernels" / "dspy-tutorial-check"
KERNEL.mkdir(parents=True, exist_ok=True)
(KERNEL / "kernel.json").write_text(json.dumps({
    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    "display_name": "DSPy offline verification", "language": "python",
}), encoding="utf-8")

SETUP = '''
from dspy.utils import DummyLM

def set_scripted_answers(answers):
    global lm
    lm = DummyLM(answers)
    dspy.configure(lm=lm)

def block_live_model(*args, **kwargs):
    raise AssertionError("Live LM creation is forbidden in offline verification")

dspy.LM = block_live_model
set_scripted_answers([{"category": "billing"}])
print("OFFLINE VERIFICATION: scripted responses, no provider calls")
'''

FIXTURES = {
    "first_prediction": 'set_scripted_answers([{"category": "billing"}])',
    "freeform_exercise": 'set_scripted_answers([{"category": "account"}])',
    "typed_prediction": 'set_scripted_answers([{"category": "billing"}])',
    "policy_exercise": 'set_scripted_answers([{"category": x} for x in ["technical", "account", "billing"]])',
    "cot_prediction": 'set_scripted_answers([{"reasoning": "Scripted explanation for testing.", "category": "billing"}])',
    "pipeline_prediction": '''set_scripted_answers([
        {"category": "billing"},
        {"reply": "Thanks for flagging this charge. Our billing team can review it."},
    ])''',
    "evaluate": '''set_scripted_answers({
        ex.ticket: {"category": ex.category if i != 5 else "billing"}
        for i, ex in enumerate(devset)
    })''',
    "optimize": '''set_scripted_answers({
        ex.ticket: {"category": ex.category} for ex in trainset + devset
    })''',
    "experiment": '''set_scripted_answers({
        ex.ticket: {"reasoning": "Scripted fixture.", "category": ex.category}
        for ex in devset
    })''',
}


def verify(optional):
    original = nbformat.read(ROOT / "dspy_fundamentals.ipynb", as_version=4)
    nbformat.validate(original)
    notebook = copy.deepcopy(original)
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        compile(cell.source, f"cell-{cell.id}", "exec")
        assert cell.execution_count is None and not cell.outputs, "Deliver a clean notebook"
        tag = cell.metadata["tags"][0]
        if tag == "configure_lm":
            cell.source = SETUP
        else:
            if tag in FIXTURES:
                cell.source = FIXTURES[tag] + "\n" + cell.source
            if optional:
                cell.source = cell.source.replace("RUN_OPTIMIZATION = False", "RUN_OPTIMIZATION = True")
                cell.source = cell.source.replace("RUN_EXPERIMENT = False", "RUN_EXPERIMENT = True")
    checks = '''
assert len(baseline.results) == 6
assert len(mistakes) == 1
assert abs(baseline.score - 83.33) < 0.01
assert response.category == "billing" and response.reply
assert set(devset[0].inputs().keys()) == {"ticket"}
'''
    if optional:
        checks += '''
assert len(optimized.demos) == 2
assert len(restored.demos) == 2
assert optimized_result.score == 100
assert challenger_result.score == 100
assert state_path.exists()
'''
    else:
        checks += "assert optimized is None and optimized_result is None\n"
    notebook.cells.append(nbformat.v4.new_code_cell(checks))
    mode = "optional-enabled" if optional else "core-default"
    workdir = TEMP / mode
    workdir.mkdir(exist_ok=True)
    client = NotebookClient(
        notebook, timeout=120, kernel_name="dspy-tutorial-check",
        resources={"metadata": {"path": str(workdir)}},
    )
    client.execute()
    nbformat.write(notebook, workdir / "executed.ipynb")
    print(f"PASS: {mode}; all code cells executed with scripted responses")


if __name__ == "__main__":
    verify(optional=False)
    verify(optional=True)
