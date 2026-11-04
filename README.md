# Learn DSPy in VS Code

Start with **[dspy_fundamentals.ipynb](dspy_fundamentals.ipynb)**. The core lesson takes about **90 minutes**; optional optimization and experimentation bring it to **2–3 hours**.

You will build a support-ticket classifier, inspect model requests, define typed signatures, compose a two-step program, and evaluate predictions. The lesson includes editable code, exercises, suggested answers, and troubleshooting. Basic Python is enough.

## Start here

1. Open this folder in VS Code. Install the recommended Microsoft **Python** and **Jupyter** extensions.
2. Open `dspy_fundamentals.ipynb`.
3. Click **Select Kernel → Python Environments** and choose this project's `.venv` (Python 3.13). If necessary, use **Select Another Kernel** or browse to `.venv\Scripts\python.exe`.
4. Run cells from the top with **Shift+Enter**. The setup cell requests your model API key privately. Stop after section 5 for the core lesson.

A local `.venv` has been created with the tutorial dependencies. If recreating it, run these commands in a PowerShell terminal in this folder (using [uv](https://docs.astral.sh/uv/)):

```powershell
uv venv --python 3.13 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

Alternatively, if Python 3.13 is installed and available through `py`:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Activation is unnecessary when using these explicit paths. Select the kernel separately in VS Code. The first notebook cell prints the actual interpreter path so you can verify it.

## Model access

The supplied configuration uses `openai/gpt-4o-mini`, as illustrated in the [DSPy language-model documentation](https://dspy.ai/learn/programming/language_models/). It requires a suitable provider API key, model access, and API credit. Model calls can incur charges. The core lesson uses roughly 20 logical calls before extra reruns or retries; optional labs are disabled until you change their switches.

The key comes from `OPENAI_API_KEY` or a hidden prompt, never from a saved code literal. The setup does not make a model call. You can adapt the model configuration for another provider using the linked guide. Examples are invented support messages.

## Files

- `dspy_fundamentals.ipynb`: the main interactive notebook; edit and save your work here.
- `dspy_fundamentals.py`: the same lesson in VS Code `# %%` cell format. Use **Run Cell** / **Run Above** in the Interactive Window if you prefer a Python file. This is an alternative, not a second lesson.
- `requirements.txt`: pinned DSPy and notebook-kernel dependencies.
- `requirements-dev.txt`: additional notebook build and verification dependencies.
- `scripts/build_notebook.py`: rebuilds the notebook from the `.py` source. **This overwrites notebook edits and outputs; it is not needed to take the lesson.**
- `scripts/verify_notebook.py`: offline execution checks using explicitly scripted model responses.

The two lesson formats are not automatically synchronized. Choose one for your personal edits. Generated caches and optional saved program state are ignored by Git.

## Verification

The notebook is validated against its Jupyter schema and executed with scripted responses using real DSPy APIs, including the optional optimization, evaluation, and save/reload paths. Offline checks validate program wiring, not model quality or provider authentication. No paid model requests were made while preparing the lesson.

To rerun those checks:

```powershell
uv pip install --python .venv\Scripts\python.exe -r requirements-dev.txt
.\.venv\Scripts\python.exe scripts/verify_notebook.py
```

See [VS Code's notebook guide](https://code.visualstudio.com/docs/datascience/jupyter-notebooks) for editor controls and kernel selection.
