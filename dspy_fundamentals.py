# %% [markdown]
# # DSPy fundamentals: build a tiny ticket classifier
#
# **Time:** about 90 minutes for sections 0–5; 2–3 hours with the optional labs.
# **Prerequisites:** Python variables, functions, lists, dictionaries, and a little familiarity
# with classes. No machine-learning training experience is needed.
#
# You will turn a support message into `billing`, `technical`, or `account`, inspect
# what DSPy sends to a language model, compose two steps, and measure correctness.
# All tickets are invented. You only need this lesson and a model API key.
#
# | Section | Budget | You will do |
# |---|---:|---|
# | 0. Setup and the mental model | 15 min | Select a kernel and connect a model |
# | 1. Your first prediction | 10 min | Run, inspect, and change an input |
# | 2. Explicit signatures | 20 min | Define a task and typed outputs |
# | 3. Compose a program | 20 min | Join two model steps with Python |
# | 4. Evaluate | 20 min | Create examples and a correctness metric |
# | 5. Check your understanding | 5 min | Explain the pieces yourself |
# | 6. Optional: optimization | 20–30 min | Collect and inspect demonstrations |
# | 7. Optional: your own experiment | 20–40 min | Make one controlled change |
#
# The time includes experimenting and answering questions, not just running cells.
# Keep your notes in the provided Markdown cells. Suggested answers are folded below exercises.
#
# ## 0. Setup and the mental model
#
# Open the project folder in VS Code and install Microsoft's **Python** and **Jupyter**
# extensions. Open `dspy_fundamentals.ipynb`, choose **Select Kernel → Python Environments**,
# and select this project's `.venv` (Python 3.13). See `README.md` if it is missing.
# Use **Shift+Enter** to execute a cell. The `.py` companion offers the same lesson
# through VS Code's **Run Cell** buttons; choose either format to work in.
#
# A notebook kernel remembers variables. Run from the top on your first pass. After changing
# a class definition, rerun the cell that creates its instances and the downstream cells.
# **Restart Kernel** clears that memory; then rerun setup. Cell outputs are your own results;
# this notebook deliberately ships without precomputed model answers.
#
# DSPy is a Python framework for building programs that call language models.
# You describe the task using named inputs and outputs. DSPy constructs the messages
# and parses the response into an object you can use in Python.
#
# ```text
# your input + signature → module → language model → Prediction with named fields
#                                                  ↓
#                                examples + metric measure how well it works
# ```
#
# | Piece | Plain meaning | In this lesson |
# |---|---|---|
# | LM | The model connection | `lm` |
# | Signature | Task instructions, inputs, outputs | `TicketCategory` |
# | Module | Callable program that performs a task | `dspy.Predict(...)` |
# | Prediction | Returned named values | `result.category` |
# | Example | Input plus a reference answer | A labeled ticket |
# | Metric | A function that grades an answer | Category matches the reference |
# | Optimizer | Uses examples to adjust a program | Optional demonstration collection |
#
# Creating a signature or module does **not** train anything or call the model.
# Calling the module does. Optimization is a separate, explicit operation.
# [Background: DSPy overview](https://dspy.ai/learn/).

# %% imports
import os
import sys
from pathlib import Path
from getpass import getpass
from typing import Literal
from importlib.metadata import version

# Keep DSPy's disk cache inside this project. Set this before importing DSPy.
os.environ["DSPY_CACHEDIR"] = str(Path.cwd() / ".cache" / "dspy")

import dspy

print("Python:", sys.version.split()[0])
print("Kernel:", sys.executable)
print("DSPy:", version("dspy"))
print("Working directory:", Path.cwd())

# %% [markdown]
# ### Connect a model
#
# This example uses the model identifier shown in DSPy's language-model guide:
# `openai/gpt-4o-mini`. You need a provider API key with access to that model and
# available API credit. The notebook prompts privately if `OPENAI_API_KEY` is not
# already in your environment; do not paste a key into a source cell.
#
# The configuration cell makes no model request. Later cells marked **MODEL CALL**
# send their inputs to the provider and can incur usage charges. One pass through
# the core lesson makes roughly 20 logical model calls, with more for exercises,
# retries, or reruns. The optional optimizer adds up to six training attempts and
# six evaluation calls, plus one reload check. These are estimates, not billing caps.
#
# DSPy caches calls by default. Repeating the same call may reuse an answer.
# `temperature=0` reduces sampling variation but does not guarantee reproducibility.
# If you already use a different provider, adjust `MODEL` and its credential setup
# using the [DSPy LM guide](https://dspy.ai/learn/programming/language_models/).

# %% configure_lm
MODEL = "openai/gpt-4o-mini"

if not os.environ.get("OPENAI_API_KEY"):
    api_key = getpass("OpenAI API key (hidden): ").strip()
    if not api_key:
        raise ValueError("Enter an API key, then rerun this cell.")
    os.environ["OPENAI_API_KEY"] = api_key
    del api_key

lm = dspy.LM(MODEL, temperature=0, max_tokens=500, cache=True)
dspy.configure(lm=lm)
print("Model configured:", MODEL)

# %% [markdown]
# ## 1. Your first prediction — 10 minutes
#
# A string signature reads from left to right: given `ticket`, produce `category`.
# Both fields default to strings. Their names convey meaning to the model.
# This first version intentionally leaves the category choices unspecified.
#
# Before running: what label do you expect? Would `payment`, `billing`, and
# `duplicate charge` all be reasonable answers to this underspecified task?
# [Reference: signatures](https://dspy.ai/learn/programming/signatures/).

# %% first_prediction
classify_freeform = dspy.Predict("ticket -> category")  # Definition: no request.

ticket = "I was charged twice for my monthly subscription."
first = classify_freeform(ticket=ticket)  # MODEL CALL: 1
print(first)
print("Category:", first.category)
print("Python type:", type(first.category).__name__)

# %% [markdown]
# ### Look at what actually happened
#
# Inspect the generated messages and response. Find your input, field names,
# task instructions, and requested output format. DSPy still uses prompts;
# it builds them from your program's structure.
# This history contains inputs and outputs, so use the invented tickets here.

# %% inspect_first
dspy.inspect_history(n=1)  # Inspection only; no new request.

# %% [markdown]
# **Try it:** change the ticket below to a password-reset problem. Run the cell,
# record the label, then try a vague message such as “It doesn't work.”
# Do you get a consistent set of categories?

# %% freeform_exercise
my_ticket = "I forgot my password and cannot sign in."
my_result = classify_freeform(ticket=my_ticket)  # MODEL CALL: 1 per run
print(my_result.category)

# %% [markdown]
# **Your notes:** Expected label: … Actual label: … What is underspecified: …
#
# <details><summary>Suggested interpretation</summary>
# A plausible answer is not necessarily a usable label. A router needs a shared
# vocabulary and a policy for overlap. Fix the task definition before judging quality.
# </details>
#
# ## 2. Make the task explicit — 20 minutes
#
# We will use a small routing policy:
#
# - **billing:** payments, invoices, refunds, or subscriptions.
# - **account:** passwords, sign-in, or profile access.
# - **technical:** crashes, errors, or other product behavior; also our fallback
#   when the message gives too little information.
#
# If multiple categories apply, use billing first, then account, then technical.
# This is a made-up policy for the exercise, not a universal support taxonomy.
#
# A class signature gives us a place for task instructions and field descriptions.
# `Literal` lists allowed output values. DSPy uses these types when parsing outputs;
# a valid label can still be the wrong label. Type constraints do not guarantee factual
# correctness, and malformed responses can cause parsing errors.

# %% typed_signature
Category = Literal["billing", "technical", "account"]

class TicketCategory(dspy.Signature):
    """Route the ticket to one category.

    billing: payments, invoices, refunds, or subscriptions.
    account: passwords, sign-in, or profile access.
    technical: crashes, errors, or other product behavior.
    For overlap, prefer billing, then account, then technical.
    If there is too little information, use technical.
    """

    ticket: str = dspy.InputField(desc="The customer's support message")
    category: Category = dspy.OutputField(desc="One category under the routing policy")

classify = dspy.Predict(TicketCategory)

# %% typed_prediction
typed = classify(ticket=ticket)  # MODEL CALL: 1
print(typed)
assert typed.category in ("billing", "technical", "account")
print("Allowed label confirmed; correctness still needs evaluation.")

# %% [markdown]
# **Experiment:** predict the labels for these three tickets on paper first.
# Then run the cell. Explain the last case using the policy rather than intuition.

# %% policy_exercise
policy_tickets = [
    "The desktop app crashes every time I open a project.",
    "Please help me change the password on my profile.",
    "I cannot sign in and I want a refund for this month's subscription.",
]
for text in policy_tickets:  # MODEL CALLS: 3
    print(classify(ticket=text).category, "|", text)

# %% [markdown]
# <details><summary>Reference labels</summary>
# technical, account, billing. The final message activates two categories;
# the explicit precedence rule makes billing the reference answer.
# </details>
#
# ### Same signature, another module
#
# `ChainOfThought` adds a generated `reasoning` field before the answer.
# We can swap the module while keeping the task signature. Inspect that field
# as model-generated explanatory text; it is not proof the answer is right.
# A longer response is not automatically better and may use more tokens.
# [Reference: modules](https://dspy.ai/learn/programming/modules/).

# %% cot_prediction
classify_with_reasoning = dspy.ChainOfThought(TicketCategory)
reasoned = classify_with_reasoning(ticket=policy_tickets[-1])  # MODEL CALL: 1
print("Reasoning:", reasoned.reasoning)
print("Category:", reasoned.category)

# %% [markdown]
# **Check:** Which part changed: task, model connection, or prompting strategy?
# What evidence would convince you to keep the change?
#
# **Your notes:** …
#
# <details><summary>Suggested answer</summary>
# The module's prompting strategy changed. The signature and configured LM stayed
# the same. Compare correctness on the same examples, alongside latency and usage,
# before preferring the more verbose module.
# </details>
#
# ## 3. Compose a small program — 20 minutes
#
# Add a second step that drafts a brief acknowledgment. Our data flow is:
#
# ```text
# ticket → classify → category
#    └────────────────┴──→ acknowledge → reply
# ```
#
# `__init__` creates the two predictors. `forward` is ordinary Python that connects
# them. Calling `assistant(ticket=...)` runs `forward` through DSPy's module machinery.
# Return a `Prediction` so callers can inspect both outputs. Nothing is sent to a
# customer; the program only produces text in your notebook.

# %% pipeline_definition
class AcknowledgeTicket(dspy.Signature):
    """Write a friendly acknowledgment of at most two sentences.

    Acknowledge the issue and say the relevant team can review it.
    Do not claim to have fixed it, accessed the account, or issued a refund.
    """

    ticket: str = dspy.InputField()
    category: Category = dspy.InputField()
    reply: str = dspy.OutputField()


class TicketAssistant(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.Predict(TicketCategory)
        self.acknowledge = dspy.Predict(AcknowledgeTicket)

    def forward(self, ticket):
        classification = self.classify(ticket=ticket)
        acknowledgment = self.acknowledge(
            ticket=ticket, category=classification.category
        )
        return dspy.Prediction(
            category=classification.category,
            reply=acknowledgment.reply,
        )


assistant = TicketAssistant()

# %% pipeline_prediction
response = assistant(ticket="My invoice includes a charge I do not recognize.")  # MODEL CALLS: 2
print("Route:", response.category)
print("Draft:", response.reply)

# %% [markdown]
# **Inspect:** expand the two history entries. Find the category passed into the
# second request. What happens to the reply if the first step chooses the wrong category?

# %% pipeline_history
dspy.inspect_history(n=2)

# %% [markdown]
# **Try it (5 minutes):** edit the acknowledgment instructions to ask for one sentence.
# Rerun the signature definition, the `TicketAssistant` definition/instantiation cell,
# and the prediction cell. Does the result obey the instruction? Do not assume it will.
#
# **Your notes:** What changed? … What stayed the same? …
#
# <details><summary>What to notice</summary>
# A new instruction affects the acknowledgment step. It does not alter the category
# policy. The reply depends on classification, so an early error can propagate.
# Existing instances retain their original signatures until you recreate them.
# </details>
#
# ## 4. Measure correctness — 20 minutes
#
# A few appealing outputs are weak evidence. Create a small labeled dataset and
# apply the same scoring rule to every prediction. We will score the **classifier**;
# that score says nothing about the quality or truthfulness of the acknowledgment.
#
# `dspy.Example` stores inputs and reference outputs. `.with_inputs("ticket")` marks
# the field the program receives; `category` stays available to the metric as the
# reference answer. Passing the reference answer as input would leak the answer.
# [Reference: data handling](https://dspy.ai/learn/evaluation/data/).

# %% dataset
train_rows = [
    ("Please refund my duplicate payment.", "billing"),
    ("I forgot my password.", "account"),
    ("The mobile app freezes on launch.", "technical"),
    ("Please cancel the renewal of my subscription.", "billing"),
    ("I cannot access my profile after changing my email.", "account"),
    ("Exporting a PDF produces an empty file.", "technical"),
]
dev_rows = [
    ("The amount on this invoice is incorrect.", "billing"),
    ("My password reset link has expired.", "account"),
    ("Clicking Save closes the application.", "technical"),
    ("I cannot sign in and also need a refund.", "billing"),
    ("My profile is inaccessible and the page shows an error.", "account"),
    ("Something is wrong. Can you help?", "technical"),
]

def make_examples(rows):
    return [
        dspy.Example(ticket=text, category=label).with_inputs("ticket")
        for text, label in rows
    ]

trainset = make_examples(train_rows)
devset = make_examples(dev_rows)
print("Program receives:", dict(devset[0].inputs()))
print("Reference answer:", devset[0].category)

# %% [markdown]
# Keep these roles separate: `trainset` supplies examples for the optional optimizer;
# `devset` lets us compare versions. Once you repeatedly inspect and tune against
# a dev set, it is no longer an untouched test. A real project needs a separate
# final test set and more representative data. These twelve examples teach the workflow.
#
# A metric is a normal Python function. Ours returns `True` for the exact reference
# label and `False` otherwise. The optional `trace` parameter lets an optimizer call
# the same metric with execution information; we do not need it for this task.
# [Reference: metrics](https://dspy.ai/learn/evaluation/metrics/).

# %% metric
def category_correct(example, prediction, trace=None):
    return prediction.category == example.category

# Verify the scoring rule without making any model calls.
reference = dspy.Example(category="billing")
assert category_correct(reference, dspy.Prediction(category="billing"))
assert not category_correct(reference, dspy.Prediction(category="technical"))
print("Metric gives credit to a match and rejects a mismatch.")

# %% evaluate
evaluate = dspy.Evaluate(
    devset=devset,
    metric=category_correct,
    num_threads=1,
    display_progress=True,
    display_table=False,
    max_errors=1,
)
baseline = evaluate(classify)  # MODEL CALLS: 6
print(f"Baseline accuracy: {baseline.score:.1f}%")

# %% [markdown]
# `baseline.score` is a percentage (0–100), whereas the per-example metric is
# Boolean (effectively 0 or 1). Inspect individual results as well as the average.
# Here one mistake moves the score by about 16.7 percentage points.

# %% inspect_evaluation
for example, prediction, correct in baseline.results:
    status = "OK" if correct else "MISS"
    print(f"{status:4} expected={example.category:9} predicted={prediction.category:9}")
    print("    ", example.ticket)

mistakes = [(ex, pred) for ex, pred, correct in baseline.results if not correct]
print("Mistakes:", len(mistakes))

# %% [markdown]
# **Exercise:** choose one result and explain why its reference label is correct
# under our policy. If every result is right, invent a harder ticket. Decide its
# reference label **before** asking the model. Add it to `dev_rows`, rerun the dataset
# and evaluator cells, and compare. This creates a new baseline on a changed dataset;
# do not compare its percentage directly to the old six-ticket score.
#
# **Your notes:** Hard ticket: … Reference label and reason: … Observed result: …
#
# ## 5. Stop here if you want the fundamentals — 5 minutes
#
# Without scrolling up, answer:
#
# 1. Which object describes the task? Which object executes it?
# 2. Does defining a module call a model? Does evaluating one?
# 3. Why is a type-valid prediction sometimes still incorrect?
# 4. Why mark only `ticket` as input on each labeled example?
# 5. Does perfect category accuracy establish that replies are good?
#
# **Your answers:** …
#
# <details><summary>Check your answers</summary>
# 1. A signature describes the task; a module executes it using a configured LM.
# 2. Definition makes no request. Evaluation calls the program on examples.
# 3. `account` is allowed but incorrect for a refund under our policy.
# 4. The target category belongs in scoring, not in the classifier's input.
# 5. No. Evaluate replies separately for their own requirements.
# </details>
#
# You now have the basic build → inspect → measure workflow. The remaining sections
# are optional. Agents, retrieval systems, and larger optimizers can wait.
#
# ## 6. Optional: a first optimizer — 20–30 minutes
#
# `BootstrapFewShot` runs a teacher program on training inputs and keeps successful
# traces as demonstrations, using your metric to decide success. With no separate
# teacher provided, it uses a copy of the student. We limit it to two bootstrapped
# demonstrations, no extra labeled demonstrations, and one pass over six examples.
#
# `compile` returns a program with demonstrations in its prompts. Here it neither
# rewrites your Python nor changes model weights. Different optimizers can adjust
# other things. Quality may improve, stay flat, or worsen; inspect and evaluate.
# [Reference: BootstrapFewShot](https://dspy.ai/api/optimizers/BootstrapFewShot/).
#
# To run this lab, change the switch below to `True`. Otherwise every cell in the
# section can run safely and will explain that it was skipped. Keep the classifier,
# metric, and dev set fixed for this before/after comparison.

# %% optimize
RUN_OPTIMIZATION = False  # Change to True when ready for the optional model calls.
optimized = None
optimized_result = None

if RUN_OPTIMIZATION:
    optimizer = dspy.BootstrapFewShot(
        metric=category_correct,
        max_bootstrapped_demos=2,
        max_labeled_demos=0,
        max_rounds=1,
        max_errors=1,
    )
    optimized = optimizer.compile(dspy.Predict(TicketCategory), trainset=trainset)
    print("Demonstrations collected:", len(optimized.demos))
    for demo in optimized.demos:
        print(dict(demo))
else:
    print("Optional optimization skipped. Set RUN_OPTIMIZATION=True above to try it.")

# %% optimize_evaluate
if optimized is not None:
    optimized_result = evaluate(optimized)  # MODEL CALLS: 6
    print(f"Baseline:  {baseline.score:.1f}%")
    print(f"Optimized: {optimized_result.score:.1f}%")
    print(f"Change:    {optimized_result.score - baseline.score:+.1f} percentage points")
    dspy.inspect_history(n=1)  # Find the new demonstrations in the prompt.
else:
    print("No optimized program yet; optional comparison skipped.")

# %% [markdown]
# **Think before continuing:** Did the instructions change? Did demonstrations
# appear? Did the score improve? If the baseline was already 100%, what could this
# dataset tell you about further gains? If no demonstrations were collected,
# inspect training predictions and the scoring rule.
#
# **Your notes:** Before: … After: … Evidence for keeping this version: …
#
# ### Save and restore program state
#
# This JSON stores program state such as instructions and demonstrations, not the
# model's weights or your Python class definitions. Restore into the same program
# structure and configure an LM in the new session. We save under `artifacts/`, which
# is ignored by Git. A rerun overwrites this tutorial's previous saved state.
# [Reference: saving and loading](https://dspy.ai/tutorials/saving/).

# %% save_reload
if optimized is not None:
    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)
    state_path = output_dir / "ticket_classifier.json"
    optimized.save(str(state_path), save_program=False)

    restored = dspy.Predict(TicketCategory)
    restored.load(str(state_path))
    print("Restored demonstrations:", len(restored.demos))
    print(restored(ticket=devset[0].ticket))  # MODEL CALL: 1 (may be cached)
    print("Saved:", state_path.resolve())
else:
    print("No optimized program yet; optional save/reload skipped.")

# %% [markdown]
# ## 7. Optional: your own controlled experiment — 20–40 minutes
#
# Compare `Predict` with `ChainOfThought` on the **same** dev set. First write a
# hypothesis. Only change the module; leave the signature, model, data, and metric alone.
# This cell defaults to skipping, so Run All does not enable extra experiments.

# %% experiment
RUN_EXPERIMENT = False
if RUN_EXPERIMENT:
    challenger = dspy.ChainOfThought(TicketCategory)
    challenger_result = evaluate(challenger)  # MODEL CALLS: len(devset)
    print(f"Predict: {baseline.score:.1f}% | ChainOfThought: {challenger_result.score:.1f}%")
    for example, prediction, correct in challenger_result.results:
        if not correct:
            print(example.ticket, "→", prediction.category)
else:
    print("Optional comparison skipped. Set RUN_EXPERIMENT=True to try it.")

# %% [markdown]
# **Your experiment log:**
#
# | Hypothesis | One change | Dataset size | Baseline | Challenger | What I learned |
# |---|---|---:|---:|---:|---|
# | … | … | … | … | … | … |
#
# If you have another 15 minutes, build a similar classifier for three kinds of
# messages in a hobby or project you know. Define the categories and overlap rule,
# write six labeled examples, implement a signature and metric, then evaluate.
# Success means you can explain each piece and its limits, not that you get 100%.
#
# ## Troubleshooting and next steps
#
# | Symptom | What to do |
# |---|---|
# | `No module named dspy` | Select this project's `.venv` kernel. See the install commands in README. |
# | `NameError` or stale behavior | Restart the kernel; rerun from the top. Recreate instances after changing classes. |
# | Authentication failure | Check the provider key in your kernel environment; restart to re-enter it if needed. |
# | Quota/rate-limit failure | Check API credit or limits, pause, and rerun the failed cell. |
# | Model unavailable | Choose a model your provider account can access and rerun configuration. |
# | Parsing error | Inspect history; check model output and signature. Restore the supplied example to isolate changes. |
# | Same answer after rerunning | Identical calls can hit cache. Change the input, or deliberately set `cache=False` in the LM cell and rerun it. |
# | No improvement after optimization | This is a valid result. Inspect examples and errors; more demonstrations are not automatically better. |
#
# Keep this reference handy: **signature defines the task; module runs it; examples
# and a metric measure it; an optimizer adjusts it using data.**
#
# Official documentation used for the lesson: [DSPy](https://dspy.ai/),
# [optimization overview](https://dspy.ai/learn/optimization/optimizers/), and
# [VS Code notebooks](https://code.visualstudio.com/docs/datascience/jupyter-notebooks).
# The local code was prepared against DSPy 3.4.1. Your model's outputs may differ.
