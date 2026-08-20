Audit a package's docstrings and comments against the style rules in AGENTS.md, and fix what fails.

Target is $ARGUMENTS — a package or file path under `src/strands_sglang/` (e.g. `eval`,
`environments/harbor`, `sglang.py`). If not provided, ask which one; never audit
everything at once.

## Read the files. Do not grep for candidates.

A script cannot tell a `# ---- Task ----` section divider from an explanatory comment, or
judge whether a private helper names a domain concept. Open each file in the target and
read it. Counting tools are for reporting scale, not for deciding.

## The decisions, in the order they resolve

**1. `Args:` is all-or-nothing.** `D417` rejects a partial section, so the question is
never "does this parameter need a line" but "do this function's parameters need
explaining at all". Default no.

Delete the whole section when every line restates its annotation:

```python
client: SGLangClient            # signature
    client: `SGLangClient` for HTTP communication with the server.   # the same fact
```

Keep it when a parameter's meaning isn't in its type: resolution order, ownership, what
`None` falls back to, units, what happens when two arguments disagree, or a name that
implies less than it does (`env_hook_path: str` is really a dotted path to a callable
returning an `AsyncEnvFactory`).

**2. `Returns:`/`Yields:` earn their place against the annotation.**
`-> tuple[list[ToolResultContent], Literal["success", "error"]]` needs no "a tuple of
(content, status)". A `-> str` that is really a JSON envelope does. The poorer the type,
the more the section is worth.

**3. `Notes:` is a footnote; the body is the explanation.** Delete the sentence and see
what breaks. If the function stops making sense it was body text — why this method
exists, which of two paths to take, what an empty return means. If all that's lost is a
warning, it stays a footnote: thread-safety, a teardown contract, an upstream quirk, a
deliberate divergence from a reference implementation.

One `Notes:` holding three unrelated facts means none of them got classified. Split it.

**4. A docstring that restates its identifier goes.** `"""Return the tool name."""` on
`tool_name` says nothing; `D1` is unselected precisely so it can be absent.

**5. Comments state the why, on the line they explain.** Cut the ones narrating the next
statement ("Return the sample with the aborted flag set" above `sample = EvalSample(...)`).
Keep the ones a reader would otherwise undo ("Dropped by default: a full token trajectory
per sample bloats results.jsonl").

**6. Class-level `Args:` describing `__init__` parameters moves onto `__init__`.** Keep
the content, relocate it to where the parameters are.

## Exempt

- `@tool` method docstrings — agent-facing specs, the model reads them
- Section dividers (`# ---- Reward ----`)
- Pydantic `Field(description=...)` — serialized into task records
- Module docstrings: there are none, and none should be added (`D100`/`D104` unselected)

## Verify the claims you make

Any docstring asserting a fact about the code is a claim to check while you're there.
Past audits found three that were wrong, all of them silent:

- `requires_env` documented `EnvironmentError`, the code raises `OSError`
- `with_timeout` documented `_TimeoutInterrupt`, the class is `TimeoutInterrupt`
- `log_rollout_metrics` said `generate()` must set `sample.metrics`; it reads
  `sample.result.metrics`, so following the docstring would not have made it work

Grep the identifier. Check the exception type. Read the attribute the prose names.

## Finish

1. `.venv/bin/pre-commit run --all-files` — must be clean. `D205` (blank line after
   summary) and `D415` (summary ends with a period) are the two that bite when reflowing.
2. Report per file: what changed and why, plus anything you deliberately left. State the
   line delta.
3. Commit per package, not per file. The commit message names the defect classes found,
   and calls out any factual error separately — those are the part worth reviewing.

## Don't

- Rewrite working prose for taste. Fix the disease, keep the healthy tissue.
- Leave a file half-done. Two styles side by side is worse than either.
- Add a docstring that wasn't there because the function looks bare.
