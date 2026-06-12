Review a pull request against this project's conventions and code quality standards.

The user provides a PR number or URL as $ARGUMENTS. If not provided, ask.

## Steps

1. **Fetch the PR** using `gh pr view <number> --json title,body,additions,deletions,files` and `gh pr diff <number>`.

2. **Review against these criteria**, in order of importance:

### Correctness & Design
- Does the change correctly implement what it claims?
- Are there edge cases or error conditions not handled?
- Is the architecture aligned with the project? (SGLangModel, SGLangClient, Rollout, ToolParser patterns)
- Are class/function names consistent with existing conventions?

### Token Tracking (TITO) Integrity
- Do changes to token handling preserve the prompt/response segment structure?
- Is `loss_mask` correctly maintained (0 for prompt, 1 for response)?
- Could changes cause tokenization drift between prompt and response segments?

### Over-engineering
- Are there unnecessary abstractions?
- Is the code proportional to the problem?
- Are there unnecessary files? (plan docs, task files, design docs should NOT be in the PR)
- Is the test-to-source ratio reasonable?

### Code Style
- License headers on all `.py` files
- Line length <= 120 (ruff enforced)
- No leftover debug files or personal paths
- Imports follow project conventions (TYPE_CHECKING guard for type-only imports)
- Conventional commit messages

### Error Handling
- New exceptions should fit the hierarchy (`SGLangClientError` base)
- HTTP error classification belongs in `_classify_http_error()`, not scattered
- Non-retryable vs retryable errors correctly categorized

### Dependency Hygiene
- New pip dependencies should be justified
- Heavy dependencies should be optional, not core

3. **Output a structured review** with:
   - A one-paragraph summary of what the PR does and overall assessment
   - A categorized list of issues (Critical / Should Fix / Nit)
   - Specific file:line references where possible
   - Suggested alternatives for major issues

Do NOT suggest changes for things that are fine. Focus on actionable feedback.
