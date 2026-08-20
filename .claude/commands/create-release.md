Create a new release: generate release notes, create a GitHub release (which triggers PyPI publish), and verify the pipeline.

The user optionally provides a version as $ARGUMENTS (e.g., `0.4.0`). If not provided, determine the next version from the latest git tag and the nature of changes (patch for fixes/chores, minor for features, major for breaking changes).

## Steps

### 1. Determine version

- Find the latest git tag: `git tag --sort=-v:refname | head -1`
- If the user didn't provide a version, suggest one based on commits since the last tag.
- Confirm the version with the user before proceeding.

### 2. Generate release notes

- List commits since the last tag: `git log --oneline <last_tag>..HEAD`
- Categorize using conventional commit prefixes:
  - `feat` → **Features**
  - `fix` → **Bug Fixes**
  - `refactor` → **Refactoring**
  - `perf` → **Performance**
  - `docs` → **Documentation**
  - `test` → **Tests**
  - `build`, `ci`, `chore` → **Maintenance** (group together)
  - Breaking changes (any commit with `!` after type, e.g., `feat!:`) → **Breaking Changes** section at the top
- Skip: merge commits, commits that only touch `.claude/` or `docs/plans/`
- Write descriptions from the **user's perspective**, not the commit message verbatim.
- Omit empty sections. Group related commits into a single bullet point.
- Keep it concise — release notes are for users, not a git log dump.

### 3. Show the user the release notes for review

Present the full release notes in the body format below and ask the user to confirm before creating the release.

**Body format**:
```markdown
## What's New

### Features
- Brief description of feature ([#PR](url) if applicable)

### Bug Fixes
- Brief description of fix

**Full Changelog**: https://github.com/strands-rl/strands-sglang/compare/<last_tag>...v<new_version>
```

### 4. Create the GitHub release

Once the user confirms, create the release using `gh release create`. This automatically creates the git tag.

```bash
gh release create v<version> --title "v<version>" --notes "<body>"
```

Use a HEREDOC for the body to preserve formatting.

### 5. Verify the publish pipeline

The GitHub release triggers `.github/workflows/publish.yml` which:
1. Builds the package (hatch-vcs derives version from the git tag)
2. Publishes to PyPI via trusted publishing (OIDC)

After creating the release:
- Run `gh run list --limit 3` to confirm the "Publish" workflow was triggered.
- Watch the run: `gh run watch <run_id>` (timeout 2 minutes).
- Report the result to the user — if it failed, show the logs with `gh run view <run_id> --log-failed`.

## Important

- Always confirm version and release notes with the user before creating the release.
- The version is derived from the git tag by hatch-vcs — there is no version field in `pyproject.toml` to update.
- Do NOT modify `pyproject.toml`.
- Get the number right the first time: the `tags-are-immutable` ruleset blocks moving or
  deleting a tag, and PyPI refuses a second upload of the same filename.
