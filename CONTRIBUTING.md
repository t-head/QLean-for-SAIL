# Contributing to QLean

Contributions are welcome. Search existing issues and pull requests before opening a new one. For substantial changes, open an issue to discuss the approach with maintainers before you start.

## Before You Start

- **Open an issue first** for new features, behavioral changes, and other substantial work.
- **Check existing issues and pull requests** to avoid duplicate work.
- **License**: QLean-owned code is licensed under [Apache-2.0](LICENSE). Submit only code you have the right to contribute, and preserve applicable copyright and third-party notices. See [NOTICE](NOTICE) for third-party components.

## Reporting Issues

Use the [issue templates](.github/ISSUE_TEMPLATE) to report bugs or propose features. Include the environment, the command or recipe you used, and relevant logs. Remove credentials, private endpoints, personal information, and confidential model or dataset details before submitting.

## Development Workflow

1. Create a topic branch from the latest `main`. Contributors without write access work in a fork; contributors with write access create the branch in the repository.

   ```bash
   git fetch origin
   git switch -c fix/scale-rounding origin/main
   ```

   Branch names use lowercase words separated by hyphens. Match the branch type to the change: `feature/`, `fix/`, `hotfix/`, `refactor/`, or `docs/`.

2. Prepare the environment described in the [installation guide](README.md#installation):

   ```bash
   python -m pip install -e ".[dev]"
   qlean --help
   ```

3. Build a wheel when you change packaging or need an installable artifact:

   ```bash
   python -m pip wheel . --no-deps --wheel-dir dist
   ```

4. Run the tests relevant to your change. For example:

   ```bash
   python -m pytest tests/cli/test_unit_format_arguments.py -m cpu -v -ra
   ```

   Before opening a pull request, run the full CPU suite. Choose a new output directory outside the repository for each run:

   ```bash
   python tests/run_checks.py --suite cpu --output /tmp/qlean-cpu
   ```

   GPTQ tests carry the `gptq` marker and are excluded from the `cpu` and `ppu` suites, because they need the optional extra's narrower dependencies. Run them in a separate environment prepared with `pip install -e ".[gptq,dev]" -c ci/gptq-constraints.txt`:

   ```bash
   python tests/run_checks.py --suite gptq --output /tmp/qlean-gptq
   ```

   The `gptq` suite asserts the installed versions against `ci/gptq-constraints.txt`. Tests marked `optional` skip by design when native or accelerator dependencies are absent; they are excluded from every audited suite and can be run directly with `python -m pytest tests -m optional`.

   If you cannot run hardware-dependent checks, state what remains untested in the pull request and ask a maintainer for help.

5. Keep the branch in sync with `main` while you work:

   ```bash
   git fetch origin
   git rebase origin/main
   git push --force-with-lease
   ```

   Use `--force-with-lease` after a rewrite, never a bare `--force`. Do not use merge commits in a topic branch.

## Commit Messages

Commits follow [Conventional Commits](https://www.conventionalcommits.org/):

```text
<type>(<scope>): <subject>
```

`<type>` is one of `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, or `perf`. The scope is optional. Write the subject in the imperative mood, start it in lowercase, omit the trailing period, and keep it within 72 characters. Do not merge messages such as `update`, `fix bug`, or `wip` into `main`.

```text
fix(mxfp4): correct scale rounding for odd group sizes
docs: update the dequantization example
```

Sign off every commit to certify the Developer Certificate of Origin:

```bash
git commit -s -m "fix(mxfp4): correct scale rounding for odd group sizes"
```

The `Signed-off-by` trailer must match the commit author. Reference an issue when the change resolves or relates to one: use `Refs #123` for related or partial work and `Closes #123` only when the change resolves the issue.

## Pull Requests

Target `main` in the official QLean repository, not your fork. Do not push directly to protected branches.

- Title the pull request with the same Conventional Commits format as a commit, for example `fix(mxfp4): correct scale rounding for odd group sizes`.
- Describe the change:
  - **Summary**: what changed and why.
  - **Related issue**: `Refs #123` or `Closes #123`.
  - **Change type**: feature, bug fix, refactor, documentation, or other.
  - **Validation**: tests run, hardware used, and any results not obtained.
  - **Risk and compatibility**: breaking changes, migration steps, or configuration changes.
- Reference at least one **open issue**. Closed issues and references inside HTML comments do not count. If you link an issue through the Development sidebar, edit the pull request description or re-run the issue-link check to refresh its result.
- Keep the pull request focused on one change. Split independently reviewable changes into separate pull requests; a large change should include a design description or be split before review.
- Address review feedback, resolve failed checks, and obtain approval from at least one non-author reviewer before merging.
- Pull requests are merged with **Squash and merge**. The pull request title becomes the commit subject in `main`, so keep it accurate and self-contained.

## Code Review

Reviewers evaluate correctness, edge cases, readability, and test coverage. Use these prefixes for review comments:

- `suggestion:` for an optional improvement.
- `must:` for a blocking issue that requires a change.
- `question:` when the author needs to explain something.

Resolve every conversation before merging. When opinions differ, ask the relevant domain owner or a maintainer to decide.

## Licensing and Security

QLean-owned code is licensed under [Apache-2.0](LICENSE). Submit only code you have the right to contribute, and preserve applicable copyright and license notices, including those for third-party code listed in [NOTICE](NOTICE).

Do not include credentials, private endpoints, confidential data, or local build artifacts in commits, issues, or attachments.

For security-sensitive reports, use the repository's private vulnerability reporting option if available. Otherwise, ask maintainers for a private reporting channel without disclosing vulnerability details publicly.
