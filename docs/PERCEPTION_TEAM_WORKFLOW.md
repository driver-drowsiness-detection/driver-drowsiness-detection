# Two-person perception workflow

The shared test endpoint is `codex/perception-integration`. It contains merge
commits preserving both original contributions. All integration corrections use
the checkout's configured author. Original authors were not impersonated.
`main`, `perception`, `perception01`, and `cnn` retain their existing remote work.
The integration pull request targets `main`; review laptop evidence before merging.

## Ownership

| Role / existing branch | Owns | Demonstration |
| --- | --- | --- |
| Person 1 / perception | capture.py, camera mode, Jetson camera probe, capture tests | Startup, moving frames, failures, bounded waiting, quit/restart |
| Person 2 / perception01 | detection.py, crops.py, quality.py, landmark/crop tests | Saved-image eyes, face selection, crops, rejection reasons |
| Both | preview script, overlay, acceptance sessions | One command runs the complete pipeline |
| Arushi and reviewer | contracts.py and cross-branch interfaces | Same sample/time/track/side reaches EAR and CNN |

Person 1's earlier branch also contained landmarks. That history remains, but
future landmark changes have one owner (Person 2). Decide shared-file changes
together before editing. Commit messages describe real changes, not pretend authors.

## Get the final code on each laptop

Run from your own clone. Start with `git status --short`. If it lists work, commit
that work on its existing branch first, using exact file paths and an honest
message. Do not overwrite it or use a hard reset.

```text
git fetch origin
git switch --track origin/codex/perception-integration
git pull --ff-only origin codex/perception-integration
git rev-parse HEAD
```

If the local integration branch already exists, use `git switch
codex/perception-integration` instead of `--track`. Everyone tests the SAME commit
from `git rev-parse HEAD`. Branch names alone do not establish identical code.
Follow README for environments, model download and run commands.

## Continue the two existing work branches

After saving any local changes, Person 1:

```text
git fetch origin
git switch perception
git pull --ff-only origin perception
git merge --ff-only origin/codex/perception-integration
```

Person 2 uses the same commands with `perception01`. If the branch is not local
yet, create it with `git switch --track origin/perception` (or `perception01`).
These fast-forwards work while the old branch is still an ancestor of integration.
If somebody pushed newer commits and Git refuses, preserve them and ask the
reviewer to integrate the new commits; never force-push to make this work.

They can then push their updated branch, make small changes, run tests and commit:

```text
git push origin perception
git add src/marvel/perception/capture.py tests/test_capture_worker.py
git diff --cached
git commit -m "Fix capture timeout handling and verify worker cleanup"
git push origin perception
```

For Person 2, stage only the changed landmark/crop/quality files and relevant tests,
then use a message such as `Reject invalid eye coordinates before crop extraction`.
Do not make an empty example commit; these are examples for actual future changes.

Open a pull request from the contributor branch to `codex/perception-integration`
while the first integration PR is pending. The other person reviews it. Both fetch
and fast-forward to the updated shared integration commit before the next joint test.
After integration is merged into `main`, future PRs target `main`; start new small
branches from updated `main` rather than recycling completed work branches.

## Merge to main after review

Use the integration PR's normal merge-commit option to retain the original team
history. No history rewrite or forced push is needed. Everyone then runs:

```text
git fetch origin
git switch main
git pull --ff-only origin main
```

Local package environments, camera settings and `.task` assets are not committed.
Use ignored `configs/local-*.yaml` for a laptop's camera index or measured thresholds.
Keep pull requests small and include the exact command, commit, tests and failures.
