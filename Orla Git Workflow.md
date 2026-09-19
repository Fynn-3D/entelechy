# Orla Git Workflow

Keep `main` stable. Develop on focused `feat/*`, `fix/*`, `refactor/*`, `docs/*`,
`test/*`, or `chore/*` branches. Preserve user changes and inspect the working tree,
recent commits, remotes, and upstream before selecting a base.

## Select the base

For independent work with an integrated upstream, update `main` by fast-forward
only, then create a branch. Check that the working tree can switch safely first:

```sh
git switch main
git pull --ff-only
git switch -c feat/change-name
```

These commands assume `main` has a configured upstream. If no remote/upstream
exists, skip pull and push. Local development and local commits remain valid.

For continued work depending on unmerged commits, continue the existing feature
branch or create a focused branch from its HEAD. Do not start from an older
`main` and lose the dependency chain. Record the dependency in the PR description
when review becomes available. Do not merge or rewrite history merely to obtain
a new branch base.

If user changes prevent a safe switch, preserve them and isolate the work as
needed. Never discard or automatically stash unrelated work to clean the tree.

## Validate and commit

Inspect the diff and run checks appropriate to the change. Runtime changes need
relevant tests, formatting, and build checks; documentation-only changes need
link/reference and content checks. Record actual verification, including limits.

Stage only intended files. Use small logical commits with English messages such
as `feat: add kernel registry` or `docs: clarify external agent boundaries`.
Update HANDOFF only with the next action and context that Git cannot supply.

## Remote review and integration

When a remote is configured and publication is authorized, push the feature
branch and open a PR. Local-commit authorization alone does not authorize pushing,
merging, or releasing. Existing explicit authorization remains valid; do not ask
for it again for the same scope.

Describe the problem, resulting behavior, validation, and any changed architecture
boundary. For stacked work, identify the dependency and use the appropriate PR
base; after its integration, reconcile the dependent branch before merging it.

Merge only after required checks and review pass. Prefer squash merge for an
independent logical change; account for dependent branches before squashing.
Delete branches only after confirming their work is integrated and no dependent
work needs them. Never force-push or rewrite history without explicit authorization.

Resolve conflicts by understanding both intended behaviors, then verify the
combined result. Never select one side blindly.

## Releases

Use semantic versioning and document intentional compatibility changes. Tag and
publish only within release authorization, after required validation. A local
commit or completed PR is not automatically a release.
