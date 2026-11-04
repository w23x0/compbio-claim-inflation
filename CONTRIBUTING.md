# Contributing

## Development setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[test,plot,api]"
python3 -m pytest -q
```

The package uses a `src/` layout; run commands from the repository root so the
config lookup (`configs/mvp.yaml`) and relative data paths resolve.

## Rules

- **Never commit `data/`, `models/`, `logs/`, `redteam_artifacts/` or `secrets/`.**
  They are ignored; keep it that way. Credentials belong in an untracked env
  file (see `.env.example`).
- **No synthetic cohorts in the production path.** Tests may use small
  in-memory fixtures, but stages P0–P8 must fail closed without real data.
- **Fail closed on LLM batches.** A missing or malformed batch item is retried
  singly; it is never scored as zero or negative.
- **Keep the audit trail.** New stages write a machine-readable artifact plus a
  markdown summary under `reports/`.
- **Update `docs/status.md`** when an acceptance criterion changes state, and
  add an entry to `CHANGELOG.md` for user-visible changes.

## Pull requests

1. Add or update tests for behaviour changes (`tests/`).
2. Run `python3 -m pytest -q` and the affected `python3 -m cbi.cli <stage>`.
3. Describe verification in the PR: the command run and the observed result.
4. Do not mix refactors with result changes in one PR.

## Reporting results

Any number quoted from `reports/` must name the artifact file and the commit.
Results produced by an invalidated detector are labelled as such in
`docs/audit.md`; do not cite them.
