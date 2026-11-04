# Security policy

## Credentials

This repository never ships credentials. Provider keys, the NCBI key and the
grader token are read from environment variables (`CBI_LLM_*`, `OPENAI_*`,
`NCBI_API_KEY`, `CBI_GRADER_TOKEN`); `.env.example` documents them and
`secrets/` is git-ignored.

If a key was ever written into a tracked file, treat it as compromised, rotate
it, and remove it from history (not just from the working tree).

## Isolation model

Red-team workers run in a bubblewrap sandbox that masks `data/eval_restricted`,
`secrets/`, grader-internal logs and sibling artifacts. The grader runs as a
separate OS account in a real deployment (`docs/deployment.md`). Reported
sandbox escapes or grader-token leaks are security bugs.

## Reporting

Open a private security advisory on GitHub, or contact the maintainers directly
if no advisory channel is configured. Please do not file a public issue for an
unfixed credential leak or sandbox escape.
