# Deployment notes (C4)

The repository runs in one Unix account for the real-data build.  For a real
deployment, separate the grader from the workers at the OS level:

```bash
sudo useradd --system --home /srv/compbio-claim-inflation --shell /usr/sbin/nologin cbi-grader
sudo chown -R root:cbi-grader /srv/compbio-claim-inflation
sudo chmod 700 /srv/compbio-claim-inflation/data/eval_restricted /srv/compbio-claim-inflation/secrets /srv/compbio-claim-inflation/logs/grader_internal
sudo chown -R cbi-grader /srv/compbio-claim-inflation/secrets /srv/compbio-claim-inflation/logs/grader_internal
sudo chmod 750 /srv/compbio-claim-inflation /srv/compbio-claim-inflation/data /srv/compbio-claim-inflation/logs
```

- Workers run as a different account with read access only to `data/processed`,
  `models`, the code, and their own artifact directory.
- The grader token lives in `/srv/compbio-claim-inflation/secrets/grader.token`, mode 0600, owned by
  `cbi-grader`; only the grader service receives `CBI_GRADER_TOKEN`.
- `cbi-grader.service`:

```ini
[Unit]
Description=CBI grader service
After=network.target

[Service]
User=cbi-grader
Group=cbi-grader
WorkingDirectory=/srv/compbio-claim-inflation
Environment=CBI_ROOT=/srv/compbio-claim-inflation
EnvironmentFile=/srv/compbio-claim-inflation/secrets/grader.env
ExecStart=/srv/compbio-claim-inflation/.venv/bin/python -m cbi.m6.service_cli \
  --root /srv/compbio-claim-inflation --detector /srv/compbio-claim-inflation/models/detector_v1 \
  /srv/compbio-claim-inflation/redteam_artifacts/round_01/w00
NoNewPrivileges=yes
ProtectSystem=strict
ReadWritePaths=/srv/compbio-claim-inflation/logs /srv/compbio-claim-inflation/data/eval_restricted
PrivateTmp=yes

[Install]
WantedBy=multi-user.target
```

For bulk grading, use the Python `GraderService` API from a grader-owned process
or a socket wrapper; do not expose the token to worker jobs.  In addition,
`cbi.m5.sandbox` masks eval, secrets, grader-internal logs, and sibling
artifacts with bubblewrap for every worker.
