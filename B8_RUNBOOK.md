# B8.0 execution runbook

The B8.0 pilot runs as a resume-safe command-line experiment on a persistent
Linux host. It is deliberately not a long-lived Colab notebook: the earlier
Colab disconnections showed that a browser runtime is the wrong execution
boundary for multi-hour RTL-to-GDS flows.

## Before the first run

1. Clone OpenROAD-flow-scripts on the execution host and check out one explicit
   commit. Do not use a moving `master` checkout after results begin.
2. Use either a verified native/source build or a pinned ORFS container image.
   Record the immutable image digest, not only a tag.
3. Keep the ORFS checkout clean. The runner refuses to freeze a dirty checkout.
4. Mount or synchronize the canonical Drive experiment directory.
5. Confirm adequate free disk space. ORFS results stay in the ORFS checkout;
   compact manifests and run records are written to Drive after every flow.

Current official ORFS images have a reported CTS `SIGILL` problem on CPUs
without AVX-512. The runner therefore blocks `docker-shell` execution on a host
that does not advertise `avx512f`. Use a verified source-built image/native
build or an AVX-512 host; do not override this check just to obtain results.

## Plan the nine-flow smoke test

```bash
python scripts/run_b8_actionability.py plan \
  --stage smoke \
  --orfs-root /path/to/OpenROAD-flow-scripts \
  --executor docker-shell \
  --container-image openroad/orfs:<pinned-tag> \
  --container-digest sha256:<immutable-digest> \
  --threads 8
```

The command prints the protocol hash. All later commands should use it:

```bash
python scripts/run_b8_actionability.py run \
  --stage smoke \
  --protocol-hash <protocol-hash> \
  --max-runs 1
```

Run one flow first. Inspect its log and artifacts, then resume the remaining
eight by removing `--max-runs`. Completed runs are skipped. Failed runs remain
frozen until explicitly retried with `--rerun-failed`.

```bash
python scripts/run_b8_actionability.py status \
  --stage smoke \
  --protocol-hash <protocol-hash>
```

Do not start the 126-flow matrix merely because commands execute. The smoke
test must produce nine completed runs, PDN checkpoints, final GDS files,
metadata, deterministic action Tcl files, exact full-deck KLayout RDB reports,
per-rule counts, logs, and completion markers. Scientific gate analysis begins
only after these real outputs confirm the pinned ORFS file contract.

## Registered action implementation

`PLACE_DENSITY_LB_ADDON` is passed directly to ORFS. Routing adjustment is not
passed through the nonexistent `ROUTING_LAYER_ADJUSTMENT` variable. The runner
generates a per-run `FASTROUTE_TCL` containing the supported
`set_global_routing_layer_adjustment` command and the paired
`set_global_routing_random -seed` command. The generated Tcl is retained inside
the pinned ORFS checkout and its action values are recorded in the manifest.
