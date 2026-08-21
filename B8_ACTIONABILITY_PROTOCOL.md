# B8.0 OpenROAD actionability protocol

Status: **preregistered plan; no B8.0 result has been inspected**.

## Decision this phase must make

B7.2 showed that the CNN is neither accurate enough nor faster than direct
KLayout DRC. B8.0 does not repair that detector. It asks a different question:

> Do existing OpenROAD actions create stable, design-dependent outcome
> differences that can be predicted before an expensive full flow?

The model, if later justified, will select an existing configuration. It will
not modify OpenROAD or KLayout internals. OpenROAD executes the selected action,
and exact post-route verification remains the acceptance authority.

OpenROAD already provides
[AutoTuner](https://openroad-flow-scripts.readthedocs.io/en/latest/user/InstructionsForAutoTuner.html),
so "automatic tuning" is not a novel contribution. The proposed contribution
is an amortized cross-design selector that uses experience from previous design
families to recommend an action for an unseen family with zero or one online
full-flow trial. AutoTuner must be compared under the same online trial budget.

## Frozen decision checkpoint

Capture inputs after floorplan/PDN generation and before global placement.
Every registered action is still legal at this checkpoint. The selector may use
only information available in that snapshot; it may not use placement, routing,
DRC, timing, or runtime outcomes produced after the action is applied.

The initial checkpoint features are:

- die/core geometry, utilization, macro/blockage and pin maps;
- cell/net/pin counts and netlist connectivity summaries;
- timing constraints and synthesis/floorplan estimates available at checkpoint;
- normalized RUDY or other deterministic pre-placement demand maps; and
- platform/tool identifiers and the candidate action vector.

If the action signal exists but these inputs are not predictive, one later
checkpoint may be registered in a new PR. Its action set must be restricted to
controls that remain legal at that time. Results cannot be used to move the
checkpoint silently.

## Reproducibility contract

- Platform: `sky130hd`.
- Designs: the seven official OpenROAD-flow-scripts families `aes`,
  `chameleon`, `gcd`, `ibex`, `jpeg`, `microwatt`, and `riscv32i`.
- Before the first result, pin the OpenROAD-flow-scripts commit, container
  digest, PDK/deck revisions, host CPU/RAM, thread count, and timeout.
- Use two paired flow seeds, 42 and 43. Record the resolved `OR_SEED`,
  `GRT_SEED`, and every effective flow variable. Seeds are nuisance
  repetitions, not actions.
- Run every flow in a fresh subprocess. A run is reusable only when its input,
  action, seed, tool, and output hashes match and its completion marker exists.
- Persist the manifest and metrics after each run. Never require all runs to
  remain in notebook memory.
- Primary exact outcome: the repository's pinned full KLayout post-route deck,
  reported both as total DRC count and count by rule. Also retain OpenROAD
  detailed-route violations as a diagnostic, not a replacement for KLayout.

The official design set is visible in the
[OpenROAD-flow-scripts `sky130hd` design tree](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts/tree/master/flow/designs/sky130hd).

## Frozen action space

Use the Cartesian product below. The `0.00 / 0.50` point is the registered
default-like reference. Record the final resolved `PLACE_DENSITY` because its
base value can be design-specific.

| Action | `PLACE_DENSITY_LB_ADDON` | FastRoute layer adjustment |
|---|---:|---:|
| A00 | 0.00 | 0.20 |
| A01 | 0.00 | 0.35 |
| A02 | 0.00 | 0.50 |
| A10 | 0.05 | 0.20 |
| A11 | 0.05 | 0.35 |
| A12 | 0.05 | 0.50 |
| A20 | 0.10 | 0.20 |
| A21 | 0.10 | 0.35 |
| A22 | 0.10 | 0.50 |

Do not add padding, macro-halo, routing-layer-range, or detailed-route knobs
until this gate passes. A larger action space can manufacture search headroom
without proving a learnable cross-design choice.

The routing value is implemented by generating a per-run `FASTROUTE_TCL` with
`set_global_routing_layer_adjustment` over the registered signal-layer range.
It is not passed as a `ROUTING_LAYER_ADJUSTMENT` environment variable, because
that is not a current ORFS flow variable. This is the same underlying control
that AutoTuner exposes as `_FR_LAYER_ADJUST`. The generated Tcl also freezes
the paired global-routing seed with `set_global_routing_random -seed`.

## Execution stages

### B8.0a: harness smoke test

Run `gcd`, `aes`, and `ibex` with A02, A11, and A20 at seed 42: nine full
flows. This stage checks container setup, checkpoint capture, exact-deck
execution, metric extraction, timeout/failure handling, hashing, and resume.
It has no scientific acceptance claim.

### B8.0b: actionability matrix

After the smoke test passes, execute seven designs by nine actions by two seeds:
126 full flows. Predeclare the run order and interleave actions/seeds to avoid a
time-ordered machine-temperature or load bias.

For every run record:

- success/failure stage and reason;
- exact KLayout DRC total, count by rule, and DRC-clean status;
- OpenROAD detailed-route violations;
- total and per-stage wall time plus peak memory;
- wirelength, via count, WNS, TNS, power/area proxies, and routing iterations;
- checkpoint features and hashes; and
- design, action, seed, resolved variables, tool/container/deck hashes, host,
  and command.

## Registered comparison rule

An action is PPA-feasible relative to the default-like A02 run at the same
design and seed when:

- wirelength is no more than 1% worse;
- WNS degradation is no greater than `max(0.05 ns, 5% of |default WNS|)`; and
- TNS degradation is no greater than `max(0.10 ns, 5% of |default TNS|)`.

Rank actions lexicographically by:

1. successful and PPA-feasible before failed/infeasible;
2. lower exact KLayout DRC count; and
3. lower total runtime when exact DRC count ties.

An action difference is **material** if it changes failure/PPA feasibility,
reduces nonzero exact DRC by at least 20% and at least one violation, or reduces
runtime by at least 20% at identical exact DRC without breaking PPA feasibility.
The per-family oracle is the best action after aggregating the paired seeds.
Normalized rank regret is zero for the oracle and one for the worst action.

## Go/no-go gates

Continue to a larger B8.1 dataset only when all five gates pass:

1. **Action effect:** at least four of seven families have a material action
   difference.
2. **Winner diversity:** no single fixed action is the oracle on five or more
   families.
3. **Oracle headroom:** the per-family oracle materially beats the best fixed
   action on at least three families and reduces its median normalized rank
   regret by at least 20%.
4. **Seed stability:** every family counted by gate 1 shows the same material
   direction in both seeds, or its paired mean effect is at least twice the
   absolute inter-seed variation.
5. **Pre-action learnability:** a leave-one-family-out transparent ranker using
   only checkpoint features reduces mean normalized rank regret by at least 20%
   versus the best fixed action, and its top two recommendations include the
   oracle on at least five of seven held-out families.

The seven-family result is a feasibility gate, not a publication-level learned
result. If gates 1–3 fail, stop the controller track: there is no useful action-
selection problem in this space. If only gate 4 or 5 fails, either stop or
preregister one later checkpoint/restricted action experiment. Do not train a
large model on a failed pilot.

## Path after a passing pilot

1. Expand to at least 20 independent, generator-disjoint design families.
2. Freeze family-disjoint train/development/final-holdout splits; never split
   seeds or action runs from one family across partitions.
3. Establish best-fixed, manual/bisection, random/grid, and official AutoTuner
   baselines before selecting a large model.
4. Train the smallest action-conditioned ranker that improves family-disjoint
   regret; add spatial CNN/graph fusion only after ablations justify it.
5. Compare online full-flow trials and turnaround under equal budgets. Also
   charge the learned method for offline trajectory generation and publish its
   amortization break-even point.
6. Use calibrated uncertainty to fall back to default or AutoTuner, then open
   the B9 holdout once.

## Canonical artifacts

Persistent runs belong below:

```text
/content/drive/MyDrive/ADVLSI2 2026 Project/
└── experiments/
    └── B8_action_control/
        └── b8_0_actionability/
            └── <protocol hash>/
                ├── manifest.json
                ├── smoke/
                ├── matrix/
                └── summary.json
```

The Git PR must contain the compact protocol, manifest, aggregate/per-family
tables, gate decisions, failure inventory, and plotting script. Large flow
directories remain in Drive.
