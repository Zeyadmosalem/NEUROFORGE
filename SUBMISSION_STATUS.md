# NEUROFORGE V2 — Status

- Four simultaneous autonomous search-driven agents with separate procedural levels and online histories: IMPLEMENTED, headlessly tested.
- Manual graphical platformer and switch into/out of arena: CODE COMPLETE; desktop Pygame smoke test outstanding.
- 26+ curated chunk variants, including airborne spikes and correct swept collision check: IMPLEMENTED.
- Frozen 12→32→16→1 neural model trained once on 6,500 explicitly synthetic rows: DONE.
- New-only online adapter updates exactly after every 10 fresh outcomes: TESTED.
- Online regularization via 12% feature dropout and scheduled partial loss of learned calibration: TESTED.
- Forgetting after every 3 updates by default, 35% of adapter feature weights erased: TESTED; pretrained NN unaffected.
- Relearning based on subsequent fresh mini-batches: VERIFIED MECHANICALLY; no guarantee this improves predictive accuracy.
- All search methods, CSP variants, impossible cases and shared deterministic physics: TESTED.
- Five-seed standard experiments re-run under updated code. One-seed exploratory forgetting experiment present; NOT evidence of causal improvement.
- Old report and presentation reflect V1 architecture, not updated to V2. Team contributions and Git history need real team members.

## V3 — Reactive dynamic world update (November 2026)

- Dynamic logical physics: moving platforms, moving enemies, timed gates with search-aware phase.
- Surprise engine: randomized mid-attempt hazards, patrols, gate/platform timing shifts, bridges and coins. Each accepted change is safe and feasible under the agent's own algorithm and forces a replan from current state.
- The arena GUI renders actual dynamic object positions, shows event/replan counters and has `U` and `E` controls. Manual GUI renders dynamic objects too.
- Surprise events are recorded in per-attempt telemetry and persist to JSONL if persistence is enabled. Base NN remains frozen; per-agent online adapter still updates only every ten new completed attempts.
- Focused tests cover reproducible surprises, safe replanning, four-algorithm validity, timed-gate safe phases, and moving-platform riding.
- Five-seed supplementary reactivity CSVs and comparison plot in `results/`.
- Local desktop Pygame execution was NOT available for runtime GUI validation; offline preview is generated from real state, not a GUI screenshot.
- Existing V1 PDF/DOCX/PPTX assets were **not** regenerated for V3 and should not be submitted as-is without updating their claims.
