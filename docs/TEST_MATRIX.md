# Acceptance test matrix

Run `python -m pytest tests/ -v`.

| Category | Case | Expected |
|---|---|---|
| Normal 1 | Seed 1, easy generated map | CSP solvable; physics-valid search route |
| Normal 2 | Seed 2, mild generated map | CSP solvable; physics-valid search route |
| Normal 3 | Seed 3, medium map | CSP solvable; physics-valid search route |
| Normal 4 | Seed 4, difficult map | CSP solvable; physics-valid search route |
| Normal 5 | Seed 5, hardest target | CSP solvable; physics-valid search route |
| Edge 1 | Player starts at/after goal | BFS solution of length 0 |
| Edge 2 | Horizontal obstacle against player | Body cannot pass; jump can lift off |
| Edge 3 | Jump held while airborne | No repeated midair upward impulse |
| Impossible 1 | Ground gap too long to jump | All routes fail with explicit `No solution` |
| Impossible 2 | CSP start forced to illegal gap template | Backtracking returns explicit `No solution` |

Additional tests check all four algorithms on a known-answer flat graph, optimal BFS/A* cost, switchable FC/AC-3 (all four combinations), seed reproducibility, higher difficulty changing chunks, neural loss decreasing, saved NN inference, telemetry updates, adaptive candidate validity and full headless integration. When a search hits its explicit budget rather than exhausts the frontier, it returns `expansion limit` (not a false proof of impossibility).
