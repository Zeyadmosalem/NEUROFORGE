# Research basis and engineering decisions

1. **Experience-driven procedural content generation:** Yannakakis, G. N. & Togelius, J. (2011). *Experience-Driven Procedural Content Generation*. IEEE Transactions on Affective Computing, 2(3), 147–161. https://researcher.itu.dk/p/en/research-outputs/experience-driven-procedural-content-generation — motivation for player-guided generation.
2. **Constraint-based platformer design:** Smith, G., Whitehead, J., & Mateas, M. (2010). *Tanagra: An Intelligent Level Design Assistant for 2D Platformers*. AAAI AIIDE. DOI https://doi.org/10.1609/aiide.v6i1.12379 — constraint/geometry and reachability inspiration.
3. **Video Game Level Corpus:** Summerville et al. (2016). *The VGLC: The Video Game Level Corpus*. https://github.com/TheVGLC/TheVGLC — optional external real level geometry download, not gameplay outcome labels.
4. **Pygame Community Edition:** https://github.com/pygame-community/pygame-ce — SDL-based cross-platform game framework used for rendering and event handling; original player art code-drawn.
5. **Kenney Pixel Platformer:** https://kenney.nl/assets/pixel-platformer — optional CC0 upgrade for visuals; *assets NOT included* in this archive.
6. **AIMA Python:** https://github.com/aimacode/aima-python — reference for well-known search and constraint-solving concepts; algorithm code in this repository is original.

## Key design decisions

- Use original, low-resolution pixel-art inspired platformer instead of copyrighted Nintendo sprites.
- Use deterministic, discrete tile motion shared by search and gameplay. It is a simplified simulator, not real Mario physics.
- For fair comparison, all search algorithms receive the exact same graph, start, goal, state transitions and action costs.
- Forward Checking and AC-3 prune local binary adjacency constraints only; global budget is checked in recursive Backtracking; search checks physical reachability after CSP.
- Train NN on explicitly labelled **simulated** outcomes. No claims of validated human skill modelling.
- Report comparisons against Logistic Regression even if baseline does better. This is the current measured outcome.

## Future research extensions (not implemented)

- Consented human-subject evaluation of prediction calibration and win-rate targets.
- Multi-frame continuous physics + velocity/acceleration with refined admissible A* heuristic.
- Dynamic enemies requiring time-based planning state.
- Learned chunk grammars trained from VGLC geometry.
- Interactive level editor showing AC-3 propagation state per assignment.
