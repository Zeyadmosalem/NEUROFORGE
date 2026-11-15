"""NEUROFORGE manual mode: an original neon platformer built with Pygame CE.

ALL gameplay movement is resolved by game.physics.step, the same as search.
No imported art: every shape is drawn in code; shared widgets live in game.ui.
Returns 'arena' (V), 'title' (Esc) or None (window closed).
"""
from __future__ import annotations
import threading
from math import sin, cos, log10, pi
from random import Random
from time import monotonic
from game.physics import step
from search.algorithms import solve
from csp.solver import ChunkCSP
from integration.adaptive import AdaptiveController, fallback_predict
from game import ui
from game.ui import WIDTH, HEIGHT, PALETTE as P, ALGO_COLORS, ALGO_LABELS

TILE = 48
TOP = 96            # screen y of level row 0; the 12 rows end at y=672
TICK = 0.105        # seconds per logical physics step
LAB_W = 410         # width of the AI Lab dashboard
ALGOS = ('bfs', 'dfs', 'astar', 'greedy')
CSP_MODES = (('BT', False, False), ('FC', True, False), ('AC-3', False, True), ('FC+AC-3', True, True))


class LabJob:
    """Solves the current level with all four searches off the render thread."""

    def __init__(self, level, first):
        self.level = level
        self.results = {}
        self.csp = None
        order = [first] + [a for a in ALGOS if a != first]
        threading.Thread(target=self._run, args=(order,), daemon=True, name='ai-lab').start()

    def _run(self, order):
        for algo in order:
            self.results[algo] = solve(self.level, algo, max_expansions=18000)
        n = len(self.level.chunks)
        self.csp = [(name, ChunkCSP(n, self.level.difficulty, self.level.seed).solve(fc, ac).stats)
                    for name, fc, ac in CSP_MODES]


def build_tiles(pg, level):
    """Static geometry of one level, drawn once and scrolled as a single blit."""
    rows = level.height
    layer = pg.Surface((level.width * TILE, (rows + 1) * TILE), pg.SRCALPHA)
    coins = []
    for x in range(level.width):
        sx = x * TILE
        for y in range(rows):
            tile = level.tile(x, y)
            sy = y * TILE
            if tile == '#':
                pg.draw.rect(layer, P['tile'], (sx, sy, TILE, TILE))
                if (x * 7 + y * 3) % 4 == 0:
                    pg.draw.line(layer, P['tile_line'], (sx + 8, sy + TILE - 10), (sx + TILE - 12, sy + TILE - 10), 2)
                if (x * 5 + y) % 6 == 0:
                    pg.draw.circle(layer, P['tile_line'], (sx + 14, sy + 22), 2)
                if level.tile(x - 1, y) != '#':
                    pg.draw.line(layer, (26, 26, 56), (sx, sy), (sx, sy + TILE - 1), 3)
                if level.tile(x + 1, y) != '#':
                    pg.draw.line(layer, (26, 26, 56), (sx + TILE - 2, sy), (sx + TILE - 2, sy + TILE - 1), 3)
                if not level.solid(x, y - 1):
                    pg.draw.rect(layer, P['tile_top'], (sx, sy, TILE, 9), border_radius=3)
                    pg.draw.rect(layer, P['tile_edge'], (sx, sy + 9, TILE, 5))
                    pg.draw.line(layer, (190, 255, 236), (sx + 4, sy + 2), (sx + TILE - 6, sy + 2), 1)
                if y == rows - 1:
                    pg.draw.rect(layer, P['tile'], (sx, sy + TILE, TILE, TILE))
            elif tile == '^':
                for k in range(2):
                    pts = [(sx + k * TILE // 2 + 3, sy + TILE), (sx + k * TILE // 2 + TILE // 4, sy + 10),
                           (sx + (k + 1) * TILE // 2 - 3, sy + TILE)]
                    pg.draw.polygon(layer, P['spike'], pts)
                    pg.draw.line(layer, P['spike_hi'], pts[0], pts[1], 2)
            elif tile == 'o':
                coins.append((x, y))
    return layer, coins


def run(seed=42, persist=True):
    try:
        import pygame as pg
    except ImportError as ex:
        raise RuntimeError('Install GUI dependency with: pip install pygame-ce==2.5.8') from ex
    pg.init()
    try:
        screen = ui.open_window(pg, 'NEUROFORGE  //  Adaptive AI Platformer')
        clock = pg.time.Clock()
        kit = ui.Kit(pg)
        backdrop = ui.Backdrop(pg, seed=seed, horizon=TOP + 10 * TILE)
        particles = ui.Particles(seed)
        toasts = ui.Toasts()
        rng = Random(seed)
        controller = AdaptiveController(seed=seed, owner='human', persist=persist)
        chosen = controller.next_level(n_candidates=5)
        level = chosen.result.level
        built_with = (True, True)
        state = level.start; prev = state
        jumps = 0; landings = 0; attempt_start = monotonic(); deaths = 0
        stage = 1; tokens = set(); score = 0
        enable_fc = True; enable_ac = True; ai_lab = False; show_help = False
        lab_algo = 'astar'; lab_job = None; lab_t = 0
        camera = 0.0; sim_acc = 0.0; game_start = monotonic()
        facing = 1; land_t = 0.0; shake = 0.0; flash = 0.0; banner_until = 0.0
        tiles = {'level': None, 'layer': None, 'coins': ()}
        model_view = {'at': -1.0, 'nn': None, 'rule': 0.5}
        player_glow = pg.Surface((72, 72), pg.SRCALPHA)
        pg.draw.circle(player_glow, (98, 231, 255, 26), (36, 36), 35)
        pg.draw.circle(player_glow, (98, 231, 255, 40), (36, 36), 24)
        toasts.push('READY  //  REACH THE PORTAL', P['cyan'], 2.4, 0.0)

        def now():
            return monotonic() - game_start

        def fresh_level():
            nonlocal level, chosen, state, prev, stage, attempt_start, jumps, landings
            nonlocal camera, sim_acc, lab_job, lab_t, tokens, built_with
            chosen = controller.next_level(n_candidates=5)
            level = chosen.result.level
            built_with = (True, True)
            state = level.start; prev = state; stage += 1
            attempt_start = monotonic(); jumps = 0; landings = 0
            camera = 0; sim_acc = 0; lab_job = None; lab_t = 0; tokens = set()
            toasts.push('AI BUILT THE NEXT CHALLENGE', P['gold'], 2.8, now())

        def reset_state():
            nonlocal state, prev, jumps, landings, attempt_start
            state = level.start; prev = state; jumps = 0; landings = 0
            attempt_start = monotonic()

        def lab():
            """The background solve for this level, started on first use."""
            nonlocal lab_job
            if lab_job is None or lab_job.level is not level:
                lab_job = LabJob(level, lab_algo)
            return lab_job

        def world_to_screen(wx, wy, ox, oy):
            return int(wx + ox), int(wy + oy)

        # ---- world -------------------------------------------------------
        def draw_world(t, alpha, ox, oy):
            if tiles['level'] is not level:
                tiles['level'] = level
                tiles['layer'], tiles['coins'] = build_tiles(pg, level)
            screen.blit(tiles['layer'], (int(ox), int(oy)))
            for x, y in tiles['coins']:
                if (x, y) in tokens:
                    continue
                cx = int(x * TILE + TILE / 2 + ox)
                if -TILE < cx < WIDTH + TILE:
                    cy = int(y * TILE + TILE / 2 + oy + sin(t * 4 + x) * 4)
                    w = max(3, int(11 * abs(cos(t * 2.6 + x * 0.7))))
                    pg.draw.ellipse(screen, (120, 86, 30), (cx - w - 2, cy - 12, 2 * w + 4, 26))
                    pg.draw.ellipse(screen, P['gold'], (cx - w, cy - 11, 2 * w, 22), 3)
                    pg.draw.line(screen, (255, 244, 200), (cx - 1, cy - 6), (cx - 1, cy + 4), 2)
            phase, before = state.phase, prev.phase
            for platform in level.platforms:
                mx, my = ui.lerp_cell(platform.position(before), platform.position(phase), alpha)
                sx = int(mx * TILE + ox); sy = int(my * TILE + oy)
                width = platform.width * TILE
                if -width < sx < WIDTH:
                    pg.draw.rect(screen, (44, 120, 140), (sx, sy + 6, width, TILE // 2 - 2), border_radius=7)
                    pg.draw.rect(screen, (101, 218, 213), (sx, sy, width, TILE // 2), border_radius=7)
                    pg.draw.rect(screen, (210, 252, 244), (sx + 4, sy + 3, width - 8, 5), border_radius=3)
                    for bolt in range(platform.width):
                        pg.draw.circle(screen, (30, 84, 110), (sx + TILE // 2 + bolt * TILE, sy + 15), 3)
            for enemy in level.enemies:
                mx, my = ui.lerp_cell(enemy.position(before), enemy.position(phase), alpha)
                look = 1 if enemy.position(phase)[0] >= enemy.position(before)[0] else -1
                sx = int(mx * TILE + TILE / 2 + ox); sy = int(my * TILE + TILE / 2 + oy) + 3
                if -TILE < sx < WIDTH + TILE:
                    for k in (-1, 0, 1):
                        pg.draw.polygon(screen, (196, 64, 104), [(sx + k * 9 - 5, sy - 14), (sx + k * 9, sy - 25),
                                                                (sx + k * 9 + 5, sy - 14)])
                    pg.draw.circle(screen, (246, 100, 138), (sx, sy), TILE // 2 - 4)
                    pg.draw.circle(screen, (250, 235, 232), (sx + 5 * look, sy - 4), 7)
                    pg.draw.circle(screen, (26, 29, 67), (sx + 7 * look, sy - 4), 3)
                    pg.draw.line(screen, (120, 30, 60), (sx - 8, sy + 9), (sx + 8, sy + 9), 3)
            for gate in level.gates:
                gx = int(gate.x * TILE + TILE // 2 + ox)
                if -TILE < gx < WIDTH + TILE:
                    top = int(gate.y_top * TILE + oy); bottom = int((gate.y_bottom + 1) * TILE + oy)
                    if gate.active(phase):
                        flicker = 0.75 + 0.25 * sin(t * 40 + gate.x)
                        pg.draw.line(screen, ui.mix((90, 20, 60), (255, 80, 150), flicker), (gx, top), (gx, bottom), 14)
                        pg.draw.line(screen, (255, 200, 230), (gx, top), (gx, bottom), 4)
                    else:
                        for yy in range(top, bottom, 16):
                            pg.draw.line(screen, (72, 104, 140), (gx, yy), (gx, min(bottom, yy + 8)), 2)
                    for cap in (top, bottom):
                        pg.draw.rect(screen, (110, 120, 170), (gx - 10, cap - 5, 20, 10), border_radius=4)
            # goal portal
            gx = level.goal_x * TILE + TILE / 2 + ox
            gy = (level.height - 2) * TILE - 46 + oy
            if -120 < gx < WIDTH + 120:
                for ring in range(3):
                    r = 40 - ring * 9 + 3 * sin(t * 4 + ring)
                    pg.draw.ellipse(screen, ui.mix(P['cyan'], (239, 182, 255), ring / 2),
                                    (int(gx - r * 0.55), int(gy - r), int(r * 1.1), int(r * 2)), 3)
                for k in range(6):
                    a = t * 2.2 + k * pi / 3
                    pg.draw.circle(screen, (220, 240, 255), (int(gx + cos(a) * 26), int(gy + sin(a) * 42)), 2)
            # AI Lab: the selected algorithm's exploration and its route
            if ai_lab and lab_job is not None and lab_job.level is level:
                res = lab_job.results.get(lab_algo)
                if res is not None:
                    color = ALGO_COLORS[lab_algo]
                    dim = ui.mix(P['sky_top'], color, 0.4)
                    for s in res.trace[:lab_t]:
                        px, py = world_to_screen(s.x * TILE + TILE / 2, (s.y - 1) * TILE + TILE / 2, ox, oy)
                        if -10 < px < WIDTH + 10:
                            pg.draw.circle(screen, dim, (px, py), 2)
                    if res.found and len(res.path) > 1:
                        pts = [world_to_screen(s.x * TILE + TILE / 2, (s.y - 1) * TILE + TILE / 2, ox, oy)
                               for s in res.path]
                        pg.draw.lines(screen, (10, 12, 30), False, pts, 7)
                        pg.draw.lines(screen, color, False, pts, 3)
            # the player: an original neon explorer
            fx, fy = ui.lerp_cell((prev.x, prev.y), (state.x, state.y), alpha)
            X = int(fx * TILE + TILE / 2 + ox); Y = int((fy - 1) * TILE + TILE / 2 + oy) + 4
            squash = max(0.0, land_t) * 10
            w, h = int(28 + squash), int(32 - squash)
            screen.blit(player_glow, (X - 36, Y - 38))
            tail = [(X - facing * 12, Y - 6), (X - facing * (26 + 4 * sin(now() * 12)), Y - 2 + 3 * sin(now() * 9)),
                    (X - facing * 12, Y + 6)]
            pg.draw.polygon(screen, P['pink'], tail)
            body = pg.Rect(0, 0, w, h); body.midbottom = (X, Y + 16)
            pg.draw.rect(screen, (23, 31, 73), body, border_radius=10)
            pg.draw.rect(screen, P['cyan'], body, width=2, border_radius=10)
            visor = pg.Rect(0, 0, w - 8, 11); visor.midtop = (X + facing * 2, body.top + 6)
            pg.draw.rect(screen, (224, 249, 253), visor, border_radius=5)
            for dx in (-4, 5):
                pg.draw.circle(screen, (35, 54, 104), (X + facing * 3 + dx, visor.centery), 2)
            for dx in (-9, 3):
                pg.draw.rect(screen, (88, 84, 189), (X + dx, body.bottom - 2, 8, 6), border_radius=2)

        # ---- heads-up display -------------------------------------------
        def draw_hud(t, alpha):
            kit.panel(screen, (16, 10, WIDTH - 32, 72), alpha=225, radius=16)
            kit.glow_text(screen, 'NEUROFORGE', (36, 38), P['cyan'], 'h1', 'midleft', strength=120)
            kit.pill(screen, (250, 38), f'WORLD {stage:02}', P['text'])
            items = (('COINS', f'{score:03}', P['gold']), ('DEATHS', f'{deaths:02}', P['pink']),
                     ('TIME', f'{monotonic() - attempt_start:4.1f}s', P['text']))
            for n, (label, value, color) in enumerate(items):
                x = 330 + n * 120
                kit.text(screen, label, (x, 18), P['muted'], 'tiny')
                kit.text(screen, value, (x, 32), color, 'numsm')
            kit.text(screen, 'DIFFICULTY', (700, 18), P['muted'], 'tiny')
            kit.bar(screen, (700, 38, 150, 8), level.difficulty, ui.mix(P['green'], P['pink'], level.difficulty))
            kit.text(screen, f'{level.difficulty:.2f}', (860, 31), P['text'], 'h3')
            using_nn = controller.use_nn and controller.model is not None
            prob = chosen.predicted_win
            kit.text(screen, 'PREDICTED WIN', (WIDTH - 118, 18), P['muted'], 'tiny', 'topright')
            kit.text(screen, f'{prob * 100:4.1f}%', (WIDTH - 118, 30),
                     P['green'] if prob >= 0.65 else P['gold'], 'num', 'topright')
            kit.pill(screen, (WIDTH - 72, 46), 'NN' if using_nn else 'RULE',
                     P['cyan'] if using_nn else P['gold'])
            fx = ui.lerp_cell((prev.x, prev.y), (state.x, state.y), alpha)[0]
            kit.bar(screen, (36, 68, WIDTH - 72, 5), fx / max(1, level.goal_x), P['cyan'])

        def draw_goal_hint(ox):
            gx = level.goal_x * TILE + TILE / 2 + ox
            right = WIDTH - (LAB_W if ai_lab else 0)
            if gx > right - 20:
                tiles_left = level.goal_x - state.x
                cy = TOP + (level.height - 3) * TILE
                x = right - 30
                pulse = 4 * sin(now() * 6)
                pg.draw.polygon(screen, P['cyan'], [(x + 14 + pulse, cy), (x - 4 + pulse, cy - 14), (x - 4 + pulse, cy + 14)])
                kit.text(screen, f'GOAL {tiles_left}', (x - 12, cy), P['cyan'], 'h3', 'midright')

        def draw_footer():
            band = pg.Surface((WIDTH, 44), pg.SRCALPHA)
            band.fill((6, 8, 22, 190))
            screen.blit(band, (0, HEIGHT - 44))
            kit.chips_row(screen, [('A / D', 'Move'), ('Space', 'Jump'), ('Tab', 'AI Lab'), ('1-4', 'Search'),
                                   ('N', 'Neural net'), ('G', 'New level'), ('V', 'Arena'), ('F1', 'Help'),
                                   ('Esc', 'Menu')], HEIGHT - 33)

        # ---- AI Lab dashboard ---------------------------------------------
        def model_estimates(t):
            if t - model_view['at'] > 0.5:
                model_view['at'] = t
                model_view['rule'] = float(fallback_predict(controller.profile, level))
                model_view['nn'] = (controller.adapter.predict(controller.model, controller.feature_vector(level))
                                    if controller.model is not None else None)
            return model_view['nn'], model_view['rule']

        def draw_lab(t):
            nonlocal lab_t
            if not ai_lab:
                return
            job = lab()
            box = pg.Rect(WIDTH - LAB_W, 92, LAB_W - 14, HEIGHT - 92 - 50)
            kit.panel(screen, box, alpha=240, radius=16)
            x, y, w = box.left, box.top, box.width
            kit.text(screen, 'AI LABORATORY', (x + 18, y + 14), P['cyan'], 'h1')
            kit.text(screen, 'Tab to close', (x + w - 18, y + 20), P['dim'], 'tiny', 'topright')
            def heading(label, note, top):
                rect = kit.text(screen, label, (x + 18, top), P['gold'], 'h3')
                kit.text(screen, note, (rect.right + 10, rect.bottom - 1), P['muted'], 'tiny', 'bottomleft')
            # search
            y0 = y + 52
            heading('SEARCH', 'nodes expanded on this level (log scale)', y0)
            done = {a: r for a, r in job.results.items()}
            peak = max([r.expanded for r in done.values()] + [10])
            for n, algo in enumerate(ALGOS):
                row = pg.Rect(x + 12, y0 + 24 + n * 30, w - 24, 26)
                selected = algo == lab_algo
                color = ALGO_COLORS[algo]
                if selected:
                    pg.draw.rect(screen, P['panel_hi'], row, border_radius=8)
                    pg.draw.rect(screen, color, row, width=1, border_radius=8)
                kit.text(screen, str(n + 1), (row.left + 10, row.centery), P['dim'], 'key', 'midleft')
                kit.text(screen, ALGO_LABELS[algo], (row.left + 28, row.centery),
                         color if selected else P['muted'], 'h3', 'midleft')
                res = done.get(algo)
                bar_rect = (row.left + 104, row.centery - 5, w - 200, 10)
                if res is None:
                    kit.text(screen, 'solving...', (row.left + 104, row.centery), P['dim'], 'tiny', 'midleft')
                else:
                    frac = log10(res.expanded + 1) / log10(peak + 1)
                    kit.bar(screen, bar_rect, frac, color if selected else ui.mix(P['panel_hi'], color, 0.55))
                    kit.text(screen, f'{res.expanded:,}', (row.right - 10, row.centery),
                             P['text'] if selected else P['muted'], 'h3', 'midright')
            res = done.get(lab_algo)
            if res is not None:
                lab_t = min(len(res.trace), lab_t + 9)
                if res.found:
                    line = (f'{ALGO_LABELS[lab_algo]} route: {res.cost} moves  |  '
                            f'{res.generated:,} generated  |  {res.elapsed_ms:.1f} ms')
                    kit.text(screen, line, (x + 18, y0 + 150), P['text'], 'small', width=w - 36)
                else:
                    kit.text(screen, f'No solution  ({res.reason})', (x + 18, y0 + 150), P['pink'], 'small')
            # constraint satisfaction
            y1 = y + 232
            heading('CONSTRAINTS', 'a CSP of this size and difficulty', y1)
            if job.csp is None:
                kit.text(screen, 'solving...', (x + 18, y1 + 40), P['dim'], 'small')
            else:
                top = max([max(s.backtracks, s.pruned) for _, s in job.csp] + [1])
                gw = (w - 36) // 4
                base = y1 + 108
                for n, (name, stats) in enumerate(job.csp):
                    gx = x + 18 + n * gw
                    current = (CSP_MODES[n][1], CSP_MODES[n][2]) == (enable_fc, enable_ac)
                    if current:
                        hl = pg.Rect(gx + 2, y1 + 22, gw - 4, 112)
                        pg.draw.rect(screen, P['panel_hi'], hl, border_radius=8)
                        pg.draw.rect(screen, P['cyan'], hl, width=1, border_radius=8)
                    for k, (value, color) in enumerate(((stats.backtracks, P['pink']), (stats.pruned, P['gold']))):
                        bh = max(2, int(62 * value / top))
                        bx = gx + gw // 2 - 20 + k * 22
                        pg.draw.rect(screen, color, (bx, base - bh, 18, bh), border_radius=3)
                        kit.text(screen, str(value), (bx + 9, base - bh - 3), P['muted'], 'tiny', 'midbottom')
                    kit.text(screen, name, (gx + gw // 2, base + 6), P['text'] if current else P['muted'], 'h3', 'midtop')
                lx = x + 18
                pg.draw.rect(screen, P['pink'], (lx, y1 + 144, 10, 10), border_radius=2)
                kit.text(screen, 'backtracks', (lx + 16, y1 + 149), P['muted'], 'tiny', 'midleft')
                pg.draw.rect(screen, P['gold'], (lx + 96, y1 + 144, 10, 10), border_radius=2)
                kit.text(screen, 'values pruned', (lx + 112, y1 + 149), P['muted'], 'tiny', 'midleft')
                kit.text(screen, 'F / C toggle   G rebuilds', (x + w - 18, y1 + 149), P['cyan'], 'tiny',
                         'midright')
            stats = chosen.result.csp.stats
            built = f'FC {"on" if built_with[0] else "off"}, AC-3 {"on" if built_with[1] else "off"}'
            kit.text(screen, f'This level ({built}): {stats.backtracks} backtracks, {stats.pruned} pruned',
                     (x + 18, y1 + 168), P['text'], 'small', width=w - 36)
            # player model
            y2 = y + 438
            heading('PLAYER MODEL', 'your chance to clear this level', y2)
            nn, rule = model_estimates(t)
            using_nn = controller.use_nn and controller.model is not None
            kit.gauge(screen, (x + w // 4 + 10, y2 + 66), 36, nn, P['cyan'],
                      '--' if nn is None else f'{nn * 100:.0f}%', 'NEURAL NET', active=using_nn)
            kit.gauge(screen, (x + 3 * w // 4 - 10, y2 + 66), 36, rule, P['gold'],
                      f'{rule * 100:.0f}%', 'RULE BASELINE', active=not using_nn)
            ad = controller.adapter
            kit.text(screen, f'N switches the model  |  online adapter: {ad.updates} updates, '
                             f'{len(ad.pending)}/10 new', (x + w // 2, box.bottom - 18), P['muted'], 'tiny', 'center')

        running = True
        while running:
            dt = min(clock.tick(60) / 1000, 0.05)
            t = now()
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    return None
                if event.type != pg.KEYDOWN:
                    continue
                if show_help:
                    if event.key in (pg.K_ESCAPE, pg.K_F1, pg.K_h):
                        show_help = False
                    continue
                if event.key == pg.K_ESCAPE:
                    return 'title'
                elif event.key in (pg.K_F1, pg.K_h):
                    show_help = True
                elif event.key == pg.K_F11:
                    ui.toggle_fullscreen(pg)
                elif event.key == pg.K_TAB:
                    ai_lab = not ai_lab; lab_t = 0
                elif event.key == pg.K_r:
                    reset_state()
                elif event.key == pg.K_x:
                    event_details = controller.adapter.forget_now('manual')
                    if event_details and controller.persist:
                        controller.adapter.save()
                    toasts.push('MEMORY GLITCH  //  ADAPTER PARTIALLY ERASED' if event_details
                                else 'NO ADAPTER MEMORY TO ERASE', P['gold'], 2.5, t)
                elif event.key == pg.K_v:
                    return 'arena'
                elif event.key == pg.K_n:
                    controller.use_nn = not controller.use_nn
                    toasts.push('NEURAL NETWORK ' + ('ENABLED' if controller.use_nn else 'OFF  //  RULE BASELINE'),
                                P['cyan'], 2.2, t)
                elif event.key == pg.K_f:
                    enable_fc = not enable_fc
                    toasts.push('FORWARD CHECKING ' + ('ON' if enable_fc else 'OFF') + '  //  G TO REBUILD', P['gold'], 2, t)
                elif event.key == pg.K_c:
                    enable_ac = not enable_ac
                    toasts.push('AC-3 ' + ('ON' if enable_ac else 'OFF') + '  //  G TO REBUILD', P['gold'], 2, t)
                elif event.key == pg.K_g:
                    from csp.generator import generate
                    result = generate(seed=controller.seed, target=level.difficulty,
                                      forward_checking=enable_fc, arc_consistency=enable_ac)
                    if result.level:
                        level = result.level; chosen.result = result; reset_state()
                        lab_job = None; lab_t = 0; built_with = (enable_fc, enable_ac)
                        toasts.push('NEW CONSTRAINT-VALID LEVEL', P['green'], 2.5, t)
                    else:
                        toasts.push('NO SOLUTION', P['pink'], 2.5, t)
                elif event.key in (pg.K_1, pg.K_2, pg.K_3, pg.K_4):
                    lab_algo = {pg.K_1: 'bfs', pg.K_2: 'dfs', pg.K_3: 'astar', pg.K_4: 'greedy'}[event.key]
                    ai_lab = True; lab_t = 0
            keys = pg.key.get_pressed()
            if not show_help:
                sim_acc += dt
            while sim_acc >= TICK:
                sim_acc -= TICK
                horizontal = (-1 if keys[pg.K_a] or keys[pg.K_LEFT] else 0) + (
                    1 if keys[pg.K_d] or keys[pg.K_RIGHT] else 0)
                if horizontal:
                    facing = horizontal
                jump = bool(keys[pg.K_SPACE] or keys[pg.K_w] or keys[pg.K_UP])
                act = ('JL' if horizontal < 0 else 'JR' if horizontal > 0 else 'J') if jump else (
                    'L' if horizontal < 0 else 'R' if horizontal > 0 else 'W')
                old_grounded = level.solid(state.x, state.y) and state.vy == 0
                if jump and old_grounded:
                    jumps += 1
                prev = state
                nxt = step(level, state, act)
                if nxt is None:
                    deaths += 1
                    particles.emit(state.x * TILE + TILE / 2, (state.y - 1) * TILE + TILE / 2, P['pink'],
                                   count=26, speed=320, life=0.7)
                    shake = 14; flash = 1.0
                    controller.record(level, False, monotonic() - attempt_start,
                                      jumps, landings, state.x / max(level.goal_x, 1),
                                      save=persist, origin='human')
                    toasts.push('TRY AGAIN  //  YOUR SKILL PROFILE UPDATED', P['pink'], 2.2, t)
                    if deaths % 3 == 0:
                        fresh_level()
                    else:
                        reset_state()
                    break
                state = nxt
                grounded = level.solid(state.x, state.y) and state.vy == 0
                if grounded and not old_grounded:
                    landings += 1; land_t = 0.25
                    particles.emit(state.x * TILE + TILE / 2, state.y * TILE, (150, 170, 210), count=8,
                                   speed=110, life=0.35, gravity=200, size=3, spread=pi * 0.9)
                # Collect rewards without mutating the level/search representation.
                for row in (state.y - 1, state.y - 2):
                    if level.tile(state.x, row) == 'o' and (state.x, row) not in tokens:
                        tokens.add((state.x, row)); score += 10
                        particles.emit(state.x * TILE + TILE / 2, row * TILE + TILE / 2, P['gold'],
                                       count=14, speed=200, life=0.5, gravity=300)
                if level.complete(state):
                    controller.record(level, True, monotonic() - attempt_start,
                                      jumps, landings, 1.0, save=persist, origin='human')
                    particles.emit(state.x * TILE + TILE / 2, (state.y - 1) * TILE, P['cyan'],
                                   count=40, speed=360, life=0.9)
                    banner_until = t + 1.8
                    fresh_level(); break
            # ---- effects bookkeeping ----------------------------------------
            particles.update(dt)
            land_t = max(0.0, land_t - dt)
            shake = max(0.0, shake - dt * 40)
            flash = max(0.0, flash - dt * 2.5)
            alpha = min(1.0, sim_acc / TICK)
            view_w = WIDTH - (LAB_W if ai_lab else 0)
            px = ui.lerp_cell((prev.x, prev.y), (state.x, state.y), alpha)[0]
            desired = px * TILE - view_w * 0.38
            desired = max(0.0, min(desired, max(0.0, level.width * TILE - view_w)))
            camera += (desired - camera) * min(1.0, dt * 7)
            jx = (rng.random() - 0.5) * shake; jy = (rng.random() - 0.5) * shake
            ox, oy = -camera + jx, TOP + jy
            # ---- draw ---------------------------------------------------------
            backdrop.draw(screen, t, camera)
            draw_world(t, alpha, ox, oy)
            particles.draw(pg, screen, ox, oy)
            draw_goal_hint(ox)
            draw_footer()
            draw_hud(t, alpha)
            draw_lab(t)
            toasts.draw(kit, screen, view_w // 2, 96, t)
            if t < banner_until:
                k = min(1.0, (banner_until - t) * 2, (1.8 - (banner_until - t)) * 5)
                kit.glow_text(screen, 'WORLD CLEAR', (view_w // 2, HEIGHT // 2 - 30), P['green'], 'logo',
                              strength=int(200 * k))
                kit.text(screen, 'the AI is building your next challenge', (view_w // 2, HEIGHT // 2 + 26),
                         P['text'], 'h2', 'center', alpha=255 * k)
            if flash > 0:
                red = pg.Surface((WIDTH, HEIGHT), pg.SRCALPHA)
                red.fill((255, 60, 100, int(110 * flash)))
                screen.blit(red, (0, 0))
            if show_help:
                kit.overlay_help(screen, 'CONTROLS  //  MANUAL MODE', ui.MANUAL_HELP)
            pg.display.flip()
    finally:
        pg.quit()


if __name__ == '__main__':
    run()
