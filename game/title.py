"""NEUROFORGE title screen: choose the manual game or the four-agent arena.

Returns 'manual', 'arena' or None (quit). Pure presentation; no AI runs here.
"""
from __future__ import annotations
from math import sin
from time import monotonic
from game import ui
from game.ui import WIDTH, HEIGHT, PALETTE as P, ALGO_COLORS

MENU = (
    ('EXPEDITION', 'Isometric alien shooter: a CSP builds the map, search drives the aliens', 'expedition'),
    ('PLAY', 'Run and jump through levels the AI builds for you', 'manual'),
    ('AI ARENA', 'Watch BFS, DFS, A* and Greedy race and replan live', 'arena'),
    ('HELP', 'Every key in one place', 'help'),
    ('QUIT', 'Close NEUROFORGE', 'quit'),
)
PILLARS = (('SEARCH', 'BFS / DFS / A* / Greedy', P['cyan']),
           ('CONSTRAINTS', 'Backtracking + FC + AC-3', P['gold']),
           ('NEURAL NETWORK', 'Predicts your chance to win', P['pink']))


def run_title(seed=42):
    try:
        import pygame as pg
    except ImportError as ex:
        raise RuntimeError('Install Pygame CE: python -m pip install pygame-ce==2.5.8') from ex
    pg.init()
    try:
        screen = ui.open_window(pg, 'NEUROFORGE')
        clock = pg.time.Clock()
        kit = ui.Kit(pg)
        backdrop = ui.Backdrop(pg, seed=seed, planet=(1150, 96))
        selected = 0
        show_help = False
        start = monotonic()
        buttons = []
        for n in range(len(MENU)):
            rect = pg.Rect(0, 0, 360, 46)
            rect.center = (WIDTH // 2, 358 + n * 54)
            buttons.append(rect)

        def choose(index):
            nonlocal show_help
            action = MENU[index][2]
            if action == 'help':
                show_help = True
                return 'stay'
            return None if action == 'quit' else action

        while True:
            clock.tick(60)
            t = monotonic() - start
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    return None
                if event.type == pg.KEYDOWN:
                    if show_help:
                        show_help = False     # any key closes the help
                        continue
                    if event.key == pg.K_ESCAPE:
                        return None
                    if event.key in (pg.K_UP, pg.K_w):
                        selected = (selected - 1) % len(MENU)
                    elif event.key in (pg.K_DOWN, pg.K_s, pg.K_TAB):
                        selected = (selected + 1) % len(MENU)
                    elif event.key in (pg.K_RETURN, pg.K_KP_ENTER, pg.K_SPACE):
                        result = choose(selected)
                        if result != 'stay':
                            return result
                    elif event.key in (pg.K_3, pg.K_e):
                        return 'expedition'
                    elif event.key in (pg.K_1, pg.K_p):
                        return 'manual'
                    elif event.key in (pg.K_2, pg.K_a, pg.K_v):
                        return 'arena'
                    elif event.key in (pg.K_F1, pg.K_h):
                        show_help = True
                    elif event.key == pg.K_F11:
                        ui.toggle_fullscreen(pg)
                elif event.type == pg.MOUSEMOTION and not show_help:
                    for n, rect in enumerate(buttons):
                        if rect.collidepoint(event.pos):
                            selected = n
                elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    if show_help:
                        show_help = False
                        continue
                    for n, rect in enumerate(buttons):
                        if rect.collidepoint(event.pos):
                            result = choose(n)
                            if result != 'stay':
                                return result

            # ---- background: drifting sky and a strip of neon ground -----
            backdrop.draw(screen, t, camera=t * 38)
            ground_y = HEIGHT - 70
            pg.draw.rect(screen, P['tile'], (0, ground_y, WIDTH, 70))
            pg.draw.rect(screen, P['tile_top'], (0, ground_y, WIDTH, 6))
            pg.draw.rect(screen, P['tile_edge'], (0, ground_y + 6, WIDTH, 4))
            off = int(t * 38) % 48
            for x in range(-off, WIDTH, 48):
                pg.draw.line(screen, P['tile_line'], (x + 6, ground_y + 40), (x + 40, ground_y + 40))
            # four little agents racing along the bottom, one per search method
            for n, algo in enumerate(('bfs', 'dfs', 'astar', 'greedy')):
                speed = (90, 120, 150, 180)[n]
                x = (t * speed + n * 330) % (WIDTH + 80) - 40
                hop = abs(sin(t * 5 + n)) * 26
                y = ground_y - 18 - hop
                color = ALGO_COLORS[algo]
                shadow = max(4, int(12 - hop * 0.25))
                pg.draw.ellipse(screen, (24, 26, 58), (int(x) - shadow, ground_y - 3, 2 * shadow, 6))
                pg.draw.rect(screen, (22, 30, 66), (x - 10, y - 12, 20, 22), border_radius=6)
                pg.draw.rect(screen, color, (x - 10, y - 12, 20, 22), width=2, border_radius=6)
                pg.draw.rect(screen, (226, 250, 254), (x - 6, y - 7, 12, 7), border_radius=3)

            # ---- logo and the three ideas ---------------------------------
            kit.glow_text(screen, 'NEUROFORGE', (WIDTH // 2, 148), P['cyan'], 'logo', strength=200)
            kit.text(screen, 'THE PLATFORMER THAT LEARNS HOW YOU PLAY', (WIDTH // 2, 206),
                     P['muted'], 'h3', 'center')
            for n, (name, detail, color) in enumerate(PILLARS):
                cx = WIDTH // 2 + (n - 1) * 300
                box = pg.Rect(0, 0, 270, 64)
                box.center = (cx, 270)
                kit.panel(screen, box, alpha=200, border=ui.mix(P['line'], color, 0.5), radius=12)
                pg.draw.rect(screen, color, (box.left + 14, box.top + 16, 4, 32), border_radius=2)
                kit.text(screen, name, (box.left + 30, box.top + 14), color, 'h3')
                kit.text(screen, detail, (box.left + 30, box.top + 36), P['text'], 'small')

            # ---- menu -----------------------------------------------------
            for n, (label, detail, _) in enumerate(MENU):
                rect = buttons[n]
                active = n == selected
                if active:
                    pulse = 0.5 + 0.5 * sin(t * 4)
                    halo = rect.inflate(10 + 4 * pulse, 10 + 4 * pulse)
                    layer = pg.Surface(halo.size, pg.SRCALPHA)
                    pg.draw.rect(layer, (*P['cyan'], 40), layer.get_rect(), border_radius=18)
                    screen.blit(layer, halo.topleft)
                kit.panel(screen, rect, fill=P['panel_hi'] if active else P['panel'],
                          border=P['cyan'] if active else P['line'], radius=14,
                          alpha=245 if active else 215, border_width=2 if active else 1)
                kit.text(screen, label, rect.center, P['text'] if active else P['muted'], 'menu', 'center')
                if active:
                    kit.text(screen, '>', (rect.left + 22, rect.centery), P['cyan'], 'menu', 'midleft')
            kit.text(screen, MENU[selected][1], (WIDTH // 2, buttons[-1].bottom + 22),
                     P['muted'], 'small', 'center')

            kit.chips_row(screen, [('Up / Down', 'Choose'), ('Enter', 'Start'), ('F1', 'Help'),
                                   ('F11', 'Fullscreen'), ('Esc', 'Quit')], HEIGHT - 44)
            if show_help:
                kit.overlay_help(screen, 'CONTROLS',
                                 [(h, rows) for h, rows in ui.MANUAL_HELP[:4]] +
                                 [('ARENA', [('E', 'Surprise every agent'), ('Space', 'Pause'),
                                             ('+ / -', 'Simulation speed'), ('M / V', 'Switch modes')])],
                                 footer='Any key to close')
            pg.display.flip()
    finally:
        pg.quit()


if __name__ == '__main__':
    print(run_title())
