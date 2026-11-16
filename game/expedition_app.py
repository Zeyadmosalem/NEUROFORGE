"""NEUROFORGE Expedition: an isometric pixel-art alien shooter (renderer + input).

The world is drawn isometrically on a 640x360 canvas and scaled x2 with
nearest-neighbour, so every pixel stays crisp; the HUD is drawn at 1280x720.
All game rules live in integration.expedition (headless and unit-tested): this
file only turns that state into pictures and keyboard / mouse input into moves.

Returns 'title' (Esc) or None (window closed).
"""
from __future__ import annotations
import threading
from math import atan2, cos, sin, degrees, hypot, log10, pi
from random import Random
from time import monotonic
from csp.rooms import RoomCSP, ROOM
from integration.expedition import Expedition, ExpeditionDirector, ALIENS
from game import ui, pixel_art as art
from game.ui import WIDTH, HEIGHT, PALETTE as P, ALGO_COLORS, ALGO_LABELS

LOW_W, LOW_H = WIDTH // 2, HEIGHT // 2      # world canvas, scaled x2 to the window
HW, HH = art.TILE_W // 2, art.TILE_H // 2     # half diamond: 16 x 8
WALL_H, PILLAR_H, CRATE_H = 22, 30, 11
AMBIENT = (78, 82, 112)                       # light level where no lamp reaches
ALGOS = ('bfs', 'dfs', 'astar', 'greedy')
CSP_MODES = (('BT', False, False), ('FC', True, False), ('AC-3', False, True), ('FC+AC-3', True, True))
LAB_W = 400
ROOM_FLOOR = {'start': (78, 88, 116), 'exit': (60, 98, 116), 'open': (72, 78, 106), 'supply': (86, 82, 102),
              'cover': (70, 76, 102), 'hazard': (62, 84, 84), 'nest': (86, 62, 100)}
GOO = {'glob': (120, 230, 90), 'skitter': (200, 110, 255), 'brute': (255, 150, 70)}
EXPEDITION_HELP = [
    ('MOVE AND FIGHT', [('W A S D', 'Move (arrow keys work too)'), ('Mouse', 'Aim'),
                        ('Click / Space', 'Shoot (hold for auto fire)'), ('R', 'Reload')]),
    ('THE ALIENS', [('Glob', 'Follows one shared BFS distance field'),
                    ('Skitter', 'Fast; replans with Greedy best-first'),
                    ('Brute', 'Tough; replans with A*'), ('Nest', 'Spawns aliens; shoot it down')]),
    ('AI LAB', [('Tab', 'Routes to the exit by all four searches'),
                ('N', 'Neural net vs rule baseline for the next map')]),
    ('WINDOW', [('Enter', 'Next sector after a win or a death'), ('F11', 'Fullscreen'),
                ('F1', 'This help'), ('Esc', 'Back to the menu')]),
]


def iso(wx, wy):
    return (wx - wy) * HW, (wx + wy) * HH


def from_screen(dx, dy):
    """Screen offset (low-res pixels) back to a world direction."""
    return (dx / HW + dy / HH) / 2, (dy / HH - dx / HW) / 2


class LabWorker:
    """Runs the AI Lab's searches off the render thread."""

    def __init__(self):
        self.routes = None
        self.routes_at = -9.0
        self.csp = None
        self.busy = False

    def refresh(self, game, now, choice, with_csp):
        if self.busy:
            return
        self.busy = True
        self.routes_at = now

        def work():
            try:
                self.routes = game.lab_routes()
                if with_csp and self.csp is None:
                    r = choice.result
                    m = choice.map
                    self.csp = [(name, RoomCSP(m.rows, m.cols, r.target, r.csp_seed).solve(fc, ac))
                                for name, fc, ac in CSP_MODES]
            finally:
                self.busy = False
        threading.Thread(target=work, daemon=True, name='expedition-lab').start()


class Visuals:
    """Pre-rendered floor, block sprites and lights for one map."""

    def __init__(self, pg, m, sprites):
        self.pg = pg
        self.map = m
        self.ox = m.height * HW + HW            # layer x of world origin
        layer_w = (m.width + m.height) * HW + 2 * HW
        layer_h = (m.width + m.height) * HH + 2 * HH
        self.floor = pg.Surface((layer_w, layer_h)).convert()
        self.floor.fill((6, 7, 14))
        self.acid = []
        self.blocks = []
        for y in range(m.height):
            for x in range(m.width):
                t = m.tile(x, y)
                room = m.rooms[(y // ROOM) * m.cols + x // ROOM]
                base = ROOM_FLOOR.get(room.template.kind, (50, 54, 76))
                shade = 5 if (x + y) % 2 else 0
                color = tuple(min(255, c + shade) for c in base)
                cx, cy = self.ox + (x - y) * HW, (x + y) * HH
                diamond = [(cx, cy), (cx + HW, cy + HH), (cx, cy + 2 * HH), (cx - HW, cy + HH)]
                if t == '~':
                    pg.draw.polygon(self.floor, (24, 70, 34), diamond)
                    self.acid.append((x, y))
                elif t != '#':
                    pg.draw.polygon(self.floor, color, diamond)
                    pg.draw.lines(self.floor, tuple(max(0, c - 14) for c in color), True, diamond)
                    if (x * 7 + y * 3) % 9 == 0:
                        pg.draw.line(self.floor, tuple(c + 10 for c in color), (cx - 4, cy + 8), (cx + 4, cy + 8))
                if t == 'S':
                    pg.draw.polygon(self.floor, (120, 220, 240), diamond, 1)
                    pg.draw.polygon(self.floor, (70, 140, 170), [(cx, cy + 4), (cx + 8, cy + 8), (cx, cy + 12),
                                                                (cx - 8, cy + 8)], 1)
                if t in '#cpN':
                    self.blocks.append(((x + y + 1.0), x, y, t))
        dark_trim = (90, 200, 220)
        self.wall = art.iso_block(pg, WALL_H, (74, 80, 112), (46, 50, 76), (34, 36, 58), dark_trim)
        self.crate = art.iso_block(pg, CRATE_H, (156, 116, 64), (114, 80, 42), (90, 62, 32), (196, 156, 96),
                                   rivets=False)
        self.pillar = art.iso_block(pg, PILLAR_H, (116, 124, 156), (72, 78, 106), (58, 62, 88), (255, 206, 110))
        self.faded = {k: s.copy() for k, s in (('#', self.wall), ('c', self.crate), ('p', self.pillar))}
        for s in self.faded.values():
            s.set_alpha(90)
        self.heights = {'#': WALL_H, 'c': CRATE_H, 'p': PILLAR_H}
        self.sprites = sprites


def run_expedition(seed=42, persist=True):
    try:
        import pygame as pg
    except ImportError as ex:
        raise RuntimeError('Install Pygame CE: python -m pip install pygame-ce==2.5.8') from ex
    pg.init()
    try:
        screen = ui.open_window(pg, 'NEUROFORGE  //  EXPEDITION')
        clock = pg.time.Clock()
        kit = ui.Kit(pg)
        rng = Random(seed)
        sprites = art.Sprites(pg)
        world = pg.Surface((LOW_W, LOW_H)).convert()
        lightmap = pg.Surface((LOW_W, LOW_H)).convert()
        lights = {
            'player': art.radial_light(pg, 88, (150, 140, 124)),
            'muzzle': art.radial_light(pg, 60, (255, 196, 110)),
            'bullet': art.radial_light(pg, 12, (255, 210, 130)),
            'acid': art.radial_light(pg, 20, (40, 130, 50)),
            'nest': art.radial_light(pg, 44, (170, 60, 190)),
            'beacon': art.radial_light(pg, 80, (70, 200, 230)),
            'pickup': art.radial_light(pg, 16, (130, 120, 70)),
            'boom': art.radial_light(pg, 50, (255, 170, 90)),
        }
        cone = art.cone_light(pg, 190, 46, (190, 182, 160))
        cone_cache = {}
        director = ExpeditionDirector(seed=seed, persist=persist)
        toasts = ui.Toasts()
        lab = LabWorker()
        state = {'sector': 0}

        def new_sector():
            choice = director.next_map()
            state['sector'] += 1
            state['choice'] = choice
            state['game'] = Expedition(choice.map, seed=seed + state['sector'] * 7)
            state['visuals'] = Visuals(pg, choice.map, sprites)
            state['visited'] = set()
            state['recorded'] = False
            state['particles'] = []
            state['decals'] = []
            state['flashes'] = []
            state['alerts'] = []
            state['shake'] = 0.0
            state['hurt'] = 0.0
            state['muzzle'] = 0.0
            g = state['game']
            px, py = iso(g.player.x, g.player.y)
            state['cam'] = [px - LOW_W / 2, py - LOW_H / 2]
            lab.routes = None; lab.csp = None; lab.routes_at = -9.0
            nests = len(choice.map.nests)
            toasts.push(f'SECTOR {state["sector"]:02}  //  {nests} NEST{"S" if nests != 1 else ""}  //  REACH THE BEACON',
                        P['cyan'], 3.0, now())

        start = monotonic()

        def now():
            return monotonic() - start

        new_sector()
        show_help = False
        ai_lab = False
        pg.mouse.set_visible(False)

        def burst(x, y, color, count, speed=3.0, up=40.0, life=0.6, size=1):
            for _ in range(count):
                a = rng.random() * 2 * pi
                v = speed * (0.3 + rng.random())
                state['particles'].append([x, y, 6.0, cos(a) * v, sin(a) * v, up * (0.4 + rng.random()),
                                           life * (0.6 + rng.random() * 0.6), life, color, size])

        def handle(events):
            g = state['game']
            for e in events:
                kind, x, y = e[0], e[1], e[2]
                if kind == 'shot':
                    state['muzzle'] = 0.06
                    state['shake'] = max(state['shake'], 1.2)
                    burst(x, y, (255, 220, 140), 2, speed=1.5, up=20, life=0.15)
                elif kind == 'hit':
                    burst(x, y, (150, 240, 110), 5, speed=2.5, up=30, life=0.35)
                elif kind == 'spark':
                    burst(x, y, (255, 214, 120), 4, speed=3, up=40, life=0.25)
                elif kind == 'kill':
                    color = GOO.get(e[3], (150, 240, 110))
                    burst(x, y, color, 22, speed=4, up=70, life=0.8, size=2)
                    state['decals'].append((x, y, ui.mix((12, 12, 20), color, 0.45), 5 + rng.random() * 4))
                    state['decals'] = state['decals'][-140:]
                    state['flashes'].append([x, y, 0.18])
                    state['shake'] = max(state['shake'], 2.5)
                elif kind == 'hurt':
                    state['hurt'] = 0.35
                    state['shake'] = max(state['shake'], 4.0)
                    burst(x, y, (255, 70, 90), 8, speed=2.5, up=50, life=0.4)
                elif kind == 'acid':
                    burst(x, y, (120, 240, 100), 3, speed=1, up=30, life=0.4)
                elif kind == 'pickup':
                    burst(x, y, (255, 230, 140), 14, speed=2, up=60, life=0.6)
                    toasts.push('+30 AMMO' if e[3] == 'ammo' else '+35 HEALTH',
                                P['gold'] if e[3] == 'ammo' else P['green'], 1.4, now())
                elif kind == 'spawn':
                    burst(x, y, (200, 110, 255), 10, speed=2, up=40, life=0.5)
                elif kind == 'nest_down':
                    burst(x, y, (220, 120, 255), 40, speed=5, up=90, life=1.0, size=2)
                    state['flashes'].append([x, y, 0.4])
                    state['shake'] = max(state['shake'], 6.0)
                    toasts.push('NEST DESTROYED', P['pink'], 2.0, now())
                elif kind == 'alert':
                    state['alerts'].append([x, y, 0.7])
                elif kind == 'reload':
                    toasts.push('RELOADING', P['muted'], 1.0, now())
                elif kind in ('win', 'dead') and not state['recorded']:
                    state['recorded'] = True
                    director.record(kind == 'win', g.time, g.shots, g.hits, 1.0 if kind == 'win' else g.progress())
                    if kind == 'dead':
                        burst(x, y, (255, 70, 90), 40, speed=4, up=80, life=1.0, size=2)

        def draw_world(t, cam):
            g = state['game']; v = state['visuals']; m = g.map
            cx, cy = cam
            world.fill((6, 7, 14))
            world.blit(v.floor, (-v.ox - cx, -cy))
            # animated acid
            for x, y in v.acid:
                sx, sy = iso(x, y)
                sx -= cx; sy -= cy
                if -HW < sx < LOW_W + HW and -2 * HH < sy < LOW_H:
                    glow = 0.5 + 0.5 * sin(t * 3 + x * 1.7 + y)
                    pg.draw.polygon(world, ui.mix((30, 90, 40), (80, 200, 70), glow),
                                    [(sx, sy + 2), (sx + HW - 3, sy + HH), (sx, sy + 2 * HH - 2), (sx - HW + 3, sy + HH)])
                    bx = int(sx + 5 * sin(t * 2 + x * 3)); by = int(sy + HH + 2 * cos(t * 2.4 + y))
                    world.set_at((bx, by), (190, 255, 160))
            # floor decals and shadows
            for x, y, color, r in state['decals']:
                sx, sy = iso(x, y)
                pg.draw.ellipse(world, color, (sx - cx - r, sy - cy - r / 2, 2 * r, r))
            bodies = [(g.player.x, g.player.y, 7)] + [(a.x, a.y, 4 + 10 * a.radius) for a in g.aliens]
            for x, y, r in bodies:
                sx, sy = iso(x, y)
                pg.draw.ellipse(world, (8, 9, 18), (sx - cx - r, sy - cy - r / 2.4, 2 * r, r / 1.2))
            # depth-sorted blocks and characters
            p = g.player
            psx, psy = iso(p.x, p.y)
            player_box = pg.Rect(psx - cx - 10, psy - cy - 28, 20, 30)
            p_depth = p.x + p.y
            drawables = []
            for depth, x, y, kind in v.blocks:
                bx, by = iso(x + 0.5, y + 0.5)
                sx, sy = bx - cx, by - cy
                if -40 < sx < LOW_W + 40 and -20 < sy < LOW_H + 50:
                    drawables.append((depth, 0, 'block', (sx, sy, kind, x, y)))
            for item in g.pickups:
                if not item.taken:
                    drawables.append((item.x + item.y, 1, 'pickup', item))
            for alien in g.aliens:
                drawables.append((alien.x + alien.y, 1, 'alien', alien))
            drawables.append((p_depth, 1, 'player', p))
            ex, ey = m.exit
            drawables.append((ex + ey + 1.0, 1, 'beacon', (ex + 0.5, ey + 0.5)))
            drawables.sort(key=lambda d: (d[0], d[1]))
            nests = {(n.x, n.y): n for n in g.nests}
            for _, _, kind, obj in drawables:
                if kind == 'block':
                    sx, sy, code, x, y = obj
                    if code == 'N':
                        nest = nests.get((x, y))
                        img = sprites.nest
                        if nest is not None and not nest.dead:
                            pulse = 1 + int(sin(t * 5 + x) > 0.4)
                            world.blit(img, (sx - img.get_width() // 2, sy - img.get_height() + 4 - pulse))
                        else:
                            dark = img.copy(); dark.fill((70, 60, 70), special_flags=pg.BLEND_RGB_MULT)
                            world.blit(dark, (sx - img.get_width() // 2, sy - img.get_height() + 6))
                        continue
                    h = v.heights[code]
                    rect = pg.Rect(sx - HW, sy - HH - h, 2 * HW, 2 * HH + h)
                    sprite = v.faded[code] if (x + y + 1.0 > p_depth and rect.colliderect(player_box)) else \
                        {'#': v.wall, 'c': v.crate, 'p': v.pillar}[code]
                    world.blit(sprite, (sx - HW, sy - HH - h))
                elif kind == 'pickup':
                    sx, sy = iso(obj.x, obj.y)
                    img = sprites.ammo if obj.kind == 'ammo' else sprites.health
                    bob = int(2 * sin(t * 4 + obj.x))
                    world.blit(img, (sx - cx - img.get_width() // 2, sy - cy - img.get_height() - 3 + bob))
                elif kind == 'alien':
                    a = obj
                    sx, sy = iso(a.x, a.y)
                    hx, hy = a.heading
                    facing = 'r' if (hx - hy) >= 0 else 'l'
                    frame = int(t * (7 if a.kind == 'skitter' else 4) + a.x * 3) % 2 if a.active else 0
                    key = (a.kind, frame, facing, 'hit') if a.hit_t > 0 else (a.kind, frame, facing)
                    img = sprites.aliens[key]
                    if not a.active:
                        img = img.copy(); img.fill((150, 150, 170), special_flags=pg.BLEND_RGB_MULT)
                    world.blit(img, (sx - cx - img.get_width() // 2, sy - cy - img.get_height() + 2))
                elif kind == 'player':
                    aim_sx = (p.aim[0] - p.aim[1]) * HW
                    facing = 'r' if aim_sx >= 0 else 'l'
                    frame = int(t * 10) % 4 if p.moving else 0
                    img = sprites.hero[(facing, frame)]
                    x0 = psx - cx - img.get_width() // 2
                    y0 = psy - cy - img.get_height() + 1
                    world.blit(img, (x0, y0))
                    gun = sprites.gun[facing]
                    gx = psx - cx + (2 if facing == 'r' else -gun.get_width() - 2)
                    gy = y0 + img.get_height() - 10
                    world.blit(gun, (gx, gy))
                    state['gun_tip'] = (gx + (gun.get_width() if facing == 'r' else 0), gy + 1)
                elif kind == 'beacon':
                    sx, sy = iso(*obj)
                    sx -= cx; sy -= cy
                    pg.draw.polygon(world, (40, 120, 140), [(sx, sy - 6), (sx + 10, sy - 1), (sx, sy + 4), (sx - 10, sy - 1)])
                    beam = 0.5 + 0.5 * sin(t * 3)
                    pg.draw.line(world, ui.mix((60, 160, 190), (200, 255, 255), beam), (sx, sy - 4), (sx, sy - 40), 3)
                    bob = 3 * sin(t * 2)
                    top = sy - 50 + bob
                    pg.draw.polygon(world, (150, 250, 255), [(sx, top - 6), (sx + 5, top), (sx, top + 6), (sx - 5, top)])

        def draw_lighting(t, cam):
            g = state['game']; m = g.map; v = state['visuals']
            cx, cy = cam
            lightmap.fill(AMBIENT)

            def add(name, wx, wy, lift=0, scale_pulse=None):
                tex = lights[name]
                sx, sy = iso(wx, wy)
                w, h = tex.get_size()
                lightmap.blit(tex, (sx - cx - w // 2, sy - cy - h // 2 - lift), special_flags=pg.BLEND_ADD)

            p = g.player
            add('player', p.x, p.y, 8)
            ax, ay = p.aim
            sdx, sdy = (ax - ay) * HW, (ax + ay) * HH
            angle = int(round(degrees(atan2(-sdy, sdx)) / 4)) * 4
            rotated = cone_cache.get(angle)
            if rotated is None:
                rotated = cone_cache[angle] = pg.transform.rotate(cone, angle)
            psx, psy = iso(p.x, p.y)
            lightmap.blit(rotated, rotated.get_rect(center=(psx - cx, psy - cy - 8)), special_flags=pg.BLEND_ADD)
            if state['muzzle'] > 0:
                add('muzzle', p.x + ax * 0.6, p.y + ay * 0.6, 10)
            for b in g.bullets:
                add('bullet', b.x, b.y, 10)
            for x, y in v.acid:
                sx, sy = iso(x + 0.5, y + 0.5)
                if -30 < sx - cx < LOW_W + 30 and -30 < sy - cy < LOW_H + 30:
                    add('acid', x + 0.5, y + 0.5)
            for nest in g.nests:
                if not nest.dead:
                    add('nest', nest.x + 0.5, nest.y + 0.5, 6)
            ex, ey = m.exit
            add('beacon', ex + 0.5, ey + 0.5, 20)
            for item in g.pickups:
                if not item.taken:
                    add('pickup', item.x, item.y, 4)
            for x, y, life in state['flashes']:
                add('boom', x, y, 6)
            world.blit(lightmap, (0, 0), special_flags=pg.BLEND_MULT)

        def draw_unlit(t, cam):
            """Glowing things drawn after lighting so they stay bright."""
            g = state['game']
            cx, cy = cam
            for b in g.bullets:
                sx, sy = iso(b.x, b.y)
                sx -= cx; sy -= cy + 10
                speed = hypot(b.dx, b.dy) or 1
                tx = (b.dx - b.dy) / speed * HW * 0.35
                ty = (b.dx + b.dy) / speed * HH * 0.35
                pg.draw.line(world, (255, 236, 170), (sx, sy), (sx - tx, sy - ty), 2)
                world.set_at((int(sx), int(sy)), (255, 255, 255))
            if state['muzzle'] > 0 and 'gun_tip' in state:
                gx, gy = state['gun_tip']
                pg.draw.circle(world, (255, 230, 150), (gx, gy), 3)
                pg.draw.circle(world, (255, 255, 255), (gx, gy), 1)
            for part in state['particles']:
                x, y, z = part[0], part[1], part[2]
                sx, sy = iso(x, y)
                k = max(0.0, part[6] / part[7])
                color = ui.mix((20, 20, 30), part[8], 0.35 + 0.65 * k)
                s = part[9]
                pg.draw.rect(world, color, (sx - cx, sy - cy - z, s, s))
            for x, y, life in state['alerts']:
                sx, sy = iso(x, y)
                sx -= cx; sy -= cy + 26 + (0.7 - life) * 10
                pg.draw.rect(world, (255, 214, 92), (sx - 1, sy - 6, 2, 5))
                pg.draw.rect(world, (255, 214, 92), (sx - 1, sy + 1, 2, 2))
            if ai_lab and lab.routes:
                for n, algo in enumerate(ALGOS):
                    res = lab.routes.get(algo)
                    if res is None or not res.found or len(res.path) < 2:
                        continue
                    off = (n - 1.5) * 1.5
                    pts = []
                    for x, y in res.path:
                        sx, sy = iso(x + 0.5, y + 0.5)
                        pts.append((sx - cx + off, sy - cy + off))
                    pg.draw.lines(world, ALGO_COLORS[algo], False, pts, 1)

        def draw_minimap(g, x0, y0):
            m = g.map
            cell = 24
            box = pg.Rect(x0 - 10, y0 - 30, m.cols * cell + 20, m.rows * cell + 40)
            kit.panel(screen, box, alpha=215, radius=12)
            kit.text(screen, 'MAP', (x0, y0 - 24), P['muted'], 'tiny')
            ptile = g.tile_of(g.player)
            here = (ptile[1] // ROOM) * m.cols + ptile[0] // ROOM
            state['visited'].add(here)
            for i, room in enumerate(m.rooms):
                r, c = divmod(i, m.cols)
                rect = pg.Rect(x0 + c * cell + 2, y0 + r * cell + 2, cell - 4, cell - 4)
                seen = i in state['visited']
                base = ROOM_FLOOR.get(room.template.kind, (50, 54, 76))
                pg.draw.rect(screen, ui.mix((16, 18, 34), tuple(min(255, v + 40) for v in base), 1.0 if seen else 0.35),
                             rect, border_radius=3)
                for side in room.doors:
                    if side == 'E':
                        pg.draw.line(screen, P['dim'], (rect.right, rect.centery), (rect.right + 4, rect.centery), 2)
                    elif side == 'S':
                        pg.draw.line(screen, P['dim'], (rect.centerx, rect.bottom), (rect.centerx, rect.bottom + 4), 2)
                if room.template.kind == 'nest':
                    alive = any(not n.dead and (n.y // ROOM) * m.cols + n.x // ROOM == i for n in g.nests)
                    if alive:
                        pg.draw.circle(screen, P['pink'], rect.center, 4)
                if room.template.kind == 'exit':
                    pg.draw.circle(screen, P['cyan'], rect.center, 5, 2)
            px = x0 + g.player.x / ROOM * cell
            py = y0 + g.player.y / ROOM * cell
            pg.draw.circle(screen, P['text'], (int(px), int(py)), 3)

        def draw_hud(t):
            g = state['game']; choice = state['choice']; p = g.player
            # top-left: title, sector, score
            kit.panel(screen, (16, 12, 470, 58), alpha=215, radius=14)
            kit.glow_text(screen, 'EXPEDITION', (34, 41), P['cyan'], 'h1', 'midleft', strength=110)
            kit.pill(screen, (232, 41), f'SECTOR {state["sector"]:02}', P['text'])
            kit.text(screen, 'KILLS', (290, 22), P['muted'], 'tiny')
            kit.text(screen, f'{g.kills:02}', (290, 34), P['pink'], 'numsm')
            kit.text(screen, 'TIME', (360, 22), P['muted'], 'tiny')
            kit.text(screen, f'{g.time:5.1f}s', (360, 34), P['text'], 'numsm')
            # top-centre: the NN's view of this map
            using_nn = director.controller.use_nn and director.controller.model is not None
            kit.panel(screen, (506, 12, 268, 58), alpha=215, radius=14)
            kit.text(screen, 'PREDICTED WIN', (522, 22), P['muted'], 'tiny')
            kit.text(screen, f'{choice.predicted_win * 100:4.1f}%', (522, 34), P['green'] if choice.predicted_win >= 0.6 else P['gold'], 'numsm')
            kit.pill(screen, (640, 41), 'NN' if using_nn else 'RULE', P['cyan'] if using_nn else P['gold'])
            kit.text(screen, 'DIFFICULTY', (684, 22), P['muted'], 'tiny')
            kit.text(screen, f'{g.map.difficulty:.2f}', (684, 34), ui.mix(P['green'], P['pink'], g.map.difficulty), 'numsm')
            if not ai_lab:
                draw_minimap(g, WIDTH - 16 - g.map.cols * 24 - 10, 52)
            # bottom-left: health
            kit.panel(screen, (16, HEIGHT - 70, 330, 54), alpha=215, radius=14)
            kit.text(screen, 'HEALTH', (34, HEIGHT - 62), P['muted'], 'tiny')
            hp = max(0.0, p.hp) / 100
            kit.bar(screen, (34, HEIGHT - 44, 220, 14), hp, ui.mix(P['red'], P['green'], hp), radius=6)
            kit.text(screen, f'{int(max(0, p.hp))}', (268, HEIGHT - 37), P['text'], 'numsm', 'midleft')
            if g.map.tile(*g.tile_of(p)) == '~':
                kit.text(screen, 'ACID!', (300, HEIGHT - 62), P['green'], 'h3')
            # bottom-right: ammo
            kit.panel(screen, (WIDTH - 346, HEIGHT - 70, 330, 54), alpha=215, radius=14)
            kit.text(screen, 'AMMO', (WIDTH - 328, HEIGHT - 62), P['muted'], 'tiny')
            for i in range(12):
                color = P['gold'] if i < p.mag else P['panel_hi']
                pg.draw.rect(screen, color, (WIDTH - 328 + i * 15, HEIGHT - 46, 10, 18), border_radius=3)
            if p.reload_left > 0:
                kit.bar(screen, (WIDTH - 328, HEIGHT - 24, 175, 4), 1 - p.reload_left / 1.1, P['cyan'])
            kit.text(screen, f'/ {p.ammo}', (WIDTH - 130, HEIGHT - 37), P['text'], 'numsm', 'midleft')
            if not ai_lab:
                kit.chips_row(screen, [('WASD', 'Move'), ('Mouse', 'Shoot'), ('Tab', 'AI Lab'),
                                       ('F1', 'Help'), ('Esc', 'Menu')], HEIGHT - 32, center_x=WIDTH // 2)

        def draw_lab(t):
            g = state['game']; choice = state['choice']
            box = pg.Rect(WIDTH - LAB_W - 14, 84, LAB_W, HEIGHT - 84 - 84)
            kit.panel(screen, box, alpha=242, radius=16)
            x, y, w = box.left, box.top, box.width
            kit.text(screen, 'AI LABORATORY', (x + 18, y + 12), P['cyan'], 'h1')
            kit.text(screen, 'Tab to close', (x + w - 18, y + 18), P['dim'], 'tiny', 'topright')

            def heading(label, note, top):
                rect = kit.text(screen, label, (x + 18, top), P['gold'], 'h3')
                kit.text(screen, note, (rect.right + 10, rect.bottom - 1), P['muted'], 'tiny', 'bottomleft')

            y0 = y + 48
            heading('SEARCH', 'routes from you to the beacon (drawn on the map)', y0)
            routes = lab.routes or {}
            peak = max([r.expanded for r in routes.values()] + [10])
            for n, algo in enumerate(ALGOS):
                row = pg.Rect(x + 12, y0 + 22 + n * 28, w - 24, 24)
                color = ALGO_COLORS[algo]
                pg.draw.rect(screen, color, (row.left + 6, row.centery - 5, 10, 10), border_radius=3)
                kit.text(screen, ALGO_LABELS[algo], (row.left + 24, row.centery), color, 'h3', 'midleft')
                res = routes.get(algo)
                if res is None:
                    kit.text(screen, 'solving...', (row.left + 100, row.centery), P['dim'], 'tiny', 'midleft')
                    continue
                kit.bar(screen, (row.left + 100, row.centery - 4, w - 250, 8),
                        log10(res.expanded + 1) / log10(peak + 1), ui.mix(P['panel_hi'], color, 0.75))
                cost = f'{res.cost} steps' if res.found else 'none'
                kit.text(screen, f'{res.expanded:,} exp', (row.right - 70, row.centery), P['text'], 'tiny', 'midright')
                kit.text(screen, cost, (row.right - 6, row.centery), P['muted'], 'tiny', 'midright')
            counts = {}
            for a in g.aliens:
                if a.active:
                    counts[a.algorithm] = counts.get(a.algorithm, 0) + 1
            kit.text(screen, f'Chasing you now: {counts.get("bfs", 0)} by BFS field, {counts.get("greedy", 0)} by Greedy, '
                             f'{counts.get("astar", 0)} by A*', (x + 18, y0 + 140), P['text'], 'small', width=w - 36)
            # CSP
            y1 = y + 220
            heading('CONSTRAINTS', "this map's room CSP under 4 settings", y1)
            if lab.csp is None:
                kit.text(screen, 'solving...', (x + 18, y1 + 30), P['dim'], 'small')
            else:
                top = max([max(r.stats.assignments, 1) for _, r in lab.csp] + [10])
                gw = (w - 36) // 4
                base = y1 + 110
                for n, (name, res) in enumerate(lab.csp):
                    gx = x + 18 + n * gw
                    if name == 'FC+AC-3':
                        hl = pg.Rect(gx + 2, y1 + 20, gw - 4, 112)
                        pg.draw.rect(screen, P['panel_hi'], hl, border_radius=8)
                        pg.draw.rect(screen, P['cyan'], hl, width=1, border_radius=8)
                    for k, (value, color) in enumerate(((res.stats.assignments, P['cyan']),
                                                        (res.stats.backtracks, P['pink']))):
                        bh = max(2, int(64 * log10(value + 1) / log10(top + 1)))
                        bx = gx + gw // 2 - 20 + k * 22
                        pg.draw.rect(screen, color, (bx, base - bh, 18, bh), border_radius=3)
                        kit.text(screen, f'{value:,}', (bx + 9, base - bh - 3), P['muted'], 'tiny', 'midbottom')
                    kit.text(screen, name, (gx + gw // 2, base + 6), P['text'], 'h3', 'midtop')
                lx = x + 18
                pg.draw.rect(screen, P['cyan'], (lx, y1 + 142, 10, 10), border_radius=2)
                kit.text(screen, 'assignments', (lx + 16, y1 + 147), P['muted'], 'tiny', 'midleft')
                pg.draw.rect(screen, P['pink'], (lx + 104, y1 + 142, 10, 10), border_radius=2)
                kit.text(screen, 'backtracks  (log scale)', (lx + 120, y1 + 147), P['muted'], 'tiny', 'midleft')
            m = g.map
            kit.text(screen, f'{m.rows * m.cols} rooms, {len(m.nests)} nests, doors matched by the CSP, '
                             f'exit checked by A*', (x + 18, y1 + 166), P['text'], 'small', width=w - 36)
            # NN
            y2 = y + 410
            heading('PLAYER MODEL', 'chance to clear this map', y2)
            using_nn = director.controller.use_nn and director.controller.model is not None
            nn = director.predict(m, use_nn=True) if director.controller.model is not None else None
            rule = director.predict(m, use_nn=False)
            kit.gauge(screen, (x + w // 4 + 10, y2 + 62), 34, nn, P['cyan'],
                      '--' if nn is None else f'{nn * 100:.0f}%', 'NEURAL NET', active=using_nn)
            kit.gauge(screen, (x + 3 * w // 4 - 10, y2 + 62), 34, rule, P['gold'],
                      f'{rule * 100:.0f}%', 'RULE BASELINE', active=not using_nn)
            kit.text(screen, 'N switches which model picks the next sector', (x + w // 2, box.bottom - 16),
                     P['muted'], 'tiny', 'center')

        def draw_crosshair():
            mx, my = pg.mouse.get_pos()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                pg.draw.line(screen, (10, 10, 20), (mx + dx * 5, my + dy * 5), (mx + dx * 13, my + dy * 13), 4)
                pg.draw.line(screen, P['cyan'], (mx + dx * 5, my + dy * 5), (mx + dx * 12, my + dy * 12), 2)
            pg.draw.circle(screen, P['text'], (mx, my), 2)

        def draw_outcome(g):
            shade = pg.Surface((WIDTH, HEIGHT), pg.SRCALPHA)
            shade.fill((4, 6, 16, 170))
            screen.blit(shade, (0, 0))
            won = g.won
            kit.glow_text(screen, 'SECTOR CLEARED' if won else 'YOU DIED', (WIDTH // 2, HEIGHT // 2 - 70),
                          P['green'] if won else P['red'], 'logo', strength=200)
            accuracy = g.hits / g.shots if g.shots else 0
            kit.text(screen, f'{g.kills} kills   |   {accuracy:.0%} accuracy   |   {g.time:.1f}s',
                     (WIDTH // 2, HEIGHT // 2 + 4), P['text'], 'h2', 'center')
            c = director.controller
            kit.text(screen, f'The player model updated your challenge to {c.challenge:.2f}  '
                             f'({len(c.adapter.pending)}/10 results toward the next online update)',
                     (WIDTH // 2, HEIGHT // 2 + 36), P['muted'], 'small', 'center')
            kit.text(screen, 'Enter or click: next sector', (WIDTH // 2, HEIGHT // 2 + 78), P['cyan'], 'h3', 'center')

        reload_pressed = False
        while True:
            dt = min(clock.tick(60) / 1000, 0.05)
            t = now()
            g = state['game']
            finished = g.won or g.dead
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    return None
                if event.type == pg.MOUSEBUTTONDOWN and event.button == 1 and finished and not show_help:
                    new_sector(); g = state['game']; finished = False
                if event.type != pg.KEYDOWN:
                    continue
                if show_help:
                    if event.key in (pg.K_ESCAPE, pg.K_F1):
                        show_help = False
                    continue
                if event.key == pg.K_ESCAPE:
                    return 'title'
                elif event.key == pg.K_F1:
                    show_help = True
                elif event.key == pg.K_F11:
                    ui.toggle_fullscreen(pg)
                elif event.key == pg.K_TAB:
                    ai_lab = not ai_lab
                    if ai_lab:
                        lab.routes_at = -9.0
                elif event.key == pg.K_r:
                    reload_pressed = True
                elif event.key == pg.K_n:
                    c = director.controller
                    c.use_nn = not c.use_nn
                    toasts.push('NEXT SECTOR CHOSEN BY ' + ('THE NEURAL NET' if c.use_nn else 'THE RULE BASELINE'),
                                P['cyan'], 2.2, t)
                elif event.key in (pg.K_RETURN, pg.K_KP_ENTER) and finished:
                    new_sector(); g = state['game']; finished = False
            # ---- input -> simulation ------------------------------------------
            keys = pg.key.get_pressed()
            sx = (keys[pg.K_d] or keys[pg.K_RIGHT]) - (keys[pg.K_a] or keys[pg.K_LEFT])
            sy = (keys[pg.K_s] or keys[pg.K_DOWN]) - (keys[pg.K_w] or keys[pg.K_UP])
            move = (sx + sy, sy - sx)                 # screen directions -> world directions
            cam = state['cam']
            psx, psy = iso(g.player.x, g.player.y)
            mx, my = pg.mouse.get_pos()
            aim = from_screen(mx / 2 + cam[0] - psx, my / 2 + cam[1] - psy)
            shoot = bool(pg.mouse.get_pressed()[0] or keys[pg.K_SPACE])
            if not show_help and not finished:
                handle(g.update(dt, move=move, aim=aim, shoot=shoot, reload=reload_pressed))
            reload_pressed = False
            if ai_lab and t - lab.routes_at > 1.0:
                lab.refresh(g, t, state['choice'], with_csp=True)
            # ---- effects ---------------------------------------------------------
            for part in state['particles']:
                part[0] += part[3] * dt; part[1] += part[4] * dt
                part[2] += part[5] * dt; part[5] -= 260 * dt
                if part[2] < 0:
                    part[2] = 0; part[5] *= -0.3; part[3] *= 0.5; part[4] *= 0.5
                part[6] -= dt
            state['particles'] = [q for q in state['particles'] if q[6] > 0][-500:]
            for f in state['flashes']:
                f[2] -= dt
            state['flashes'] = [f for f in state['flashes'] if f[2] > 0]
            for a in state['alerts']:
                a[2] -= dt
            state['alerts'] = [a for a in state['alerts'] if a[2] > 0]
            state['muzzle'] = max(0.0, state['muzzle'] - dt)
            state['hurt'] = max(0.0, state['hurt'] - dt)
            state['shake'] = max(0.0, state['shake'] - dt * 18)
            # camera: follow the player and lean toward the cursor
            psx, psy = iso(g.player.x, g.player.y)
            lean_x = max(-60, min(60, (mx / 2 - LOW_W / 2) * 0.25))
            lean_y = max(-40, min(40, (my / 2 - LOW_H / 2) * 0.25))
            view_w = LOW_W - (LAB_W // 2 if ai_lab else 0)
            target = (psx - view_w / 2 + lean_x, psy - LOW_H / 2 - 8 + lean_y)
            cam[0] += (target[0] - cam[0]) * min(1.0, dt * 8)
            cam[1] += (target[1] - cam[1]) * min(1.0, dt * 8)
            shake = state['shake']
            view = (int(cam[0] + (rng.random() - 0.5) * shake), int(cam[1] + (rng.random() - 0.5) * shake))
            # ---- draw ------------------------------------------------------------------
            draw_world(t, view)
            draw_lighting(t, view)
            draw_unlit(t, view)
            pg.transform.scale(world, (WIDTH, HEIGHT), screen)
            if state['hurt'] > 0:
                red = pg.Surface((WIDTH, HEIGHT), pg.SRCALPHA)
                red.fill((255, 40, 70, int(140 * state['hurt'])))
                screen.blit(red, (0, 0))
            draw_hud(t)
            if ai_lab:
                draw_lab(t)
            toasts.draw(kit, screen, (WIDTH - (LAB_W if ai_lab else 0)) // 2, 84, t)
            if finished:
                draw_outcome(g)
            if show_help:
                kit.overlay_help(screen, 'CONTROLS  //  EXPEDITION', EXPEDITION_HELP)
            draw_crosshair()
            pg.display.flip()
    finally:
        try:
            pg.mouse.set_visible(True)
        except pg.error:
            pass
        pg.quit()


if __name__ == '__main__':
    run_expedition()
