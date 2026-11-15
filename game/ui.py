"""Shared look and drawing helpers for every NEUROFORGE screen.

Pure presentation: nothing here touches physics, search, the CSP or the NN.
A Kit is created per Pygame session, because fonts die with pygame.quit().
"""
from __future__ import annotations
from math import sin, cos, pi
from random import Random

WIDTH, HEIGHT = 1280, 720   # logical size of every screen; the window scales it

PALETTE = {
    'sky_top': (8, 10, 26), 'sky_bottom': (34, 26, 70),
    'panel': (16, 20, 44), 'panel_hi': (28, 34, 70), 'line': (60, 72, 122),
    'text': (234, 242, 255), 'muted': (150, 162, 196), 'dim': (96, 108, 146),
    'cyan': (98, 230, 246), 'pink': (255, 104, 172), 'gold': (255, 206, 110),
    'green': (122, 236, 172), 'purple': (172, 132, 252), 'red': (255, 86, 116),
    'tile': (40, 40, 80), 'tile_top': (72, 210, 188), 'tile_edge': (30, 118, 124),
    'tile_line': (62, 60, 104), 'spike': (248, 92, 148), 'spike_hi': (255, 196, 214),
}
ALGO_COLORS = {'bfs': (89, 218, 236), 'dfs': (181, 121, 248),
               'astar': (111, 237, 170), 'greedy': (255, 117, 164)}
ALGO_LABELS = {'bfs': 'BFS', 'dfs': 'DFS', 'astar': 'A*', 'greedy': 'GREEDY'}

MANUAL_HELP = [
    ('MOVE', [('A / D', 'Walk left / right (arrows work too)'),
              ('Space / W', 'Jump'), ('R', 'Restart this attempt')]),
    ('AI LAB', [('Tab', 'Open / close the AI Lab dashboard'),
                ('1 2 3 4', 'Show BFS, DFS, A*, Greedy on this level')]),
    ('CONSTRAINTS (CSP)', [('F', 'Forward Checking on / off'), ('C', 'AC-3 on / off'),
                           ('G', 'Generate a new level with those settings')]),
    ('NEURAL NETWORK', [('N', 'Neural network vs rule baseline'),
                        ('X', 'Memory glitch: erase part of the online adapter')]),
    ('WINDOW', [('V', 'Four-agent arena'), ('F11', 'Fullscreen'),
                ('F1 / H', 'This help'), ('Esc', 'Back to the menu')]),
]
ARENA_HELP = [
    ('SIMULATION', [('Space', 'Pause / resume'), ('+ / -', 'Faster / slower'),
                    ('R', 'Restart with four new worlds'), ('H', 'Show / hide planned routes')]),
    ('WORLD SURPRISES', [('E', 'Surprise every agent now'),
                         ('U', 'Random surprises on / off')]),
    ('ONLINE LEARNING', [('X', 'Memory glitch now'), ('D', 'Scheduled glitches on / off')]),
    ('WINDOW', [('S', 'Save a screenshot to results/'), ('M', 'Manual mode'),
                ('F11', 'Fullscreen'), ('F1', 'This help'), ('Esc', 'Back to the menu')]),
]


def lerp(a, b, t):
    return a + (b - a) * t


def lerp_cell(a, b, alpha):
    """Blend two grid positions; snap when they are not one continuous move."""
    if abs(b[0] - a[0]) > 2 or abs(b[1] - a[1]) > 4:
        return b
    return a[0] + (b[0] - a[0]) * alpha, a[1] + (b[1] - a[1]) * alpha


def mix(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(lerp(a, b, t)) for a, b in zip(c1, c2))


def open_window(pg, caption):
    """A 1280x720 logical screen in a resizable, sharply scaled window."""
    pg.display.set_caption(caption)
    attempts = ((pg.SCALED | pg.RESIZABLE, {'vsync': 1}), (pg.SCALED | pg.RESIZABLE, {}), (0, {}))
    for flags, extra in attempts:
        try:
            return pg.display.set_mode((WIDTH, HEIGHT), flags, **extra)
        except (pg.error, TypeError):
            continue
    return pg.display.set_mode((WIDTH, HEIGHT))


def toggle_fullscreen(pg):
    try:
        pg.display.toggle_fullscreen()
    except pg.error:
        pass


class Kit:
    """Fonts, cached text and the small widgets every screen shares."""

    def __init__(self, pg):
        self.pg = pg
        display = 'bahnschrift,segoeui,dejavusans,arial'
        body = 'segoeui,dejavusans,arial'
        mono = 'consolas,dejavusansmono,couriernew'
        font = pg.font.SysFont
        self.fonts = {
            'logo': font(display, 76, bold=True), 'title': font(display, 30, bold=True),
            'h1': font(display, 22, bold=True), 'h2': font(display, 17, bold=True),
            'h3': font(display, 14, bold=True), 'body': font(body, 15),
            'bodyb': font(body, 15, bold=True), 'small': font(body, 13),
            'tiny': font(body, 11), 'num': font(display, 28, bold=True),
            'numsm': font(display, 19, bold=True), 'mono': font(mono, 13),
            'key': font(display, 13, bold=True), 'menu': font(display, 24, bold=True),
        }
        self._text = {}
        self._glow = {}

    # ---- text -----------------------------------------------------------
    def render(self, s, color, face='body'):
        key = (s, color, face)
        surf = self._text.get(key)
        if surf is None:
            if len(self._text) > 4000:
                self._text.clear()
            surf = self.fonts[face].render(s, True, color)
            self._text[key] = surf
        return surf

    def fit(self, s, face, width):
        """Shorten `s` with an ellipsis so it renders no wider than `width`."""
        font = self.fonts[face]
        if font.size(s)[0] <= width:
            return s
        while s and font.size(s + '...')[0] > width:
            s = s[:-1]
        return s + '...'

    def text(self, dest, s, pos, color=PALETTE['text'], face='body', anchor='topleft',
             alpha=None, width=None):
        s = str(s)
        if width is not None:
            s = self.fit(s, face, width)
        surf = self.render(s, color, face)
        if alpha is not None and alpha < 255:
            surf = surf.copy()
            surf.set_alpha(max(0, int(alpha)))
        rect = surf.get_rect(**{anchor: (int(pos[0]), int(pos[1]))})
        dest.blit(surf, rect)
        return rect

    def glow_text(self, dest, s, pos, color, face='logo', anchor='center', strength=170):
        """Text with a soft neon halo (the halo is blurred once and cached)."""
        pg = self.pg
        key = (s, color, face)
        glow = self._glow.get(key)
        if glow is None:
            base = self.render(s, color, face)
            pad = 24
            glow = pg.Surface((base.get_width() + 2 * pad, base.get_height() + 2 * pad), pg.SRCALPHA)
            glow.blit(base, (pad, pad))
            try:
                glow = pg.transform.gaussian_blur(glow, 12)
            except (AttributeError, ValueError, pg.error):
                small = pg.transform.smoothscale(glow, (max(1, glow.get_width() // 6),
                                                        max(1, glow.get_height() // 6)))
                glow = pg.transform.smoothscale(small, glow.get_size())
            self._glow[key] = glow
        rect = self.render(s, color, face).get_rect(**{anchor: (int(pos[0]), int(pos[1]))})
        halo = glow.copy()
        halo.set_alpha(strength)
        dest.blit(halo, halo.get_rect(center=rect.center))
        dest.blit(self.render(s, color, face), rect)
        return rect

    # ---- shapes ---------------------------------------------------------
    def panel(self, dest, rect, fill=PALETTE['panel'], border=PALETTE['line'], radius=14,
              alpha=235, border_width=1):
        pg = self.pg
        rect = pg.Rect(rect)
        if alpha is None or alpha >= 255:
            pg.draw.rect(dest, fill, rect, border_radius=radius)
        else:
            layer = pg.Surface(rect.size, pg.SRCALPHA)
            pg.draw.rect(layer, (*fill, int(alpha)), layer.get_rect(), border_radius=radius)
            dest.blit(layer, rect.topleft)
        if border is not None and border_width:
            pg.draw.rect(dest, border, rect, width=border_width, border_radius=radius)
        return rect

    def bar(self, dest, rect, frac, color, track=PALETTE['panel_hi'], radius=4):
        pg = self.pg
        rect = pg.Rect(rect)
        pg.draw.rect(dest, track, rect, border_radius=radius)
        frac = max(0.0, min(1.0, float(frac)))
        if frac > 0:
            fill = rect.copy()
            fill.width = max(rect.height, int(rect.width * frac))
            pg.draw.rect(dest, color, fill, border_radius=radius)

    def chip(self, dest, x, y, key, label, key_color=PALETTE['text'], label_color=PALETTE['muted']):
        """A keyboard key cap followed by its meaning. Returns the next free x."""
        pg = self.pg
        ksurf = self.render(key, key_color, 'key')
        box = pg.Rect(x, y, ksurf.get_width() + 14, 22)
        pg.draw.rect(dest, PALETTE['panel_hi'], box, border_radius=6)
        pg.draw.rect(dest, PALETTE['line'], box, width=1, border_radius=6)
        dest.blit(ksurf, ksurf.get_rect(center=box.center))
        x = box.right + 7
        if label:
            lsurf = self.render(label, label_color, 'small')
            dest.blit(lsurf, lsurf.get_rect(midleft=(x, box.centery)))
            x += lsurf.get_width()
        return x + 16

    def chips_width(self, items):
        width = 0
        for key, label in items:
            width += self.fonts['key'].size(key)[0] + 14 + 7 + 16
            if label:
                width += self.fonts['small'].size(label)[0]
        return width - 16

    def chips_row(self, dest, items, y, center_x=WIDTH // 2):
        x = center_x - self.chips_width(items) // 2
        for key, label in items:
            x = self.chip(dest, x, y, key, label)

    def pill(self, dest, center, s, color, face='key', fill=None, pad=10, height=22):
        pg = self.pg
        surf = self.render(s, color if fill is None else PALETTE['sky_top'], face)
        rect = pg.Rect(0, 0, surf.get_width() + 2 * pad, height)
        rect.center = center
        if fill is None:
            pg.draw.rect(dest, PALETTE['panel_hi'], rect, border_radius=height // 2)
            pg.draw.rect(dest, color, rect, width=1, border_radius=height // 2)
        else:
            pg.draw.rect(dest, fill, rect, border_radius=height // 2)
        dest.blit(surf, surf.get_rect(center=rect.center))
        return rect

    def gauge(self, dest, center, radius, frac, color, value_text, label, active=False):
        """A 270-degree dial for a probability."""
        pg = self.pg
        cx, cy = center
        rect = pg.Rect(cx - radius, cy - radius, 2 * radius, 2 * radius)
        start, sweep = -pi * 0.25, pi * 1.5     # opening at the bottom
        pg.draw.arc(dest, PALETTE['panel_hi'], rect, start, start + sweep, 9)
        if frac is not None:
            frac = max(0.0, min(1.0, frac))
            if frac > 0.004:
                pg.draw.arc(dest, color if active else PALETTE['dim'], rect,
                            start + sweep * (1 - frac), start + sweep, 9)
        self.text(dest, value_text, (cx, cy - 4), color if active else PALETTE['muted'], 'numsm', 'center')
        self.text(dest, label, (cx, cy + radius - 2), PALETTE['text'] if active else PALETTE['muted'],
                  'h3', 'center')

    def overlay_help(self, dest, title, sections, footer='F1 or Esc to close'):
        """Full-screen key guide drawn on top of everything else."""
        pg = self.pg
        shade = pg.Surface((WIDTH, HEIGHT), pg.SRCALPHA)
        shade.fill((4, 6, 18, 205))
        dest.blit(shade, (0, 0))
        box = pg.Rect(0, 0, 980, 560)
        box.center = (WIDTH // 2, HEIGHT // 2)
        self.panel(dest, box, alpha=245, radius=20)
        self.glow_text(dest, title, (box.centerx, box.top + 40), PALETTE['cyan'], 'title', strength=120)
        columns = 2 if len(sections) > 2 else len(sections)
        col_w = (box.width - 80) // columns
        heights = [0] * columns
        for heading, rows in sections:
            col = heights.index(min(heights))
            x = box.left + 40 + col * col_w
            y = box.top + 86 + heights[col]
            self.text(dest, heading, (x, y), PALETTE['gold'], 'h3')
            y += 26
            for key, desc in rows:
                self.chip(dest, x, y, key, '')
                self.text(dest, desc, (x + 118, y + 11), PALETTE['text'], 'small', 'midleft')
                y += 30
            heights[col] = y - (box.top + 86) + 16
        self.text(dest, footer, (box.centerx, box.bottom - 26), PALETTE['muted'], 'small', 'center')


class Backdrop:
    """Night sky with twinkling stars, a planet and three parallax ridges."""

    def __init__(self, pg, seed=7, horizon=HEIGHT, planet=None):
        self.pg = pg
        self.horizon = horizon
        self.planet = planet      # fixed (x, y), or None to drift with the camera
        rng = Random(seed)
        self.sky = pg.Surface((WIDTH, HEIGHT)).convert()
        top, bottom = PALETTE['sky_top'], PALETTE['sky_bottom']
        for y in range(HEIGHT):
            pg.draw.line(self.sky, mix(top, bottom, (y / HEIGHT) ** 1.4), (0, y), (WIDTH, y))
        self.stars = [(rng.random() * WIDTH * 2, rng.random() * HEIGHT * 0.62,
                       rng.random() * 1.5 + 0.5, rng.random() * 6.28) for _ in range(150)]
        self.ridges = [((26, 30, 66), 0.06, 0.62, 0.0), ((32, 36, 78), 0.12, 0.72, 1.7),
                       ((40, 46, 92), 0.2, 0.82, 3.1)]

    def draw(self, dest, t, camera=0.0):
        pg = self.pg
        dest.blit(self.sky, (0, 0))
        for x, y, r, ph in self.stars:
            tw = 0.55 + 0.45 * sin(t * 1.3 + ph)
            sx = int((x - camera * 0.03) % (WIDTH + 20)) - 10
            pg.draw.circle(dest, mix((40, 52, 96), (210, 226, 255), tw), (sx, int(y)), max(1, int(r)))
        if self.planet is None:
            px, py = int(1010 - camera * 0.02) % (WIDTH + 400) - 120, 150
        else:
            px, py = self.planet
        pg.draw.circle(dest, (66, 46, 112), (px, py), 74)
        pg.draw.circle(dest, (128, 82, 168), (px, py), 62)
        pg.draw.circle(dest, (176, 104, 176), (px - 14, py - 10), 44)
        for color, rate, level, phase in self.ridges:
            base = self.horizon * level
            pts = [(0, HEIGHT)]
            off = camera * rate
            for x in range(0, WIDTH + 41, 40):
                u = (x + off) * 0.006 + phase
                y = base - abs(sin(u)) * 70 - abs(sin(u * 2.7)) * 22
                pts.append((x, int(y)))
            pts.append((WIDTH, HEIGHT))
            pg.draw.polygon(dest, color, pts)


class Particles:
    """Short-lived sparks and dust in world pixel coordinates."""

    def __init__(self, seed=3):
        self.items = []
        self.rng = Random(seed)

    def emit(self, x, y, color, count=10, speed=160, life=0.6, gravity=420, size=3, spread=pi * 2,
             direction=-pi / 2):
        rng = self.rng
        for _ in range(count):
            a = direction + (rng.random() - 0.5) * spread
            v = speed * (0.4 + rng.random() * 0.8)
            self.items.append([x, y, cos(a) * v, sin(a) * v, life * (0.6 + rng.random() * 0.6),
                               life, color, size, gravity])

    def update(self, dt):
        keep = []
        for p in self.items:
            p[4] -= dt
            if p[4] > 0:
                p[0] += p[2] * dt
                p[1] += p[3] * dt
                p[3] += p[8] * dt
                keep.append(p)
        self.items = keep

    def draw(self, pg, dest, ox=0.0, oy=0.0):
        for x, y, _, _, left, life, color, size, _ in self.items:
            k = max(0.0, left / life)
            r = max(1, int(size * (0.4 + 0.6 * k)))
            pg.draw.circle(dest, mix((20, 20, 40), color, 0.3 + 0.7 * k), (int(x + ox), int(y + oy)), r)


class Toasts:
    """Short messages that slide in under the header and fade away."""

    def __init__(self):
        self.items = []

    def push(self, message, color=PALETTE['gold'], seconds=2.4, now=0.0):
        self.items = [i for i in self.items if i[0] != message][-2:]
        self.items.append([message, color, now, now + seconds])

    def draw(self, kit, dest, center_x, y, now):
        self.items = [i for i in self.items if i[3] > now]
        for n, (message, color, start, end) in enumerate(reversed(self.items)):
            age, left = now - start, end - now
            alpha = 255 * min(1.0, age * 6, left * 3)
            slide = max(0.0, 1 - age * 6) * -12
            surf = kit.render(message, color, 'h2')
            rect = surf.get_rect(midtop=(center_x, y + n * 40 + slide))
            box = rect.inflate(34, 16)
            layer = kit.pg.Surface(box.size, kit.pg.SRCALPHA)
            kit.pg.draw.rect(layer, (*PALETTE['panel'], int(alpha * 0.92)), layer.get_rect(),
                             border_radius=box.height // 2)
            kit.pg.draw.rect(layer, (*color, int(alpha * 0.8)), layer.get_rect(), width=1,
                             border_radius=box.height // 2)
            dest.blit(layer, box.topleft)
            if alpha < 255:
                surf = surf.copy()
                surf.set_alpha(int(alpha))
            dest.blit(surf, rect)
