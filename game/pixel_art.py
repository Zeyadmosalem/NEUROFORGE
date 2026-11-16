"""Pixel art for Expedition, built in code.

The hero's head is the user's flat white skull logo (assets/hero_skull.png)
shrunk to 16 px and reduced to a 3-colour palette with a dark outline. Every
other sprite (aliens, nest, pickups) is an original character grid drawn here.
Isometric blocks are drawn as a lit top diamond plus two shaded faces.
"""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERO_LOGO = ROOT / 'assets' / 'hero_skull.png'
TILE_W, TILE_H = 32, 16             # isometric diamond size in low-res pixels

INK = (18, 14, 26)                  # outline colour
COLORS = {
    # hero
    'W': (238, 238, 230), 'w': (176, 178, 192), 'x': (26, 22, 34),
    'k': (36, 38, 62), 'K': (64, 70, 108), 'r': (222, 52, 66), 'R': (255, 110, 110),
    'h': (172, 176, 190), 'b': (22, 20, 30), 'g': (120, 126, 146), 'G': (210, 214, 226),
    # aliens
    'd': (34, 104, 52), 'm': (78, 186, 84), 'M': (160, 246, 122), 'e': (250, 250, 236), 'p': (20, 16, 24),
    'v': (74, 34, 92), 'u': (150, 70, 190), 'U': (226, 136, 255), 'y': (255, 214, 92),
    'o': (110, 46, 24), 'n': (214, 104, 44), 'N': (255, 172, 82), 's': (64, 58, 70), 'S': (124, 118, 130),
    # pickups
    'A': (255, 206, 92), 'a': (156, 112, 34), 'c': (240, 240, 240), 'C': (230, 54, 70),
}

FALLBACK_SKULL = [
    "....xxxxxxx..x..",
    "...xWWWWWWWxxWx.",
    "..xWWWWWWWWWWWx.",
    ".xWWWWWWWWWWWWx.",
    ".xWWxxxWWWWxxWx.",
    ".xWxxxxxWWxxxxWx",
    ".xWxxxxxWWxxxxWx",
    ".xWWxxxWWWWxxWWx",
    ".xWWWWWWxxWWWWWx",
    "..xWWWWxxxxWWWx.",
    "...xWWWWWWWWWx..",
    "...xWxWxWxWxWx..",
    "....xxxxxxxxx...",
]

BODY = [
    "....KkkkkkkK....",
    "...KkkkrrkkkK...",
    "..hKkkrrrrkkKh..",
    "..hKkkkrrkkkKh..",
    "...KkkkkkkkkK...",
    "....kkKKKKkk....",
]
LEGS = [
    ["....kkk..kkk....",
     "....kkk..kkk....",
     "....kk....kk....",
     "...bbb....bbb..."],
    ["...kkk....kkk...",
     "..kkk......kk...",
     "..kk.......kk...",
     ".bbb.......bbb.."],
    ["....kkk..kkk....",
     "....kkk..kkk....",
     "....kk....kk....",
     "...bbb....bbb..."],
    ["...kkk....kkk...",
     "...kk......kkk..",
     "...kk.......kk..",
     "..bbb.......bbb."],
]
GUN = [
    "..gggggggG",
    "GGgggggggg",
    "gg.gg.....",
    "g.........",
]

ALIENS = {
    'glob': [
        ["....dddddd....",
         "..ddmmmmmmdd..",
         ".dmMMmmmmmmmd.",
         ".dmMmeemmeemmd",
         "dmmmeppmmeppmd",
         "dmmmmmmmmmmmmd",
         "dmmmmpppppmmmd",
         ".dmmmmmmmmmmd.",
         ".ddmmmmmmmmdd.",
         "dd.dddddddd.dd"],
        ["..............",
         "....dddddd....",
         "..ddmmmmmmdd..",
         ".dmMMmeemmeemd",
         "dmmMmeppmmeppd",
         "dmmmmmmmmmmmmd",
         "dmmmmpppppmmmd",
         "dmmmmmmmmmmmmd",
         ".ddmmmmmmmmdd.",
         "ddddddddddddd."],
    ],
    'skitter': [
        ["u...........u.",
         ".u..vvvvvv..u.",
         "..uvuuuuuuvu..",
         "...vUUyuuyUv..",
         "uuuvuuuuuuuvuu",
         "...vuuuuuuuv..",
         "..u.vvvvvv.u..",
         ".u..........u."],
        [".u.........u..",
         "..u.vvvvvv.u..",
         "...vuuuuuuvu..",
         "uuuvUUyuuyUvuu",
         "...vuuuuuuuv..",
         "..uvuuuuuuvu..",
         ".u..vvvvvv..u.",
         "u...........u."],
    ],
    'brute': [
        ["....oooooooo......",
         "...onnnnnnnnoo....",
         "..onNNnnnnnnnno...",
         ".onNnnyyonnyynno..",
         ".onnnnnoonnnnnnno.",
         "oonnnnsSSSSsnnnnoo",
         "onnnnnsSSSSsnnnnno",
         "onnnnnnnnnnnnnnnno",
         ".onnnnnnnnnnnnnno.",
         "..oonnnooonnnoo...",
         "..ooo.oo..oo.ooo..",
         ".oooo.......oooo.."],
        ["....oooooooo......",
         "...onnnnnnnnoo....",
         "..onNNnnnnnnnno...",
         ".onNnnyyonnyynno..",
         ".onnnnnoonnnnnnno.",
         "oonnnnsSSSSsnnnnoo",
         "onnnnnsSSSSsnnnnno",
         "onnnnnnnnnnnnnnnno",
         ".onnnnnnnnnnnnnno.",
         "...oonnnooonnnoo..",
         "...ooo.oo..oo.ooo.",
         "..oooo.......oooo."],
    ],
}
NEST = [
    "......vvvvvv......",
    "....vvuuuuuuvv....",
    "...vuuUUuuuuuuv...",
    "..vuuUyyuuuuyuuv..",
    "..vuuuyyuuuuuuuv..",
    ".vuuuuuuuuyyuuuuv.",
    ".vuuuuuuuuyyuuuuv.",
    "vvuuuuuuuuuuuuuuvv",
    "vvvvvuuuuuuuuvvvvv",
    ".vvvvvvvvvvvvvvvv.",
]
AMMO = [
    "..AAAAAA..",
    ".AaAaAaAA.",
    "AAAAAAAAAA",
    "AaaaaaaaaA",
    "AaAAAAAAaA",
    "AaaaaaaaaA",
    "AAAAAAAAAA",
]
HEALTH = [
    ".cccccccc.",
    "cccCCCcccc",
    "cccCCCcccc",
    "cCCCCCCCcc",
    "cCCCCCCCcc",
    "cccCCCcccc",
    "cccCCCcccc",
    ".cccccccc.",
]


def grid_sprite(pg, rows, colors=COLORS, outline=False):
    width, height = len(rows[0]), len(rows)
    pad = 1 if outline else 0
    surf = pg.Surface((width + 2 * pad, height + 2 * pad), pg.SRCALPHA)
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch != '.':
                surf.set_at((x + pad, y + pad), colors[ch])
    if outline:
        add_outline(pg, surf)
    return surf


def add_outline(pg, surf, color=INK):
    """Dark one-pixel rim around every opaque pixel (keeps sprites readable)."""
    w, h = surf.get_size()
    solid = {(x, y) for y in range(h) for x in range(w) if surf.get_at((x, y)).a > 0}
    for x, y in solid:
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in solid:
                surf.set_at((nx, ny), color)
    return surf


def skull_head(pg, size=16):
    """The user's logo reduced to a crisp pixel skull (white, shade, black)."""
    try:
        logo = pg.image.load(str(HERO_LOGO)).convert_alpha()
    except (FileNotFoundError, pg.error):
        return grid_sprite(pg, FALLBACK_SKULL, outline=True)
    logo = logo.subsurface(logo.get_bounding_rect(min_alpha=40)).copy()
    w, h = logo.get_size()
    scale = size / max(w, h)
    small = pg.transform.smoothscale(logo, (max(1, round(w * scale)), max(1, round(h * scale))))
    sw, sh = small.get_size()
    head = pg.Surface((sw + 2, sh + 2), pg.SRCALPHA)
    for y in range(sh):
        for x in range(sw):
            c = small.get_at((x, y))
            if c.a < 110:
                continue
            light = (c.r + c.g + c.b) / 3
            color = COLORS['W'] if light > 175 else COLORS['w'] if light > 90 else COLORS['x']
            head.set_at((x + 1, y + 1), color)
    return add_outline(pg, head)


class Sprites:
    """Every Expedition sprite, built once per Pygame session."""

    def __init__(self, pg):
        self.pg = pg
        head = skull_head(pg)
        body = grid_sprite(pg, BODY)
        self.hero = {}
        for frame, legs in enumerate(LEGS):
            leg = grid_sprite(pg, legs)
            hw, hh = head.get_size()
            width = max(hw, 16) + 2
            height = hh + len(BODY) + len(legs) - 1
            canvas = pg.Surface((width, height), pg.SRCALPHA)
            canvas.blit(leg, ((width - 16) // 2, hh + len(BODY) - 1))
            canvas.blit(body, ((width - 16) // 2, hh - 1))
            canvas.blit(head, ((width - hw) // 2, 0))
            add_outline(pg, canvas)
            self.hero[('r', frame)] = canvas
            self.hero[('l', frame)] = pg.transform.flip(canvas, True, False)
        gun = grid_sprite(pg, GUN, outline=True)
        self.gun = {'r': gun, 'l': pg.transform.flip(gun, True, False)}
        self.aliens = {}
        for kind, frames in ALIENS.items():
            for n, rows in enumerate(frames):
                sprite = grid_sprite(pg, rows, outline=True)
                flash = sprite.copy()
                flash.fill((255, 255, 255, 0), special_flags=pg.BLEND_RGBA_MAX)
                for facing, img in (('r', sprite), ('l', pg.transform.flip(sprite, True, False))):
                    self.aliens[(kind, n, facing)] = img
                for facing, img in (('r', flash), ('l', pg.transform.flip(flash, True, False))):
                    self.aliens[(kind, n, facing, 'hit')] = img
        self.nest = grid_sprite(pg, NEST, outline=True)
        self.ammo = grid_sprite(pg, AMMO, outline=True)
        self.health = grid_sprite(pg, HEALTH, outline=True)


def iso_block(pg, height, top, left, right, trim=None, alpha=255, rivets=True):
    """A raised isometric tile: a lit top diamond and two shaded faces.

    The ground diamond's centre sits at (16, 8 + height) in the returned surface.
    """
    w, h = TILE_W, TILE_H
    surf = pg.Surface((w + 1, h + height + 1), pg.SRCALPHA)
    top_pts = [(w // 2, 0), (w, h // 2), (w // 2, h), (0, h // 2)]
    left_pts = [(0, h // 2), (w // 2, h), (w // 2, h + height), (0, h // 2 + height)]
    right_pts = [(w // 2, h), (w, h // 2), (w, h // 2 + height), (w // 2, h + height)]
    pg.draw.polygon(surf, left, left_pts)
    pg.draw.polygon(surf, right, right_pts)
    pg.draw.polygon(surf, top, top_pts)
    if trim is not None and height >= 10:
        # a lit seam halfway up the faces (sci-fi bulkhead panel line)
        y = h // 2 + height // 2
        pg.draw.line(surf, trim, (0, y), (w // 2, y + h // 2))
        pg.draw.line(surf, trim, (w // 2, y + h // 2), (w, y))
    if rivets and height >= 14:
        for x, y in ((4, h // 2 + 4), (w // 2 - 4, h + 2), (w // 2 + 4, h + 2), (w - 4, h // 2 + 4)):
            surf.set_at((x, y), trim or top)
    edge = tuple(min(255, c + 34) for c in top)
    pg.draw.line(surf, edge, top_pts[3], top_pts[0])
    pg.draw.line(surf, edge, top_pts[0], top_pts[1])
    pg.draw.lines(surf, INK, True, [top_pts[3], top_pts[0], top_pts[1], right_pts[2], right_pts[3],
                                   left_pts[3]])
    if alpha < 255:
        surf.set_alpha(alpha)
    return surf


def radial_light(pg, radius, color, strength=1.0):
    """Soft round light texture for additive blending onto a light map."""
    surf = pg.Surface((radius * 2, radius * 2))
    steps = 18
    for i in range(steps, 0, -1):
        k = (1 - i / steps) ** 1.6 * strength
        c = tuple(min(255, int(v * k)) for v in color)
        pg.draw.circle(surf, c, (radius, radius), int(radius * i / steps))
    return surf


def cone_light(pg, length, spread_deg, color):
    """Flashlight cone pointing right with its tip at the surface centre, so a
    rotated copy blitted centred on the player still starts at the player."""
    from math import cos, sin, radians
    surf = pg.Surface((length * 2, length * 2))
    steps = 22
    half = radians(spread_deg) / 2
    for i in range(steps, 0, -1):
        r = length * i / steps
        k = (1 - i / steps) ** 0.9
        c = tuple(min(255, int(v * k)) for v in color)
        pts = [(length, length)]
        for j in range(9):
            a = -half + 2 * half * j / 8
            pts.append((length + r * cos(a), length + r * sin(a)))
        pg.draw.polygon(surf, c, pts)
    return surf
