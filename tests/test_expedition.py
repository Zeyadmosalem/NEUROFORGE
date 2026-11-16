"""Expedition mode: grid search, room CSP, simulation and the NN director (no Pygame)."""
import pytest
from search.algorithms import solve_grid, bfs_distance_field

GRID = ["......",
        ".####.",
        ".#..#.",
        ".#..#.",
        ".####.",
        "......"]          # a walled-off 2x2 pocket at (2..3, 2..3)


def open_cell(x, y):
    return 0 <= y < len(GRID) and 0 <= x < len(GRID[0]) and GRID[y][x] == '.'


def test_grid_search_known_shortest_path():
    for algo in ('bfs', 'dfs', 'astar', 'greedy'):
        r = solve_grid(open_cell, (0, 0), (5, 5), algo)
        assert r.found and r.path[0] == (0, 0) and r.path[-1] == (5, 5)
        assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip(r.path, r.path[1:]))
        assert all(open_cell(*cell) for cell in r.path)
    assert solve_grid(open_cell, (0, 0), (5, 5), 'bfs').cost == 10
    assert solve_grid(open_cell, (0, 0), (5, 5), 'astar').cost == 10


def test_grid_search_walled_off_goal_says_no_solution():
    r = solve_grid(open_cell, (0, 0), (2, 2), 'bfs')
    assert not r.found and r.reason == 'No solution'


@pytest.mark.parametrize('fc,ac', [(False, False), (True, False), (False, True), (True, True)])
def test_room_csp_modes_give_consistent_doors(fc, ac):
    from csp.rooms import RoomCSP
    r = RoomCSP(4, 4, 0.5, seed=7).solve(fc, ac)
    rooms = r.chunks
    assert rooms is not None and r.stats.assignments >= 16
    grid = [rooms[i * 4:(i + 1) * 4] for i in range(4)]
    for y in range(4):
        for x in range(4):
            d = grid[y][x].doors
            assert ('W' in d) == (x > 0 and 'E' in grid[y][x - 1].doors)
            assert ('N' in d) == (y > 0 and 'S' in grid[y - 1][x].doors)
            assert d and (x < 3 or 'E' not in d) and (y < 3 or 'S' not in d)
            if x < 3:
                assert not (grid[y][x].template.kind == 'nest' and grid[y][x + 1].template.kind == 'nest')
    assert grid[0][0].template.kind == 'start' and grid[3][3].template.kind == 'exit'


def test_room_csp_impossible_says_no_solution():
    from csp.rooms import RoomCSP
    r = RoomCSP(impossible=True).solve()
    assert r.chunks is None and r.reason == 'No solution'


@pytest.mark.parametrize('seed', [1, 2, 3, 4, 5])
def test_generated_maps_are_reachable(seed):
    from csp.rooms import generate_map
    result = generate_map(seed=seed, target=0.18 * seed)
    m = result.map
    assert m is not None and result.search.found
    assert m.width == 40 and m.height == 40
    assert m.tile(*m.start) == 'S' and m.tile(*m.exit) == 'E'
    assert solve_grid(m.passable, m.start, m.exit, 'bfs').found


def test_aliens_close_the_distance():
    from csp.rooms import generate_map
    from integration.expedition import Expedition
    m = generate_map(seed=3, target=0.4).map
    ex = Expedition(m, seed=3, spawning=False)
    route = solve_grid(m.passable, m.start, m.exit, 'bfs').path
    for kind, cell in zip(('glob', 'skitter', 'brute'), (route[7], route[9], route[11])):
        ex.spawn(kind, cell[0] + 0.5, cell[1] + 0.5)
    before = [abs(a.x - ex.player.x) + abs(a.y - ex.player.y) for a in ex.aliens]
    for _ in range(45):
        ex.update(1 / 30)
    after = [abs(a.x - ex.player.x) + abs(a.y - ex.player.y) for a in ex.aliens]
    assert len(after) == 3 and all(b < a for a, b in zip(before, after))
    assert {a.kind: a.algorithm for a in ex.aliens} == {'glob': 'bfs', 'skitter': 'greedy', 'brute': 'astar'}


def test_player_following_route_reaches_exit():
    from csp.rooms import generate_map
    from integration.expedition import Expedition
    m = generate_map(seed=5, target=0.3).map
    ex = Expedition(m, seed=5, spawning=False)
    route = solve_grid(m.passable, m.start, m.exit, 'astar').path
    events = []
    for cx, cy in route:
        for _ in range(60):
            dx, dy = cx + 0.5 - ex.player.x, cy + 0.5 - ex.player.y
            if abs(dx) + abs(dy) < 0.15 or ex.won:
                break
            events += ex.update(1 / 30, move=(dx, dy))
    assert ex.won and not ex.dead
    assert any(e[0] == 'win' for e in events)


def test_shooting_kills_an_alien_and_counts_hits():
    from csp.rooms import generate_map
    from integration.expedition import Expedition
    m = generate_map(seed=5, target=0.3).map
    ex = Expedition(m, seed=5, spawning=False)
    ex.spawn('skitter', ex.player.x + 2.5, ex.player.y)
    ex.aliens[0].frozen = True
    events = []
    for _ in range(60):
        events += ex.update(1 / 30, aim=(1, 0), shoot=True)
    assert ex.kills == 1 and ex.hits >= 1 and ex.shots >= ex.hits
    assert any(e[0] == 'kill' for e in events)


def test_director_scores_maps_and_learns():
    from integration.expedition import ExpeditionDirector, map_features
    d = ExpeditionDirector(seed=9)
    choice = d.next_map(n_candidates=3)
    assert choice.map is not None and 0 <= choice.predicted_win <= 1
    feats = map_features(choice.map)
    assert {'length', 'gap_tiles', 'max_gap', 'hazards', 'coins', 'platforms', 'difficulty'} <= set(feats)
    before = d.controller.challenge
    d.record(True, 30.0, shots=20, hits=12, progress=1.0)
    assert d.controller.challenge > before
    assert len(d.controller.adapter.pending) == 1


def test_bfs_distance_field():
    field = bfs_distance_field(open_cell, (0, 0), 6, 6)
    assert field[(0, 0)] == 0 and field[(5, 0)] == 5 and field[(5, 5)] == 10
    assert (2, 2) not in field
