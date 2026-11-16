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


def test_bfs_distance_field():
    field = bfs_distance_field(open_cell, (0, 0), 6, 6)
    assert field[(0, 0)] == 0 and field[(5, 0)] == 5 and field[(5, 5)] == 10
    assert (2, 2) not in field
