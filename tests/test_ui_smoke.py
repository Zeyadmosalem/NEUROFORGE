"""Smoke tests: every screen draws real frames in a hidden window and its keys work.

They check that the UI code runs and returns the right next screen; they do not
judge how it looks. Skipped when Pygame CE is not installed.
"""
import pytest

pg = pytest.importorskip('pygame')


@pytest.fixture(autouse=True)
def hidden_window(monkeypatch):
    monkeypatch.setenv('SDL_VIDEODRIVER', 'dummy')
    monkeypatch.setenv('SDL_AUDIODRIVER', 'dummy')


def drive(monkeypatch, frames, keys=None):
    """Press `keys` ({frame: key}) and close the window after `frames` frames."""
    keys = keys or {}
    count = [0]
    real_flip = pg.display.flip

    def flip():
        count[0] += 1
        if count[0] in keys:
            pg.event.post(pg.event.Event(pg.KEYDOWN, key=keys[count[0]], mod=0, unicode='', scancode=0))
        if count[0] >= frames:
            pg.event.post(pg.event.Event(pg.QUIT))
        return real_flip()

    monkeypatch.setattr(pg.display, 'flip', flip)
    return count


def test_title_draws_help_and_quits(monkeypatch):
    from game.title import run_title
    count = drive(monkeypatch, 20, {4: pg.K_F1, 8: pg.K_F1, 10: pg.K_DOWN})
    assert run_title(42) is None
    assert count[0] >= 20


def test_title_enter_starts_the_selected_mode(monkeypatch):
    from game.title import run_title
    drive(monkeypatch, 60, {3: pg.K_DOWN, 5: pg.K_RETURN})
    assert run_title(42) == 'arena'


def test_manual_game_draws_lab_help_and_model_switch(monkeypatch):
    from game.app import run
    count = drive(monkeypatch, 70, {3: pg.K_TAB, 8: pg.K_2, 14: pg.K_F1, 20: pg.K_F1,
                                    24: pg.K_n, 30: pg.K_f, 34: pg.K_TAB})
    assert run(42, persist=False) is None
    assert count[0] >= 70


def test_manual_escape_returns_to_title_and_v_opens_arena(monkeypatch):
    from game.app import run
    drive(monkeypatch, 60, {5: pg.K_ESCAPE})
    assert run(42, persist=False) == 'title'
    drive(monkeypatch, 60, {5: pg.K_v})
    assert run(42, persist=False) == 'arena'


def test_arena_draws_help_then_escape_returns_to_title(monkeypatch):
    from game.arena_app import run_arena
    drive(monkeypatch, 400, {40: pg.K_F1, 45: pg.K_ESCAPE, 50: pg.K_ESCAPE})
    assert run_arena(42, persist=False) == 'title'
