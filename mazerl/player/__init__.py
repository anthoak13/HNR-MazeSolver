"""NYT-style animated maze player (HTML + SVG + JavaScript, no dependencies).

Renders recorded maze episodes inline in Jupyter / Colab outputs, or as standalone
``.html`` files. Everything (JS, CSS, data) is inlined; nothing is fetched from the
network.

Episode data contract (all coordinates are ``[row, col]``, row 0 at the top)::

    {
      "grid":  rows x cols list of int  (0 floor, 1 wall, 2 weak spot, 3 outside),
      "start": [r, c],  "flag": [r, c],
      "agents": [{"path": [[r, c], ...],          # path[0] = start, path[t] = after move t
                  "score": [float, ...],           # cumulative reward, same length as path
                  "events": [{"t": int, "type": str, "value": float, "cell": [r, c]}, ...],
                  "label": str (optional)}, ...],
      "title": str (optional), "caption": str (optional),
      "options": {"speed": 6, "autoplay": True, "loop": True,
                  "tint_outside": "auto" | "always" | "never",
                  "callouts": True, "show_score": True, "labels": True,
                  "trail_length": None}            # optional; None = whole trail
    }

Event types: ``probe``, ``through_wall``, ``outside``, ``new_cell``, ``flag``.

Public API: :func:`render`, :func:`render_grid`, :func:`show`, :func:`save`.
"""

from __future__ import annotations

import html as _html
import json
import math
import uuid
from importlib import resources
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Union

__all__ = ["render", "render_grid", "show", "save"]

Episode = Mapping[str, Any]

_ASSETS: dict = {}


def _asset(name: str) -> str:
    if name not in _ASSETS:
        _ASSETS[name] = resources.files(__name__).joinpath(name).read_text(encoding="utf-8")
    return _ASSETS[name]


def _plain(x: Any) -> Any:
    """Convert numpy / torch scalars and arrays, tuples and sets to plain JSON-able Python."""
    if x is None or isinstance(x, (bool, str)):
        return x
    if isinstance(x, int):
        return int(x)
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, Mapping):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, frozenset)):
        return [_plain(v) for v in x]
    if hasattr(x, "tolist"):  # numpy arrays & scalars, torch tensors
        return _plain(x.tolist())
    if hasattr(x, "item"):
        return _plain(x.item())
    if isinstance(x, Iterable):
        return [_plain(v) for v in x]
    raise TypeError(f"cannot serialise {type(x).__name__} for the maze player")


def _prepare_episode(episode: Episode, options: Mapping[str, Any] | None = None) -> dict:
    if not isinstance(episode, Mapping):
        raise TypeError("episode must be a dict (see mazerl.player docstring)")
    ep = _plain(episode)
    grid = ep.get("grid")
    if not grid or not isinstance(grid, list) or not isinstance(grid[0], list):
        raise ValueError("episode['grid'] must be a non-empty 2-D list of ints")
    ep.setdefault("agents", [])
    if options:
        ep["options"] = {**(ep.get("options") or {}), **_plain(dict(options))}
    return ep


def _js_json(obj: Any) -> str:
    """JSON that is safe to drop inside a <script> element."""
    s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return (s.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
             .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def _snippet(payload: dict, width: int, title: str | None, caption: str | None) -> str:
    uid = "mzp-" + uuid.uuid4().hex[:12]
    parts = [f'<div id="{uid}" class="mzp-root" style="max-width:{int(width)}px">']
    parts.append(f"<style>{_asset('player.css')}</style>")
    if title:
        parts.append(f'<div class="mzp-title">{_html.escape(str(title))}</div>')
    parts.append('<div class="mzp-stage"><div class="mzp-fallback">'
                 "Maze animation (needs JavaScript; run the cell or open the saved .html file)."
                 "</div></div>")
    if caption:
        parts.append(f'<div class="mzp-caption">{_html.escape(str(caption))}</div>')
    parts.append(
        # Find the root via currentScript (works even if the output is not attached to the
        # document yet, as in some notebook front ends), else by id, retrying briefly.
        "<script>(function(){var ID=" + json.dumps(uid) + ",S=document.currentScript,MZP="
        + _asset("player.js").strip().rstrip(";")
        + ",DATA=" + _js_json(payload) + ";"
        "function go(n){var p=S&&S.parentNode,el=(p&&p.id===ID)?p:document.getElementById(ID);"
        "if(el){MZP.mount(el,DATA);}else if(n<100){setTimeout(function(){go(n+1);},100);}}"
        "go(0);})();</script>"
    )
    parts.append("</div>")
    return "".join(parts)


def render(episode: Episode, width: int = 520, *, title: str | None = None,
           caption: str | None = None, **options: Any) -> str:
    """HTML snippet that animates one episode.

    ``title`` / ``caption`` override the episode's own; any other keyword
    (``speed``, ``autoplay``, ``loop``, ``tint_outside``, ``callouts``,
    ``show_score``, ``labels``, ``trail_length``) overrides ``episode['options']``.
    """
    ep = _prepare_episode(episode, options)
    payload = {"mode": "single", "width": int(width), "episodes": [ep]}
    return _snippet(payload, width,
                    title if title is not None else ep.get("title"),
                    caption if caption is not None else ep.get("caption"))


def render_grid(episodes: Sequence[Episode], cols: int = 3, width: int = 900, *,
                title: str | None = None, caption: str | None = None, **options: Any) -> str:
    """HTML snippet with several mazes side by side sharing one clock and one set of controls.

    Each episode's ``title`` becomes a small heading over its maze. Text labels
    (AGENT / GOAL / OUTSIDE) are off by default in the grid, as in the article;
    pass ``labels=True`` to turn them on. Playback speed / autoplay / loop come
    from the first episode (or from keyword overrides).
    """
    episodes = list(episodes)
    if not episodes:
        raise ValueError("render_grid needs at least one episode")
    eps = [_prepare_episode(e, options) for e in episodes]
    payload = {"mode": "grid", "width": int(width), "cols": max(1, int(cols)), "episodes": eps}
    return _snippet(payload, width, title, caption)


def _render_any(episode_or_list: Union[Episode, Sequence[Episode]], **kw: Any) -> str:
    if isinstance(episode_or_list, Mapping):
        return render(episode_or_list, **kw)
    return render_grid(list(episode_or_list), **kw)


def show(episode_or_list: Union[Episode, Sequence[Episode]], **kw: Any):
    """Return an ``IPython.display.HTML`` object (a list of episodes gives a grid)."""
    from IPython.display import HTML  # lazy: the player works without IPython

    return HTML(_render_any(episode_or_list, **kw))


def save(episode_or_list: Union[Episode, Sequence[Episode]], path: Union[str, Path], **kw: Any) -> Path:
    """Write a standalone HTML page with the animation; returns the path."""
    body = _render_any(episode_or_list, **kw)
    if isinstance(episode_or_list, Mapping):
        page_title = kw.get("title") or episode_or_list.get("title") or "Maze episode"
    else:
        page_title = kw.get("title") or "Maze episodes"
    doc = (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<title>{_html.escape(str(page_title))}</title>\n"
        "<style>html,body{margin:0;background:#ffffff;color:#121212}"
        "body{padding:24px 16px}.mzp-page{max-width:960px;margin:0 auto}</style>\n"
        "</head>\n<body>\n<div class=\"mzp-page\">\n" + body + "\n</div>\n</body>\n</html>\n"
    )
    path = Path(path)
    path.write_text(doc, encoding="utf-8")
    return path
