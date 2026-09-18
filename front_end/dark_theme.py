"""
dark_theme.py — make the app's inline-styled HTML follow the Streamlit theme.

Streamlit's [theme] in .streamlit/config.toml recolours its own widgets, but this
app also builds a lot of HTML with inline colours chosen for a light page
(dark text, pale backgrounds, pale borders) — about 70 places, many of them
assembled at run time.  Rather than hand-editing each one (and re-editing all of
them every time the theme flips), install() wraps st.markdown / DeltaGenerator.markdown
so that, when the theme is dark, every unsafe_allow_html snippet has its CSS
colours mapped to a dark equivalent on the way to the page.

The mapping is by ROLE, keeping the hue so status colours keep their meaning
(red still reads as an error, green as OK):
  * text    (color:)              dark/mid colours are lightened; greys are flipped
  * surface (background:)         pale surfaces become dark tinted surfaces; saturated
                                  badge colours (green/blue/orange fills) are left alone
  * border  (border*, box-shadow) pale lines become subtle dark lines
Light theme: install() does nothing, so the app renders exactly as before.
A snippet starting with <!--nodark--> is passed through untouched.
"""
from __future__ import annotations

import colorsys
import re

import streamlit as st

_HEX = re.compile(r"#(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{3})\b")
_DECL = re.compile(
    r"(?P<prop>(?<![-\w])(?:background(?:-color)?|color|border(?:-[a-z]+)*|outline(?:-color)?|box-shadow))"
    r"(?P<sep>\s*:\s*)(?P<val>[^;\"'}]*)",
    re.I,
)


def _rgb(h: str):
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _hex(r: float, g: float, b: float) -> str:
    return "#%02x%02x%02x" % tuple(round(max(0.0, min(1.0, v)) * 255) for v in (r, g, b))


def _lum(r: float, g: float, b: float) -> float:
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def map_text(h: str) -> str:
    """Text colour on a dark surface: lighten dark/mid colours, flip greys."""
    r, g, b = _rgb(h)
    lum = _lum(r, g, b)
    if lum >= 0.62:
        return h                                    # already light (e.g. #fff on a coloured badge)
    hh, ll, ss = colorsys.rgb_to_hls(r, g, b)
    _, sv, _ = colorsys.rgb_to_hsv(r, g, b)
    if lum < 0.22:                                  # near-black / very dark: bright, faintly cool text
        l2, s2 = min(0.95, max(0.84, 1 - ll)), min(ss, 0.35)
    elif sv < 0.25:                                 # greys and slates: flip lightness
        l2, s2 = min(0.93, max(0.66, 1 - ll)), ss
    else:                                           # coloured text: lighten, keep the hue
        l2, s2 = max(ll, 0.68), ss
    return _hex(*colorsys.hls_to_rgb(hh, l2, s2))


def map_surface(h: str) -> str:
    """Background: pale surfaces -> dark tinted surfaces; saturated fills stay."""
    r, g, b = _rgb(h)
    lum = _lum(r, g, b)
    if lum < 0.72:
        return h
    hh, _, _ = colorsys.rgb_to_hls(r, g, b)
    _, sv, _ = colorsys.rgb_to_hsv(r, g, b)
    if sv < 0.03:
        hh = 0.6                                    # pure greys/white take the UI's navy hue
    l2 = 0.13 + (1.0 - lum) * 0.4
    s2 = min(0.55, sv * 3.0 + 0.14)
    return _hex(*colorsys.hls_to_rgb(hh, l2, s2))


def map_border(h: str) -> str:
    r, g, b = _rgb(h)
    if _lum(r, g, b) < 0.6:
        return h
    return _hex(*colorsys.hls_to_rgb(0.6, 0.27, 0.28))


def transform(html: str) -> str:
    """Map every CSS colour in an HTML/CSS snippet to its dark-theme equivalent."""
    if "#" not in html or html.lstrip().startswith("<!--nodark-->"):
        return html

    def repl(m: re.Match) -> str:
        prop, val = m.group("prop").lower(), m.group("val")
        if prop.startswith("background"):
            f = map_surface
        elif prop == "color":
            f = map_text
        else:                                       # border*, outline, box-shadow
            f = map_border
        return m.group("prop") + m.group("sep") + _HEX.sub(lambda h: f(h.group(0)), val)

    return _DECL.sub(repl, html)


def is_dark() -> bool:
    try:
        return str(st.get_option("theme.base")).lower() != "light"
    except Exception:
        return True


_installed = False


def install() -> None:
    """Idempotent.  No-op on the light theme."""
    global _installed
    if _installed or not is_dark():
        return
    from streamlit.delta_generator import DeltaGenerator

    def wrap(fn):
        def markdown(*args, **kwargs):
            # Called as (html, ...) via st.markdown but as (self, html, ...) via slot/container.markdown,
            # so transform the first string argument rather than assuming a position.
            if kwargs.get("unsafe_allow_html"):
                args = list(args)
                for i, a in enumerate(args):
                    if isinstance(a, str):
                        args[i] = transform(a)
                        break
                else:
                    if isinstance(kwargs.get("body"), str):
                        kwargs["body"] = transform(kwargs["body"])
                args = tuple(args)
            return fn(*args, **kwargs)
        markdown.__wrapped__ = fn
        return markdown

    # `st.markdown` is bound to the main DeltaGenerator at import time, so patch it and the class
    # (the class patch covers slot.markdown / st.sidebar.markdown / container.markdown).
    DeltaGenerator.markdown = wrap(DeltaGenerator.markdown)
    st.markdown = wrap(st.markdown)
    _installed = True
