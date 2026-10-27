"""
plotstyle.py — let matplotlib render CJK labels, falling back to English automatically if no CJK font is found.

matplotlib's default font DejaVu Sans has no CJK glyphs; CJK labels flood the console with
"Glyph xxxxx missing from font(s) DejaVu Sans" and appear as boxes in the figure.

This module does two things:
1. puts a CJK font that actually exists on the system at the front of font.sans-serif;
2. if there is none (common in slim Linux containers), switches `T()` to English labels,
   so figures are always readable rather than full of boxes.

Usage::

    from plotstyle import setup, T
    setup()
    ax.set_xlabel(T("Voltage (V)", "Voltage (V)"))
"""

from __future__ import annotations

__all__ = ["setup", "T", "has_cjk", "set_language", "LANG"]

# search in the usual order Windows → macOS → Linux
_CANDIDATES = [
    "Microsoft YaHei", "Microsoft JhengHei", "SimHei", "SimSun", "KaiTi",
    "PingFang SC", "Hiragino Sans GB", "Heiti SC", "STHeiti", "Songti SC",
    "Noto Sans CJK SC", "Noto Sans SC", "Source Han Sans SC", "Source Han Sans CN",
    "WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "Arial Unicode MS",
]

_has_cjk = False
_done = False

# figure label language. "en" = all English (default, convenient for English papers/reports);
# "zh" = use CJK labels if a CJK font is available. Change here or call set_language("zh").
LANG = "en"


def set_language(lang: str):
    """Switch the figure label language: "en" or "zh". Affects figures only, not terminal output."""
    global LANG
    if lang not in ("en", "zh"):
        raise ValueError('lang must be "en" or "zh"')
    LANG = lang


def setup(verbose: bool = True) -> bool:
    """Configure matplotlib fonts. Returns True if CJK labels are available."""
    global _has_cjk, _done
    if _done:
        return _has_cjk
    _done = True

    try:
        import matplotlib
        from matplotlib import rcParams
        import matplotlib.font_manager as fm
    except ImportError:
        return False

    available = {f.name for f in fm.fontManager.ttflist}
    picked = [n for n in _CANDIDATES if n in available]

    if picked:
        rcParams["font.sans-serif"] = picked + [
            n for n in rcParams["font.sans-serif"] if n not in picked
        ]
        rcParams["font.family"] = "sans-serif"
        _has_cjk = True
        if verbose:
            print(f"Figure CJK font: {picked[0]}")
    else:
        _has_cjk = False
        if verbose:
            print("No CJK font found; figure labels switched to English (terminal output unaffected).\n"
                  "  Usually bundled on Windows/macOS; on Linux install: "
                  "sudo apt install fonts-noto-cjk")

    # most CJK fonts lack the U+2212 math minus; use the ASCII hyphen instead,
    # otherwise the minus sign also becomes a box.
    rcParams["axes.unicode_minus"] = False
    return _has_cjk


def has_cjk() -> bool:
    return _has_cjk


def T(zh: str, en: str) -> str:
    """
    Choose one of two labels.

    With LANG == "en" (default) always returns English; with LANG == "zh"
    returns the CJK label only if a CJK font is actually installed, otherwise still falls back to English to avoid boxes.
    """
    if not _done:
        setup(verbose=False)
    if LANG == "en":
        return en
    return zh if _has_cjk else en
