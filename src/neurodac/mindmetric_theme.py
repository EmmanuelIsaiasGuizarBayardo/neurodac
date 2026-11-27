"""Colores del demo MindMetric para las graficas de Plotly.

Plotly no lee variables CSS, asi que aqui viven los mismos valores que los
tokens de assets/mindmetric.css, convertidos de OKLCH a sRGB. El sistema es el
tema Console de Hallmark (https://github.com/nutlope/hallmark) en sus dos
colorways: "oscuro" (graphite) y "claro" (paper). En ambos, ambar es lo que hace
una persona y azul frio lo que mide un instrumento; rojo y verde, solo estado.

El espectrograma conserva batlow (figuras-cientificas): es un mapa secuencial
uniforme y legible con daltonismo, que el tema no sustituye.
"""

from __future__ import annotations

__all__ = ["DARK", "FONT_BODY", "FONT_MEASURED", "LIGHT", "THEMES", "palette"]

FONT_BODY = "Host Grotesk, Segoe UI, system-ui, sans-serif"
FONT_MEASURED = "Atkinson Hyperlegible Mono, Consolas, ui-monospace, monospace"

DARK = {
    "paper_2": "#15191c",
    "paper_3": "#06080b",
    "ink": "#eceff2",
    "ink_2": "#c3c8cc",
    "muted": "#9da2a8",
    "rule": "#2d3135",
    "rule_2": "#484e54",
    "accent": "#f7b83d",
    "cool": "#6ebcfc",
    "cool_soft": "#142e43",
    "cool_faint": "rgba(110, 188, 252, 0.30)",
    "bad": "#f66d67",
    "bad_fill": "rgba(246, 109, 103, 0.10)",
    "baseline_fill": "rgba(157, 162, 168, 0.10)",
}

LIGHT = {
    "paper_2": "#fffdfa",
    "paper_3": "#f3f0ea",
    "ink": "#17130f",
    "ink_2": "#393430",
    "muted": "#69625d",
    "rule": "#dddad5",
    "rule_2": "#bdbab3",
    "accent": "#a15c00",
    "cool": "#116bb5",
    "cool_soft": "#d5ebfe",
    "cool_faint": "rgba(17, 107, 181, 0.25)",
    "bad": "#c53637",
    "bad_fill": "rgba(197, 54, 55, 0.08)",
    "baseline_fill": "rgba(105, 98, 93, 0.10)",
}

THEMES = {"oscuro": DARK, "claro": LIGHT}


def palette(name: str | None) -> dict[str, str]:
    """Paleta del tema; oscuro si el nombre no se reconoce."""
    return THEMES.get(name or "oscuro", DARK)


# Las graficas se construyen con la paleta oscura; `recolor` las traduce a la
# clara cambiando cada color por su par. Asi hay una sola definicion de cada
# figura y el cambio de tema no toca la logica de las graficas.
PAPER_2 = DARK["paper_2"]
PAPER_3 = DARK["paper_3"]
INK = DARK["ink"]
INK_2 = DARK["ink_2"]
MUTED = DARK["muted"]
RULE = DARK["rule"]
RULE_2 = DARK["rule_2"]
ACCENT = DARK["accent"]
COOL = DARK["cool"]
COOL_SOFT = DARK["cool_soft"]
COOL_FAINT = DARK["cool_faint"]
BAD = DARK["bad"]
BAD_FILL = DARK["bad_fill"]
BASELINE_FILL = DARK["baseline_fill"]

_TO_LIGHT = {DARK[k].lower(): LIGHT[k] for k in DARK}


def recolor(obj, name: str | None):
    """Devuelve `obj` con los colores oscuros cambiados por los del tema `name`."""
    if name != "claro":
        return obj
    if isinstance(obj, dict):
        return {k: recolor(v, name) for k, v in obj.items()}
    if isinstance(obj, list):
        return [recolor(v, name) for v in obj]
    if isinstance(obj, str):
        return _TO_LIGHT.get(obj.lower(), obj)
    return obj
