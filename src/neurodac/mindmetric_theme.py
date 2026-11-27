"""Colores del demo MindMetric para las graficas de Plotly.

Plotly no lee variables CSS, asi que aqui viven los mismos valores que los
tokens de assets/mindmetric.css, convertidos de OKLCH a sRGB. El sistema es el
tema Console de Hallmark (https://github.com/nutlope/hallmark): un grafito en
tres pasos, tinta palida, ambar para lo que hace una persona y azul frio para
lo que mide un instrumento. Rojo y verde solo para estado.

El espectrograma conserva batlow (figuras-cientificas): es un mapa secuencial
uniforme y legible con daltonismo, que el tema no sustituye.
"""

from __future__ import annotations

__all__ = [
    "ACCENT",
    "BAD",
    "BAD_SOFT",
    "COOL",
    "COOL_SOFT",
    "FONT_BODY",
    "FONT_MEASURED",
    "INK",
    "INK_2",
    "MUTED",
    "PAPER_2",
    "PAPER_3",
    "RULE",
    "RULE_2",
]

PAPER_2 = "#15191c"
PAPER_3 = "#06080b"
INK = "#eceff2"
INK_2 = "#c3c8cc"
MUTED = "#9da2a8"
RULE = "#2d3135"
RULE_2 = "#484e54"
ACCENT = "#f7b83d"
COOL = "#6ebcfc"
COOL_SOFT = "#142e43"
BAD = "#f66d67"
BAD_SOFT = "#2f1a18"

FONT_BODY = "Host Grotesk, Segoe UI, system-ui, sans-serif"
FONT_MEASURED = "Atkinson Hyperlegible Mono, Consolas, ui-monospace, monospace"
