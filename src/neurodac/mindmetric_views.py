"""Vista "Como lo medimos" del demo de MindMetric: figuras y actualizaciones.

Separada de pages/mindmetric.py para que las figuras se prueben sin levantar
la aplicacion. Las funciones `*_patch` convierten un Snapshot o un HistoryView
en actualizaciones parciales de Plotly; la pagina decide cuando llamarlas.

Colores de datos: figuras-cientificas. Categoricos con categorical(6), Petroff
(2024); el espectrograma con sequential(), batlow de Crameri, uniforme en
percepcion y legible con daltonismo.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objs as go
from dash import Patch, dcc, html, no_update

from . import mindmetric_theme as theme
from .content import mindmetric as text
from .mindmetric_analysis import (
    ALPHA_HALF_WIDTH_HZ,
    SPECTRUM_DISPLAY_HZ,
    Z_DISPLAY_LIMIT,
    ZONE_Z,
    Snapshot,
)
from .mindmetric_history import (
    SPECTROGRAM_S,
    STIMULUS_POST_S,
    STIMULUS_PRE_S,
    TRACK_S,
    FunnelCounts,
    HistoryView,
    StimulusResponse,
)

__all__ = [
    "BATLOW",
    "excluded_shapes",
    "fit_figure",
    "fit_patch",
    "funnel_children",
    "measurement_view",
    "quality_children",
    "spectrogram_figure",
    "stimulus_figure",
    "stimulus_patch",
    "spectrogram_patch",
    "track_figure",
    "track_patch",
]

# Tema Console (mindmetric_theme.py): frío para lo que mide el instrumento,
# ámbar para lo que hace una persona, rojo solo para lo que no se usa.
TRACE_COLOR = theme.COOL
ALPHA_BAND_COLOR = theme.ACCENT
INK = theme.INK
MUTED = theme.MUTED
RULE = theme.RULE
ZONE_FILL = theme.COOL_SOFT
#: Fondo del espectrograma mientras aún no hay un minuto de datos.
GAP_COLOR = theme.PAPER_3
#: Marco de los tramos que el análisis descarta: se ven, pero no se usan.
EXCLUDED_LINE = theme.BAD
EXCLUDED_FILL = theme.BAD_FILL


def excluded_shapes(spans: list[tuple[float, float]], y_ref: str = "paper") -> list[dict]:
    """Marcos rojos ligeros sobre los tramos descartados."""
    return [
        {
            "type": "rect",
            "xref": "x",
            "yref": y_ref,
            "x0": start,
            "x1": end,
            "y0": 0,
            "y1": 1,
            "fillcolor": EXCLUDED_FILL,
            "line": {"color": EXCLUDED_LINE, "width": 1.2},
            "layer": "above",
        }
        for start, end in spans
    ]


BASELINE_FILL = theme.BASELINE_FILL

#: figuras_cientificas.sequential() (batlow), muestreado en 11 puntos.
BATLOW = [
    [0.0, "#011959"],
    [0.1, "#0f3c5f"],
    [0.2, "#185562"],
    [0.3, "#30685c"],
    [0.4, "#577647"],
    [0.5, "#828231"],
    [0.6, "#b38e2f"],
    [0.7, "#e19752"],
    [0.8, "#fba689"],
    [0.9, "#fdb9c2"],
    [1.0, "#faccfa"],
]

#: Percentiles del rango de color: recortan colas sin que un parpadeo extremo
#: aplane toda la escala.
COLOR_RANGE_PCT = (5.0, 99.5)


def _layout(**axes) -> go.Layout:
    margin = axes.pop("margin", {"t": 4, "l": 54, "r": 8, "b": 28})
    plot_bg = axes.pop("plot_bgcolor", "rgba(0,0,0,0)")
    return go.Layout(
        margin=margin,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=plot_bg,
        font={"family": theme.FONT_MEASURED, "color": MUTED, "size": 11},
        showlegend=False,
        uirevision="mm-medimos",
        **axes,
    )


def _axis(**extra) -> dict:
    return {"gridcolor": RULE, "zeroline": False, "fixedrange": True, "linecolor": RULE, **extra}


def _rounded(values: np.ndarray, digits: int = 2) -> list[float | None]:
    return [None if not np.isfinite(v) else round(float(v), digits) for v in values]


# ============================================================ espectrograma


def spectrogram_figure() -> go.Figure:
    """Mapa de calor frecuencia por tiempo, con la IAF marcada."""
    return go.Figure(
        data=[
            go.Heatmap(
                x=[],
                y=[],
                z=[[]],
                colorscale=BATLOW,
                hoverongaps=False,
                colorbar={"thickness": 8, "len": 0.9, "ticksuffix": " dB", "outlinewidth": 0},
            ),
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                line={"color": "#ffffff", "width": 1, "dash": "dot"},
                hoverinfo="skip",
            ),
        ],
        layout=_layout(
            xaxis=_axis(
                range=[-SPECTROGRAM_S, 0],
                ticksuffix=" s",
                showgrid=False,
            ),
            yaxis=_axis(range=list(SPECTRUM_DISPLAY_HZ), ticksuffix=" Hz", showgrid=False),
            plot_bgcolor=GAP_COLOR,
        ),
    )


def spectrogram_patch(
    view: HistoryView, iaf_hz: float | None, theme_name: str | None = None
) -> object:
    """Actualiza el mapa completo; no_update si aun no hay espectro."""
    if view.spec_db is None or view.spec_hz is None or view.spec_t.size == 0:
        return no_update
    finite = view.spec_db[np.isfinite(view.spec_db)]
    patch = Patch()
    patch["data"][0]["x"] = _rounded(view.spec_t)
    patch["data"][0]["y"] = _rounded(view.spec_hz)
    patch["data"][0]["z"] = [_rounded(row, 1) for row in view.spec_db]
    if finite.size:
        low, high = np.percentile(finite, COLOR_RANGE_PCT)
        patch["data"][0]["zmin"] = round(float(low), 1)
        patch["data"][0]["zmax"] = round(float(high), 1)
    if iaf_hz is not None:
        patch["data"][1]["x"] = [-SPECTROGRAM_S, 0]
        patch["data"][1]["y"] = [iaf_hz, iaf_hz]
    patch["layout"]["shapes"] = theme.recolor(excluded_shapes(view.excluded_spans), theme_name)
    return patch


# ================================================= curva 1/f y pico alfa


def fit_figure() -> go.Figure:
    """Espectro, recta aperiodica y la potencia periodica alfa sombreada."""
    return go.Figure(
        data=[
            go.Scatter(x=[], y=[], mode="lines", line={"color": theme.INK_2, "width": 1.5}),
            go.Scatter(
                x=[], y=[], mode="lines", line={"color": MUTED, "width": 1.4, "dash": "dash"}
            ),
            go.Scatter(x=[], y=[], mode="lines", line={"width": 0}, hoverinfo="skip"),
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor=theme.COOL_SOFT,
                hoverinfo="skip",
            ),
        ],
        layout=_layout(
            xaxis=_axis(range=[2, 40], ticksuffix=" Hz"),
            yaxis=_axis(ticksuffix=" dB", nticks=4),
        ),
    )


def fit_patch(snap: Snapshot) -> object:
    """Dibuja el ultimo espectro util con su recta y el area alfa entre ambas."""
    if snap.spectrum_hz is None or snap.fit_hz is None or snap.fit_db is None:
        return no_update
    f = snap.fit_hz
    spectrum_on_fit = np.interp(f, snap.spectrum_hz, snap.spectrum_db)
    lo, hi = snap.iaf_hz - ALPHA_HALF_WIDTH_HZ, snap.iaf_hz + ALPHA_HALF_WIDTH_HZ
    band = (f >= lo) & (f <= hi)
    patch = Patch()
    patch["data"][0]["x"] = _rounded(f)
    patch["data"][0]["y"] = _rounded(spectrum_on_fit)
    patch["data"][1]["x"] = _rounded(f)
    patch["data"][1]["y"] = _rounded(snap.fit_db)
    patch["data"][2]["x"] = _rounded(f[band])
    patch["data"][2]["y"] = _rounded(snap.fit_db[band])
    patch["data"][3]["x"] = _rounded(f[band])
    patch["data"][3]["y"] = _rounded(np.maximum(spectrum_on_fit[band], snap.fit_db[band]))
    return patch


# ======================================================= respuesta a estimulo

#: Curvas individuales que se dibujan, ademas del promedio.
MAX_STIMULUS_CURVES = 12


def stimulus_figure() -> go.Figure:
    """Cambio de alfa alrededor del estimulo: curvas tenues y su promedio."""
    return go.Figure(
        data=[go.Scatter(x=[], y=[], mode="lines", line={"color": TRACE_COLOR, "width": 2.4})],
        layout=_layout(
            xaxis=_axis(
                range=[-STIMULUS_PRE_S, STIMULUS_POST_S],
                title={"text": text("organizacion", "demos", "estimulo", "eje_x")},
            ),
            yaxis=_axis(title={"text": text("organizacion", "demos", "estimulo", "eje_y")}),
            shapes=[
                {
                    "type": "line",
                    "xref": "x",
                    "yref": "paper",
                    "x0": 0,
                    "x1": 0,
                    "y0": 0,
                    "y1": 1,
                    "line": {"color": ALPHA_BAND_COLOR, "width": 2},
                },
                {
                    "type": "line",
                    "xref": "paper",
                    "yref": "y",
                    "x0": 0,
                    "x1": 1,
                    "y0": 0,
                    "y1": 0,
                    "line": {"color": RULE, "width": 1},
                },
            ],
        ),
    )


def stimulus_patch(response: StimulusResponse, theme_name: str | None = None) -> object:
    """Reemplaza las curvas: una tenue por estimulo y el promedio encima."""
    if not response.curves:
        return no_update
    x = _rounded(response.offsets)
    faint = [
        {
            "type": "scatter",
            "mode": "lines",
            "x": x,
            "y": _rounded(c),
            "line": {"color": theme.COOL_FAINT, "width": 1},
            "hoverinfo": "skip",
        }
        for c in response.curves[-MAX_STIMULUS_CURVES:]
    ]
    mean = {
        "type": "scatter",
        "mode": "lines",
        "x": x,
        "y": _rounded(response.mean),
        "line": {"color": TRACE_COLOR, "width": 2.4},
    }
    patch = Patch()
    patch["data"] = theme.recolor([*faint, mean], theme_name)
    return patch


# ================================================================ historial


def _track_shapes(spans: list[tuple[float, float]]) -> list[dict]:
    zone = {
        "type": "rect",
        "xref": "x",
        "yref": "y",
        "x0": -TRACK_S,
        "x1": 0,
        "y0": -ZONE_Z,
        "y1": ZONE_Z,
        "fillcolor": ZONE_FILL,
        "line": {"width": 0},
        "layer": "below",
    }
    baselines = [
        {
            "type": "rect",
            "xref": "x",
            "yref": "paper",
            "x0": max(start, -TRACK_S),
            "x1": end,
            "y0": 0,
            "y1": 1,
            "fillcolor": BASELINE_FILL,
            "line": {"width": 0},
            "layer": "below",
        }
        for start, end in spans
        if end > -TRACK_S
    ]
    return [zone, *baselines]


def track_figure() -> go.Figure:
    """Indicador de los ultimos minutos; sin numeros en el eje: no es un puntaje."""
    limit = Z_DISPLAY_LIMIT + 0.2
    return go.Figure(
        data=[go.Scatter(x=[], y=[], mode="lines", line={"color": TRACE_COLOR, "width": 2})],
        layout=_layout(
            xaxis=_axis(range=[-TRACK_S, 0], ticksuffix=" s"),
            yaxis=_axis(
                range=[-limit, limit],
                showticklabels=False,
                showgrid=False,
            ),
            shapes=_track_shapes([]),
            annotations=[
                {
                    "x": -TRACK_S,
                    "y": 0,
                    "xref": "x",
                    "yref": "y",
                    "xanchor": "left",
                    "text": text("persona", "zonas", "zona"),
                    "showarrow": False,
                    "font": {"size": 11, "color": MUTED},
                }
            ],
        ),
    )


def track_patch(view: HistoryView, theme_name: str | None = None) -> Patch:
    """Linea del indicador y bandas de las lineas base."""
    patch = Patch()
    patch["data"][0]["x"] = _rounded(view.track_t)
    patch["data"][0]["y"] = _rounded(view.track_z)
    patch["layout"]["shapes"] = theme.recolor(_track_shapes(view.baseline_spans), theme_name)
    return patch


# ================================================================== calidad


def quality_children(view: HistoryView | None) -> list:
    """Tres lecturas de calidad; "sin dato" cuando todavía no hay medición."""
    missing = text("medimos", "sin_dato")
    if view is None:
        clean, blinks, ready = missing, missing, missing
    else:
        clean = missing if view.clean_fraction is None else f"{100 * view.clean_fraction:.0f} %"
        blinks = str(view.blinks)
        ready = missing if view.time_to_ready_s is None else f"{view.time_to_ready_s:.0f} s"
    return [
        _stat(clean, text("medimos", "limpio")),
        _stat(blinks, text("medimos", "parpadeos")),
        _stat(ready, text("medimos", "tiempo_senal")),
    ]


def _stat(value: str, label: str) -> html.Div:
    return html.Div(
        className="mm-stat",
        children=[
            html.Div(value, className="mm-stat-value"),
            html.Div(label, className="mm-stat-label"),
        ],
    )


def funnel_children(counts: FunnelCounts) -> list:
    """El embudo del stand como fila de lecturas: un conteo y su etapa."""
    stages = text("medimos", "embudo_etapas")
    return [
        _stat(str(counts.started), stages["iniciada"]),
        _stat(str(counts.completed), stages["completa"]),
        _stat(str(counts.practiced), stages["practica"]),
    ]


# ================================================================== layout


def _graph(component_id: str, figure: go.Figure, height: str) -> dcc.Graph:
    return dcc.Graph(
        id=component_id,
        figure=figure,
        config={"displayModeBar": False, "staticPlot": True},
        style={"height": height},
    )


def _panel(
    span: str, title: str, meta: str | None, *children, panel_id: str | None = None
) -> html.Div:
    head = [html.H2(title)]
    if meta:
        head.append(html.Span(meta, className="mm-meta"))
    if panel_id:
        # Solo se ve cuando el panel trae la clase is-excluded.
        head.append(
            html.Span(text("avisos", "excluida"), className="mm-chip is-bad mm-excluded-chip")
        )
    extra = {"id": panel_id} if panel_id else {}
    return html.Div(
        className=f"mm-panel {span}",
        children=[html.Div(className="mm-panel-head", children=head), *children],
        **extra,
    )


def measurement_view(funnel: FunnelCounts) -> html.Section:
    """Vista completa; empieza oculta. Paneles cortados de una sola superficie."""
    return html.Section(
        id="mm-view-medimos",
        className="mm-view mm-medimos is-hidden",
        children=[
            _panel(
                "mm-span-spectrogram",
                text("medimos", "espectrograma"),
                text("medimos", "espectrograma_meta"),
                _graph("mm-spectrogram", spectrogram_figure(), "100%"),
            ),
            _panel(
                "mm-span-fit",
                text("medimos", "curva"),
                None,
                _graph("mm-fit", fit_figure(), "100%"),
                html.Span(id="mm-exponent", className="mm-measured"),
                panel_id="mm-fit-card",
            ),
            _panel(
                "mm-span-track",
                text("medimos", "historial"),
                text("medimos", "historial_meta"),
                _graph("mm-track", track_figure(), "100%"),
            ),
            _panel(
                "mm-span-quality",
                text("medimos", "calidad"),
                None,
                html.Div(quality_children(None), id="mm-quality", className="mm-stats"),
                html.Div(
                    className="mm-panel-head mm-subhead",
                    children=[html.H2(text("medimos", "embudo"))],
                ),
                html.Div(funnel_children(funnel), id="mm-funnel", className="mm-stats"),
                html.Div(
                    className="mm-refs",
                    children=[
                        html.Span(text("medimos", "respaldo_titulo")),
                        *[html.Span(r) for r in text("medimos", "respaldo")],
                    ],
                ),
            ),
        ],
    )
