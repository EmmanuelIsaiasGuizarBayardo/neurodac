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

from .content import mindmetric as text
from .content import notice, render
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
    "funnel_figure",
    "measurement_view",
    "quality_children",
    "spectrogram_figure",
    "stimulus_figure",
    "stimulus_patch",
    "spectrogram_patch",
    "track_figure",
    "track_patch",
]

TRACE_COLOR = "#5790fc"
ALPHA_BAND_COLOR = "#f89c20"
INK = "#142328"
MUTED = "#52646a"
RULE = "#d9e0de"
ZONE_FILL = "rgba(87, 144, 252, 0.14)"
#: Fondo del espectrograma mientras aún no hay un minuto de datos.
GAP_COLOR = "#dfe5e3"
#: Marco de los tramos que el análisis descarta: se ven, pero no se usan.
EXCLUDED_LINE = "rgba(228, 37, 54, 0.85)"
EXCLUDED_FILL = "rgba(228, 37, 54, 0.10)"


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


BASELINE_FILL = "rgba(82, 100, 106, 0.12)"

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
    margin = axes.pop("margin", {"t": 6, "l": 50, "r": 12, "b": 36})
    plot_bg = axes.pop("plot_bgcolor", "rgba(0,0,0,0)")
    return go.Layout(
        margin=margin,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=plot_bg,
        font={"family": "Outfit, Segoe UI, sans-serif", "color": MUTED, "size": 12},
        showlegend=False,
        uirevision="mm-medimos",
        **axes,
    )


def _axis(**extra) -> dict:
    return {"gridcolor": RULE, "zeroline": False, "fixedrange": True, **extra}


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
                colorbar={"title": {"text": "dB", "side": "right"}, "thickness": 10, "len": 0.9},
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
                title={"text": text("medimos", "eje_hace")},
                showgrid=False,
            ),
            yaxis=_axis(range=list(SPECTRUM_DISPLAY_HZ), title={"text": "Hz"}, showgrid=False),
            plot_bgcolor=GAP_COLOR,
        ),
    )


def spectrogram_patch(view: HistoryView, iaf_hz: float | None) -> object:
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
    patch["layout"]["shapes"] = excluded_shapes(view.excluded_spans)
    return patch


# ================================================= curva 1/f y pico alfa


def fit_figure() -> go.Figure:
    """Espectro, recta aperiodica y la potencia periodica alfa sombreada."""
    return go.Figure(
        data=[
            go.Scatter(x=[], y=[], mode="lines", line={"color": INK, "width": 1.6}),
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
                fillcolor="rgba(248, 156, 32, 0.45)",
                hoverinfo="skip",
            ),
        ],
        layout=_layout(
            xaxis=_axis(range=[2, 40], title={"text": "Hz"}),
            yaxis=_axis(title={"text": "dB"}),
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


def stimulus_patch(response: StimulusResponse) -> object:
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
            "line": {"color": "rgba(87, 144, 252, 0.28)", "width": 1},
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
    patch["data"] = [*faint, mean]
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
            xaxis=_axis(range=[-TRACK_S, 0], title={"text": text("medimos", "eje_hace")}),
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


def track_patch(view: HistoryView) -> Patch:
    """Linea del indicador y bandas de las lineas base."""
    patch = Patch()
    patch["data"][0]["x"] = _rounded(view.track_t)
    patch["data"][0]["y"] = _rounded(view.track_z)
    patch["layout"]["shapes"] = _track_shapes(view.baseline_spans) + excluded_shapes(
        [s for s in view.excluded_spans if s[1] > -TRACK_S]
    )
    return patch


# ================================================================== calidad


def quality_children(view: HistoryView | None) -> list:
    """Tres cifras de calidad; "sin dato" cuando todavia no hay medicion."""
    missing = text("medimos", "sin_dato")
    if view is None:
        clean, blinks, ready = missing, missing, missing
    else:
        clean = missing if view.clean_fraction is None else f"{100 * view.clean_fraction:.0f} %"
        blinks = str(view.blinks)
        ready = missing if view.time_to_ready_s is None else f"{view.time_to_ready_s:.0f} s"
    return [
        _figure_item(clean, text("medimos", "limpio")),
        _figure_item(blinks, text("medimos", "parpadeos")),
        _figure_item(ready, text("medimos", "tiempo_senal")),
    ]


def _figure_item(value: str, label: str) -> html.Div:
    return html.Div(
        className="mm-kpi",
        children=[
            html.Div(value, className="mm-kpi-value"),
            html.Div(label, className="mm-kpi-label"),
        ],
    )


def funnel_figure(counts: FunnelCounts) -> go.Figure:
    """Barras horizontales del embudo del stand, con porcentaje sobre el inicio."""
    stages = text("medimos", "embudo_etapas")
    labels = [stages["iniciada"], stages["completa"], stages["practica"]]
    values = [counts.started, counts.completed, counts.practiced]
    base = max(counts.started, 1)
    labels_text = [
        f"{v}" if i == 0 else f"{v} ({100 * v / base:.0f} %)" for i, v in enumerate(values)
    ]
    return go.Figure(
        data=[
            go.Bar(
                x=values,
                y=labels,
                orientation="h",
                marker={"color": [TRACE_COLOR, TRACE_COLOR, ALPHA_BAND_COLOR]},
                text=labels_text,
                textposition="outside",
                textfont={"size": 12, "color": INK},
                cliponaxis=False,
                hoverinfo="skip",
            )
        ],
        layout=_layout(
            margin={"t": 4, "l": 8, "r": 60, "b": 4},
            xaxis=_axis(visible=False, range=[0, base * 1.15]),
            yaxis=_axis(autorange="reversed", automargin=True, showgrid=False),
            bargap=0.35,
            uniformtext={"minsize": 12, "mode": "show"},
        ),
    )


# ================================================================== layout


def _graph(component_id: str, figure: go.Figure, height: str) -> dcc.Graph:
    return dcc.Graph(
        id=component_id,
        figure=figure,
        config={"displayModeBar": False, "staticPlot": True},
        style={"height": height},
    )


def _card(title: str, note: str, *children, card_id: str | None = None) -> html.Div:
    content = [html.H3(title), html.P(render(note), className="mm-note"), *children]
    extra = {}
    if card_id:
        # El aviso solo se ve cuando la tarjeta trae la clase is-excluded.
        content.append(html.Div(text("avisos", "excluida"), className="mm-excluded-badge"))
        extra["id"] = card_id
    return html.Div(className="mm-plot", children=content, **extra)


def measurement_view(funnel: FunnelCounts) -> html.Section:
    """Seccion completa de la vista; empieza oculta."""
    return html.Section(
        id="mm-view-medimos",
        className="mm-medimos is-hidden",
        children=[
            html.Div(
                className="mm-company-head",
                children=[
                    html.Div(
                        [
                            html.H2(text("medimos", "titulo")),
                            html.P(text("medimos", "subtitulo"), className="mm-note"),
                        ]
                    ),
                    html.Div(
                        notice("simulada"),
                        id="mm-sim-notice-medimos",
                        className="mm-sim-notice is-hidden",
                    ),
                ],
            ),
            html.Div(
                className="mm-measure-grid",
                children=[
                    _card(
                        text("medimos", "espectrograma_titulo"),
                        text("medimos", "espectrograma_nota"),
                        _graph("mm-spectrogram", spectrogram_figure(), "20vh"),
                    ),
                    _card(
                        text("medimos", "curva_titulo"),
                        text("medimos", "curva_nota"),
                        _graph("mm-fit", fit_figure(), "17vh"),
                        html.P(id="mm-exponent", className="mm-note"),
                        card_id="mm-fit-card",
                    ),
                    _card(
                        text("medimos", "historial_titulo"),
                        text("medimos", "historial_nota"),
                        _graph("mm-track", track_figure(), "14vh"),
                    ),
                    _card(
                        text("medimos", "calidad_titulo"),
                        text("medimos", "calidad_nota"),
                        html.Div(quality_children(None), id="mm-quality", className="mm-kpis"),
                        html.H3(text("medimos", "embudo_titulo"), className="mm-subhead"),
                        _graph("mm-funnel", funnel_figure(funnel), "7vh"),
                        html.P(render(text("medimos", "embudo_nota")), className="mm-note"),
                    ),
                ],
            ),
            html.Div(
                className="mm-references",
                children=[
                    html.Span(text("medimos", "respaldo_titulo"), className="mm-references-title"),
                    *[html.Span(render(item)) for item in text("medimos", "respaldo")],
                ],
            ),
        ],
    )
