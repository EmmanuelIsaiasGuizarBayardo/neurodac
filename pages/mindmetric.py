"""Demo provisional de MindMetric, solo para la pantalla de la laptop.

Es una prueba para el Foro FCA, no el demo planeado (web con celulares). Vive
en una ruta aparte, `/mindmetric`, fuera de la barra de NeuroDAC, y reutiliza
la adquisicion de NeuroDAC: la pagina no abre puertos ni lanza hilos, solo
pide una sesion al registro y lee de ella.

Todo el texto visible viene de la seccion `mindmetric` del contenido.

No usa componentes de Bootstrap a proposito: NeuroDAC carga su tema desde una
CDN, y sin internet esos componentes pierden el estilo. Esta pagina se arma
solo con `html`, `dcc` y `assets/mindmetric.css`, que viajan con el repositorio.
"""

from __future__ import annotations

import threading
import time

import dash
import numpy as np
import plotly.graph_objs as go
from dash import ALL, Input, Output, Patch, State, ctx, dcc, html, no_update

from neurodac.acquisition import registry
from neurodac.content import mindmetric as text
from neurodac.content import notice, render, visitor
from neurodac.mindmetric_analysis import (
    ALPHA_HALF_WIDTH_HZ,
    DISPLAY_S,
    IAF_DEFAULT_HZ,
    RAW_TO_UV,
    SPECTRUM_DISPLAY_HZ,
    Z_DISPLAY_LIMIT,
    LiveAnalyzer,
    MetricStatus,
    Snapshot,
)
from neurodac.mindmetric_panel import (
    CURRENT_WEEK,
    GROUPS,
    PILOT_WEEKS,
    ReportKind,
    report,
    simulate_pilot,
)
from neurodac.mindmetric_sim import MindMetricSimulatedSource
from neurodac.ui import render_quality

dash.register_page(
    __name__,
    path="/mindmetric",
    name="MindMetric",
    title=f"{text('marca', 'nombre')}, {text('marca', 'respaldo').lower()}",
)

SIM_NAME = "mindmetric-simulada"
TICK_MS = 150

#: Colores de datos de figuras-cientificas, categorical(6) con Petroff (2024).
TRACE_COLOR = "#5790fc"
ALPHA_BAND_COLOR = "#f89c20"
ARTIFACT_COLOR = "#e42536"
INK = "#142328"
MUTED = "#52646a"
RULE = "#d9e0de"

TRACE_RANGE_UV = 100.0

_analyzers: dict[str, LiveAnalyzer] = {}
_analyzers_lock = threading.Lock()
_pilot = simulate_pilot()


def _analyzer(name: str, *, reset: bool = False) -> LiveAnalyzer:
    """Un analizador por sesion; se reinicia al conectar."""
    with _analyzers_lock:
        if reset or name not in _analyzers:
            _analyzers[name] = LiveAnalyzer()
        return _analyzers[name]


# =============================================================== figuras


def _base_layout(**axes) -> go.Layout:
    return go.Layout(
        margin={"t": 8, "l": 52, "r": 12, "b": 40},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Outfit, Segoe UI, sans-serif", "color": MUTED, "size": 12},
        showlegend=False,
        uirevision="mm",
        **axes,
    )


def trace_figure() -> go.Figure:
    """Senal filtrada; el artefacto se dibuja encima en otro color."""
    axis = {"gridcolor": RULE, "zeroline": False, "fixedrange": True}
    return go.Figure(
        data=[
            go.Scatter(x=[], y=[], mode="lines", line={"color": TRACE_COLOR, "width": 1.3}),
            go.Scatter(x=[], y=[], mode="lines", line={"color": ARTIFACT_COLOR, "width": 1.8}),
        ],
        layout=_base_layout(
            xaxis={**axis, "range": [-DISPLAY_S, 0], "title": {"text": "s"}},
            yaxis={
                **axis,
                "range": [-TRACE_RANGE_UV, TRACE_RANGE_UV],
                "title": {"text": "µV"},
                "zeroline": True,
                "zerolinecolor": RULE,
            },
        ),
    )


def spectrum_figure() -> go.Figure:
    """Espectro suavizado con la banda alfa sombreada."""
    axis = {"gridcolor": RULE, "zeroline": False, "fixedrange": True}
    figure = go.Figure(
        data=[go.Scatter(x=[], y=[], mode="lines", line={"color": INK, "width": 1.6})],
        layout=_base_layout(
            xaxis={**axis, "range": list(SPECTRUM_DISPLAY_HZ), "title": {"text": "Hz"}},
            yaxis={**axis, "title": {"text": "dB"}},
        ),
    )
    figure.add_vrect(
        x0=IAF_DEFAULT_HZ - ALPHA_HALF_WIDTH_HZ,
        x1=IAF_DEFAULT_HZ + ALPHA_HALF_WIDTH_HZ,
        fillcolor=ALPHA_BAND_COLOR,
        opacity=0.18,
        line_width=0,
        layer="below",
    )
    return figure


def trend_figure() -> go.Figure:
    """Participacion semanal de todo el piloto, ya con las reglas aplicadas."""
    reports = _pilot.weekly_report(GROUPS[0].key)
    weeks = list(range(1, CURRENT_WEEK + 1))
    values = [r.percent for r in reports]
    axis = {"gridcolor": RULE, "zeroline": False, "fixedrange": True}
    return go.Figure(
        data=[
            go.Scatter(
                x=weeks,
                y=values,
                mode="lines+markers",
                line={"color": TRACE_COLOR, "width": 2},
                marker={"size": 7},
                hovertemplate="%{y} %<extra></extra>",
            )
        ],
        layout=_base_layout(
            xaxis={
                **axis,
                "range": [0.5, PILOT_WEEKS + 0.5],
                "dtick": 1,
                "title": {"text": text("empresa", "semana")},
            },
            yaxis={
                **axis,
                "range": [0, 105],
                "dtick": 20,
                "title": {"text": text("empresa", "participacion_eje")},
            },
        ),
    )


def _graph(component_id: str, figure: go.Figure, height: str) -> dcc.Graph:
    return dcc.Graph(
        id=component_id,
        figure=figure,
        config={"displayModeBar": False, "staticPlot": True},
        style={"height": height},
    )


# ================================================================ layout


def _audience_button(key: str, active: bool) -> html.Button:
    return html.Button(
        text("vistas", key),
        id=f"mm-aud-{key}",
        className="mm-switch-option is-active" if active else "mm-switch-option",
        **{"aria-pressed": "true" if active else "false"},
    )


def _header() -> html.Header:
    return html.Header(
        className="mm-header",
        children=[
            html.Div(
                className="mm-brand",
                children=[
                    html.Span(text("marca", "nombre"), className="mm-brand-name"),
                    html.Span(text("marca", "respaldo"), className="mm-brand-by"),
                ],
            ),
            html.Div(
                className="mm-switch",
                role="group",
                children=[_audience_button("persona", True), _audience_button("empresa", False)],
            ),
            html.Div(
                className="mm-header-end",
                children=[
                    html.Span(
                        text("fuente", "sin_fuente"), id="mm-source", className="mm-source is-off"
                    ),
                    html.Button(
                        text("operacion", "abrir"), id="mm-op-open", className="mm-op-button"
                    ),
                ],
            ),
        ],
    )


def _gauge() -> html.Div:
    zones = text("persona", "zonas")
    return html.Div(
        className="mm-gauge",
        children=[
            html.Div(
                className="mm-gauge-track",
                children=[
                    html.Div(className="mm-gauge-zone"),
                    html.Div(id="mm-gauge-needle", className="mm-gauge-needle is-off"),
                ],
            ),
            html.Div(
                className="mm-gauge-labels",
                children=[
                    html.Span(zones["debajo"]),
                    html.Span(zones["zona"]),
                    html.Span(zones["encima"]),
                ],
            ),
        ],
    )


def _plot_card(title: str, note: str, graph: dcc.Graph) -> html.Div:
    return html.Div(
        className="mm-plot",
        children=[html.H2(title), html.P(render(note), className="mm-note"), graph],
    )


def _persona_view() -> html.Section:
    return html.Section(
        id="mm-view-persona",
        className="mm-persona",
        children=[
            html.Div(notice("simulada"), id="mm-sim-notice", className="mm-sim-notice is-hidden"),
            html.Div(
                className="mm-plots",
                children=[
                    _plot_card(
                        text("persona", "senal_titulo"),
                        text("persona", "senal_nota"),
                        _graph("mm-trace", trace_figure(), "30vh"),
                    ),
                    _plot_card(
                        text("persona", "espectro_titulo"),
                        text("persona", "espectro_nota"),
                        _graph("mm-spectrum", spectrum_figure(), "22vh"),
                    ),
                ],
            ),
            html.Div(
                className="mm-instrument",
                children=[
                    html.Div(visitor("desconectada"), id="mm-gate", className="mm-gate"),
                    html.H2(text("persona", "indicador_titulo")),
                    html.Div(id="mm-zone", className="mm-zone is-empty"),
                    _gauge(),
                    html.P(id="mm-status", className="mm-status"),
                    html.Div(
                        id="mm-progress",
                        className="mm-progress is-hidden",
                        children=html.Div(id="mm-progress-fill", className="mm-progress-fill"),
                    ),
                    html.Button(
                        text("persona", "boton_medir"), id="mm-baseline", className="mm-action"
                    ),
                    html.P(text("persona", "indicador_nota"), className="mm-note"),
                    html.P(text("persona", "linea_base_abreviada"), className="mm-note"),
                    html.P(text("avisos", "no_se_guarda"), className="mm-note"),
                ],
            ),
        ],
    )


def _group_button(key: str, label: str, active: bool) -> html.Button:
    return html.Button(
        label,
        id={"type": "mm-group", "key": key},
        className="mm-group-option is-active" if active else "mm-group-option",
    )


def _company_view() -> html.Section:
    groups = text("empresa", "grupos")
    return html.Section(
        id="mm-view-empresa",
        className="mm-empresa is-hidden",
        children=[
            html.Div(
                className="mm-company-head",
                children=[
                    html.Div(
                        [
                            html.H2(text("empresa", "titulo")),
                            html.P(text("empresa", "subtitulo"), className="mm-note"),
                        ]
                    ),
                    html.Div(text("empresa", "aviso_simulados"), className="mm-sim-notice"),
                ],
            ),
            html.Div(
                className="mm-company-grid",
                children=[
                    html.Div(
                        className="mm-plot",
                        children=[
                            html.H3(text("empresa", "participacion_titulo")),
                            _graph("mm-trend", trend_figure(), "30vh"),
                            html.P(
                                [
                                    html.Strong(f"{_pilot.median_sessions(CURRENT_WEEK):.0f} "),
                                    text("empresa", "sesiones"),
                                ],
                                className="mm-note",
                            ),
                        ],
                    ),
                    html.Div(
                        className="mm-groups",
                        children=[
                            html.H3(text("empresa", "grupos_titulo")),
                            html.Div(
                                className="mm-group-picker",
                                role="group",
                                children=[
                                    _group_button(g.key, groups[g.key], g is GROUPS[0])
                                    for g in GROUPS
                                ],
                            ),
                            html.Div(
                                group_card(GROUPS[0].key),
                                id="mm-group-card",
                                className="mm-group-card",
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="mm-company-lists",
                children=[
                    _list(text("empresa", "nunca_titulo"), text("empresa", "nunca")),
                    _list(text("empresa", "acuerdo_titulo"), text("empresa", "acuerdo")),
                ],
            ),
        ],
    )


def _list(title: str, items: list[str]) -> html.Div:
    return html.Div([html.H3(title), html.Ul([html.Li(i) for i in items])])


def _source_button(kind: str, label: str, active: bool) -> html.Button:
    return html.Button(
        label,
        id=f"mm-kind-{kind}",
        className="mm-op-choice is-active" if active else "mm-op-choice",
    )


def _operator_panel() -> html.Aside:
    op = text("operacion")
    return html.Aside(
        id="mm-op",
        className="mm-op is-closed",
        children=[
            html.Div(
                className="mm-op-head",
                children=[
                    html.H2(op["titulo"]),
                    html.Button(op["cerrar"], id="mm-op-close", className="mm-op-close"),
                ],
            ),
            html.Div(op["fuente"], className="mm-op-label"),
            html.Div(
                className="mm-op-choices",
                role="group",
                children=[
                    _source_button("real", op["diadema"], True),
                    _source_button("sim", op["simulada"], False),
                ],
            ),
            html.Label(op["puerto"], htmlFor="mm-port", className="mm-op-label"),
            dcc.Input(id="mm-port", type="text", placeholder="COM5", className="mm-op-input"),
            html.Div(
                className="mm-op-actions",
                children=[
                    html.Button(op["conectar"], id="mm-connect", className="mm-op-connect"),
                    html.Button(op["detener"], id="mm-stop", className="mm-op-stop"),
                ],
            ),
            html.Div(id="mm-op-status", className="mm-op-status"),
            html.Div(id="mm-op-quality", className="nd-quality mm-op-quality"),
            html.P(id="mm-op-iaf", className="mm-op-note"),
            html.P(op["guion_simulado"], className="mm-op-note"),
        ],
    )


def group_card(group_key: str) -> list:
    """Tarjeta de un grupo con las reglas de agregacion ya aplicadas."""
    size = next(g.size for g in GROUPS if g.key == group_key)
    result = report(size, _pilot.participants(group_key, CURRENT_WEEK))
    if result.kind is ReportKind.SUPPRESSED:
        figure = html.Div(text("empresa", "oculto"), className="mm-figure is-suppressed")
    elif result.kind is ReportKind.RANGE:
        figure = html.Div(
            text("empresa", "rangos", result.range_key), className="mm-figure is-range"
        )
    else:
        figure = html.Div(f"{result.percent} %", className="mm-figure")
    return [
        html.Div(text("empresa", "tamano").format(n=size), className="mm-note"),
        figure,
        html.P(text("empresa", "reglas", result.kind.value), className="mm-rule"),
    ]


layout = html.Div(
    className="mm-root",
    children=[
        _header(),
        html.Main(className="mm-main", children=[_persona_view(), _company_view()]),
        html.Footer(
            className="mm-footer",
            children=[html.Span(text("limites")), html.Span(text("avisos", "prototipo"))],
        ),
        _operator_panel(),
        dcc.Interval(id="mm-interval", interval=TICK_MS, disabled=True),
        dcc.Store(id="mm-session", data=None),
        dcc.Store(id="mm-source-kind", data="real"),
    ],
)


# ============================================================= callbacks


@dash.callback(
    Output("mm-view-persona", "className"),
    Output("mm-view-empresa", "className"),
    Output("mm-aud-persona", "className"),
    Output("mm-aud-empresa", "className"),
    Output("mm-aud-persona", "aria-pressed"),
    Output("mm-aud-empresa", "aria-pressed"),
    Input("mm-aud-persona", "n_clicks"),
    Input("mm-aud-empresa", "n_clicks"),
    prevent_initial_call=True,
)
def switch_audience(_persona, _empresa):
    """Cambia de audiencia en un clic, sin tocar la sesion."""
    company = ctx.triggered_id == "mm-aud-empresa"
    on, off = "mm-switch-option is-active", "mm-switch-option"
    if company:
        return "mm-persona is-hidden", "mm-empresa", off, on, "false", "true"
    return "mm-persona", "mm-empresa is-hidden", on, off, "true", "false"


@dash.callback(
    Output("mm-op", "className"),
    Input("mm-op-open", "n_clicks"),
    Input("mm-op-close", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_operator(_open, _close):
    """El panel de operacion queda fuera de lo que mira el publico."""
    return "mm-op" if ctx.triggered_id == "mm-op-open" else "mm-op is-closed"


@dash.callback(
    Output("mm-source-kind", "data"),
    Output("mm-kind-real", "className"),
    Output("mm-kind-sim", "className"),
    Output("mm-port", "disabled"),
    Input("mm-kind-real", "n_clicks"),
    Input("mm-kind-sim", "n_clicks"),
    prevent_initial_call=True,
)
def choose_source(_real, _sim):
    """Diadema o simulada; con simulada el puerto no aplica."""
    on, off = "mm-op-choice is-active", "mm-op-choice"
    if ctx.triggered_id == "mm-kind-sim":
        return "sim", off, on, True
    return "real", on, off, False


@dash.callback(
    Output("mm-op-status", "children"),
    Output("mm-interval", "disabled"),
    Output("mm-session", "data"),
    Output("mm-gate", "children", allow_duplicate=True),
    Output("mm-gate", "className", allow_duplicate=True),
    Output("mm-trace", "figure", allow_duplicate=True),
    Output("mm-spectrum", "figure", allow_duplicate=True),
    Input("mm-connect", "n_clicks"),
    Input("mm-stop", "n_clicks"),
    State("mm-source-kind", "data"),
    State("mm-port", "value"),
    State("mm-session", "data"),
    prevent_initial_call=True,
)
def manage_connection(_connect, _stop, source_kind, port, current):
    """Conecta o detiene delegando en el registro; no abre nada por su cuenta."""
    op = text("operacion")
    # Sin sesion, las graficas se vacian: una senal congelada aparentaria
    # seguir en vivo.
    idle = (visitor("desconectada"), "mm-gate", trace_figure(), spectrum_figure())
    if ctx.triggered_id == "mm-stop":
        if current:
            registry.stop(current)
        return _message(op["detenida"], "is-neutral"), True, None, *idle

    if source_kind == "real" and not (port or "").strip():
        return _message(op["falta_puerto"], "is-warning"), True, None, *idle

    name = SIM_NAME if source_kind == "sim" else port.strip().upper()
    if current and current != name:
        registry.stop(current)
    source = MindMetricSimulatedSource(SIM_NAME) if source_kind == "sim" else _serial(name)
    try:
        registry.acquire(source, "raw")
    except Exception as exc:  # noqa: BLE001
        # El mensaje del sistema operativo distingue puerto ocupado de inexistente.
        return _message(op["error"].format(error=exc), "is-danger"), True, None, *idle
    _analyzer(name, reset=True)
    message = _message(op["conectada"].format(nombre=name), "is-ok")
    return message, False, name, no_update, no_update, trace_figure(), spectrum_figure()


@dash.callback(
    Output("mm-baseline", "children"),
    Input("mm-baseline", "n_clicks"),
    State("mm-session", "data"),
    prevent_initial_call=True,
)
def start_baseline(_clicks, name):
    """Empieza una linea base nueva para la persona que tiene la diadema."""
    if not name:
        return no_update
    _analyzer(name).start_baseline()
    return text("persona", "boton_repetir")


@dash.callback(
    Output("mm-trace", "figure"),
    Output("mm-spectrum", "figure"),
    Output("mm-gate", "children"),
    Output("mm-gate", "className"),
    Output("mm-zone", "children"),
    Output("mm-zone", "className"),
    Output("mm-gauge-needle", "className"),
    Output("mm-gauge-needle", "style"),
    Output("mm-status", "children"),
    Output("mm-progress", "className"),
    Output("mm-progress-fill", "style"),
    Output("mm-op-quality", "children"),
    Output("mm-op-iaf", "children"),
    Input("mm-interval", "n_intervals"),
    State("mm-session", "data"),
    prevent_initial_call=True,
)
def refresh(_n, name):
    """Vacia la cola de la sesion, analiza y actualiza solo lo que cambio."""
    session = registry.get(name) if name else None
    if session is None:
        return (no_update,) * 13

    quality = session.quality
    samples_uv = np.asarray(session.drain(), dtype=float) * RAW_TO_UV
    snap = _analyzer(name).update(samples_uv, time.monotonic(), quality.is_ready)

    gate = None if quality.is_ready else visitor(str(quality.state))
    zone_text, zone_class = _zone(snap)
    needle_class, needle_style = _needle(snap)
    progress_class = "mm-progress" if snap.baseline_active else "mm-progress is-hidden"
    iaf_text = text("operacion", "iaf").format(iaf=f"{snap.iaf_hz:.1f}")
    return (
        _trace_patch(snap),
        _spectrum_patch(snap),
        gate,
        "mm-gate" if gate else "mm-gate is-hidden",
        zone_text,
        zone_class,
        needle_class,
        needle_style,
        text("persona", "estados", _status_key(snap)),
        progress_class,
        {"width": f"{100 * snap.baseline_progress:.0f}%"},
        render_quality(quality),
        iaf_text,
    )


@dash.callback(
    Output("mm-source", "children"),
    Output("mm-source", "className"),
    Output("mm-sim-notice", "className"),
    Input("mm-session", "data"),
)
def show_source(name):
    """La fuente se declara en pantalla; el aviso sale del nombre de la sesion."""
    if not name:
        return text("fuente", "sin_fuente"), "mm-source is-off", "mm-sim-notice is-hidden"
    if name == SIM_NAME:
        return text("operacion", "simulada"), "mm-source is-sim", "mm-sim-notice"
    return text("fuente", "en_vivo"), "mm-source is-live", "mm-sim-notice is-hidden"


@dash.callback(
    Output("mm-group-card", "children"),
    Output({"type": "mm-group", "key": ALL}, "className"),
    Input({"type": "mm-group", "key": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def show_group(_clicks):
    """Aplica las reglas de agregacion al grupo elegido."""
    key = ctx.triggered_id["key"] if ctx.triggered_id else GROUPS[0].key
    classes = ["mm-group-option is-active" if g.key == key else "mm-group-option" for g in GROUPS]
    return group_card(key), classes


# ============================================================== auxiliares


def _status_key(snap: Snapshot) -> str:
    """Mientras se mide la linea base, una ventana descartada no cambia el mensaje."""
    if snap.baseline_active and snap.status is MetricStatus.HOLD:
        return MetricStatus.BASELINE.value
    return snap.status.value


def _zone(snap: Snapshot) -> tuple[str, str]:
    if snap.zone is None:
        return "", "mm-zone is-empty"
    css = "mm-zone is-held" if snap.status is MetricStatus.HOLD else "mm-zone"
    return text("persona", "zonas", snap.zone.value), css


def _needle(snap: Snapshot) -> tuple[str, dict]:
    if snap.z_display is None:
        return "mm-gauge-needle is-off", no_update
    left = 50.0 + 50.0 * snap.z_display / Z_DISPLAY_LIMIT
    css = "mm-gauge-needle is-held" if snap.status is MetricStatus.HOLD else "mm-gauge-needle"
    return css, {"left": f"{left:.1f}%"}


def _rounded(values: np.ndarray) -> list[float | None]:
    return [None if np.isnan(v) else round(float(v), 2) for v in values]


def _trace_patch(snap: Snapshot) -> Patch:
    patch = Patch()
    t = _rounded(snap.trace_t)
    patch["data"][0]["x"] = t
    patch["data"][0]["y"] = _rounded(snap.trace_uv)
    patch["data"][1]["x"] = t
    patch["data"][1]["y"] = _rounded(snap.trace_artifact_uv)
    return patch


def _spectrum_patch(snap: Snapshot) -> object:
    if snap.spectrum_hz is None:
        return no_update
    patch = Patch()
    patch["data"][0]["x"] = _rounded(snap.spectrum_hz)
    patch["data"][0]["y"] = _rounded(snap.spectrum_db)
    patch["layout"]["shapes"][0]["x0"] = snap.iaf_hz - ALPHA_HALF_WIDTH_HZ
    patch["layout"]["shapes"][0]["x1"] = snap.iaf_hz + ALPHA_HALF_WIDTH_HZ
    return patch


def _serial(port: str):
    """Import diferido: pyserial solo hace falta con diadema real."""
    from neurodac.acquisition import SerialSource

    return SerialSource(port)


def _message(message: str, tone: str) -> html.Div:
    return html.Div(message, className=f"mm-op-message {tone}")
