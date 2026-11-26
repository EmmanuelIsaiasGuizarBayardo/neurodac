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
from neurodac.mindmetric_history import SessionHistory, StandFunnel
from neurodac.mindmetric_panel import (
    CURRENT_WEEK,
    GROUPS,
    ReportKind,
    report,
    simulate_pilot,
)
from neurodac.mindmetric_sim import MindMetricSimulatedSource
from neurodac.mindmetric_views import (
    fit_figure,
    fit_patch,
    funnel_figure,
    measurement_view,
    quality_children,
    spectrogram_figure,
    spectrogram_patch,
    track_figure,
    track_patch,
)
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

#: Estado de la vista "Como lo medimos". El candado protege el analizador y el
#: historial, porque dos callbacks los leen y escriben en hilos distintos.
_state_lock = threading.RLock()
_histories: dict[str, SessionHistory] = {}
_last_snapshots: dict[str, Snapshot] = {}
_funnel = StandFunnel()
_funnel_shown: list = [None]

AUDIENCES = ("persona", "empresa", "medimos")
#: Segundos que cada aplicación queda resaltada en la vista para
#: organizaciones. Es solo CSS: no hay callback.
APP_SPOTLIGHT_S = 3
#: La vista de medicion se refresca cada 2 ciclos y el espectrograma cada 4:
#: es la parte pesada y a ~1.6 Hz se ve fluido.
MEASURE_EVERY = 2
SPECTROGRAM_EVERY = 4


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


def spark_figure() -> go.Figure:
    """Traza mínima de los últimos 2 s, para la etapa "Señal" del flujo."""
    hidden = {"visible": False, "fixedrange": True}
    return go.Figure(
        data=[go.Scatter(x=[], y=[], mode="lines", line={"color": TRACE_COLOR, "width": 1.2})],
        layout=go.Layout(
            margin={"t": 0, "l": 0, "r": 0, "b": 0},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            uirevision="mm-spark",
            xaxis={**hidden, "range": [-2, 0]},
            yaxis={**hidden, "range": [-TRACE_RANGE_UV, TRACE_RANGE_UV]},
        ),
    )


def _graph(component_id: str, figure: go.Figure, height: str) -> dcc.Graph:
    return dcc.Graph(
        id=component_id,
        figure=figure,
        config={"displayModeBar": False, "staticPlot": True, "responsive": True},
        style={"height": height, "width": "100%"},
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
                children=[_audience_button(key, key == "persona") for key in AUDIENCES],
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


def _plot_card(title: str, note: str, graph: dcc.Graph, card_id: str | None = None) -> html.Div:
    children = [html.H2(title), html.P(render(note), className="mm-note"), graph]
    if card_id:
        # El aviso solo se ve cuando la tarjeta trae la clase is-excluded.
        children.append(html.Div(text("avisos", "excluida"), className="mm-excluded-badge"))
    extra = {"id": card_id} if card_id else {}
    return html.Div(className="mm-plot", children=children, **extra)


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
                        card_id="mm-spectrum-card",
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


def _stage(key: str, number: int, live: object) -> html.Div:
    stage = text("organizacion", "etapas", key)
    return html.Div(
        className="mm-stage",
        style={"--mm-stage-delay": f"{(number - 1) * 0.5}s"},
        children=[
            html.Div(str(number), className="mm-stage-num"),
            html.H3(stage["titulo"]),
            html.P(stage["texto"], className="mm-note"),
            live,
        ],
    )


def _flow_link() -> html.Div:
    return html.Div(className="mm-flow-link", children=[html.Span(), html.Span(), html.Span()])


def _app_card(key: str, index: int) -> html.Div:
    app = text("organizacion", "apps", key)
    return html.Div(
        className="mm-app",
        style={"animationDelay": f"{index * APP_SPOTLIGHT_S}s"},
        children=[
            html.Span(
                text("organizacion", "estados", app["estado"]),
                className=f"mm-app-status is-{app['estado']}",
            ),
            html.H3(app["titulo"]),
            html.P(app["texto"], className="mm-note"),
        ],
    )


def _live_text(index: int) -> html.Div:
    return html.Div(
        text("organizacion", "vivo", "sin_conexion"),
        id=f"mm-stage-live-{index}",
        className="mm-stage-live",
    )


def _company_view() -> html.Section:
    groups = text("empresa", "grupos")
    apps = list(text("organizacion", "apps").keys())
    return html.Section(
        id="mm-view-empresa",
        className="mm-empresa is-hidden",
        children=[
            html.Div(
                className="mm-company-head",
                children=[
                    html.Div(
                        [
                            html.H2(text("organizacion", "titulo")),
                            html.P(text("organizacion", "subtitulo"), className="mm-note"),
                        ]
                    ),
                    html.Div(
                        notice("simulada"),
                        id="mm-sim-notice-org",
                        className="mm-sim-notice is-hidden",
                    ),
                ],
            ),
            html.H3(text("organizacion", "flujo_titulo"), className="mm-section-title"),
            html.Div(
                id="mm-flow",
                className="mm-flow",
                children=[
                    _stage("sensor", 1, _live_text(1)),
                    _flow_link(),
                    _stage("senal", 2, _graph("mm-spark", spark_figure(), "44px")),
                    _flow_link(),
                    _stage("limpieza", 3, _live_text(3)),
                    _flow_link(),
                    _stage("analisis", 4, _live_text(4)),
                    _flow_link(),
                    _stage(
                        "resultado",
                        5,
                        html.Div(
                            text("organizacion", "vivo", "resultado"), className="mm-stage-live"
                        ),
                    ),
                ],
            ),
            html.H3(text("organizacion", "apps_titulo"), className="mm-section-title"),
            html.Div(
                className="mm-apps",
                style={"--mm-spotlight-cycle": f"{len(apps) * APP_SPOTLIGHT_S}s"},
                children=[_app_card(key, i) for i, key in enumerate(apps)],
            ),
            html.Div(
                className="mm-privacy",
                children=[
                    html.Div(
                        [
                            html.H3(text("organizacion", "privacidad_titulo")),
                            html.P(text("organizacion", "privacidad_nota"), className="mm-note"),
                            html.Div(text("empresa", "aviso_simulados"), className="mm-sim-notice"),
                        ]
                    ),
                    html.Div(
                        className="mm-group-picker",
                        role="group",
                        children=[
                            _group_button(g.key, groups[g.key], g is GROUPS[0]) for g in GROUPS
                        ],
                    ),
                    html.Div(
                        group_card(GROUPS[0].key), id="mm-group-card", className="mm-group-card"
                    ),
                ],
            ),
        ],
    )


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
        html.Main(
            className="mm-main",
            children=[_persona_view(), _company_view(), measurement_view(_funnel.counts)],
        ),
        html.Footer(
            className="mm-footer",
            children=[html.Span(text("limites")), html.Span(text("avisos", "prototipo"))],
        ),
        _operator_panel(),
        dcc.Interval(id="mm-interval", interval=TICK_MS, disabled=True),
        dcc.Store(id="mm-session", data=None),
        dcc.Store(id="mm-source-kind", data="real"),
        dcc.Store(id="mm-audience", data="persona"),
    ],
)


# ============================================================= callbacks


@dash.callback(
    [Output(f"mm-view-{key}", "className") for key in AUDIENCES],
    [Output(f"mm-aud-{key}", "className") for key in AUDIENCES],
    [Output(f"mm-aud-{key}", "aria-pressed") for key in AUDIENCES],
    Output("mm-audience", "data"),
    [Input(f"mm-aud-{key}", "n_clicks") for key in AUDIENCES],
    prevent_initial_call=True,
)
def switch_audience(*_clicks):
    """Cambia de audiencia en un clic, sin tocar la sesion."""
    chosen = str(ctx.triggered_id).removeprefix("mm-aud-")
    if chosen not in AUDIENCES:
        chosen = "persona"
    views = [f"mm-{key}" if key == chosen else f"mm-{key} is-hidden" for key in AUDIENCES]
    buttons = [
        "mm-switch-option is-active" if key == chosen else "mm-switch-option" for key in AUDIENCES
    ]
    pressed = ["true" if key == chosen else "false" for key in AUDIENCES]
    # Dash aplana las listas de Output: la respuesta va plana, en el mismo orden.
    return (*views, *buttons, *pressed, chosen)


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
    Output("mm-spectrogram", "figure", allow_duplicate=True),
    Output("mm-fit", "figure", allow_duplicate=True),
    Output("mm-track", "figure", allow_duplicate=True),
    Output("mm-quality", "children", allow_duplicate=True),
    Output("mm-exponent", "children", allow_duplicate=True),
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
    idle = (visitor("desconectada"), "mm-gate", *_empty_figures())
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
    with _state_lock:
        _analyzer(name, reset=True)
        _histories[name] = SessionHistory(time.monotonic())
        _last_snapshots.pop(name, None)
    message = _message(op["conectada"].format(nombre=name), "is-ok")
    return message, False, name, no_update, no_update, *_empty_figures()


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
    with _state_lock:
        _analyzer(name).start_baseline()
        _funnel.baseline_started()
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
    Output("mm-spectrum-card", "className"),
    Input("mm-interval", "n_intervals"),
    State("mm-session", "data"),
    prevent_initial_call=True,
)
def refresh(_n, name):
    """Vacia la cola de la sesion, analiza y actualiza solo lo que cambio."""
    session = registry.get(name) if name else None
    if session is None:
        return (no_update,) * 14

    quality = session.quality
    samples_uv = np.asarray(session.drain(), dtype=float) * RAW_TO_UV
    now = time.monotonic()
    with _state_lock:
        snap = _analyzer(name).update(samples_uv, now, quality.is_ready)
        _last_snapshots[name] = snap
        history = _histories.get(name)
        if history is not None:
            history.record(snap, now, quality.is_ready)
        _funnel.observe(snap, now)

    gate = None if quality.is_ready else visitor(str(quality.state))
    zone_text, zone_class = _zone(snap)
    needle_class, needle_style = _needle(snap)
    progress_class = "mm-progress" if snap.baseline_active else "mm-progress is-hidden"
    iaf_text = [
        text("operacion", "iaf").format(iaf=f"{snap.iaf_hz:.1f}"),
        html.Br(),
        text("operacion", "umbral_parpadeo").format(umbral=f"{snap.blink_threshold_uv:.0f}"),
    ]
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
        _card_class(snap, quality.is_ready),
    )


@dash.callback(
    Output("mm-source", "children"),
    Output("mm-source", "className"),
    Output("mm-sim-notice", "className"),
    Output("mm-sim-notice-medimos", "className"),
    Output("mm-sim-notice-org", "className"),
    Input("mm-session", "data"),
)
def show_source(name):
    """La fuente se declara en pantalla; el aviso sale del nombre de la sesion."""
    hidden = "mm-sim-notice is-hidden"
    if not name:
        return text("fuente", "sin_fuente"), "mm-source is-off", hidden, hidden, hidden
    if name == SIM_NAME:
        shown = "mm-sim-notice"
        return text("operacion", "simulada"), "mm-source is-sim", shown, shown, shown
    return text("fuente", "en_vivo"), "mm-source is-live", hidden, hidden, hidden


@dash.callback(
    Output("mm-spectrogram", "figure"),
    Output("mm-fit", "figure"),
    Output("mm-track", "figure"),
    Output("mm-quality", "children"),
    Output("mm-exponent", "children"),
    Output("mm-funnel", "figure"),
    Output("mm-fit-card", "className"),
    Input("mm-interval", "n_intervals"),
    State("mm-session", "data"),
    State("mm-audience", "data"),
    prevent_initial_call=True,
)
def refresh_measurement(n, name, audience):
    """Actualiza "Como lo medimos" solo cuando esta a la vista."""
    skip = (no_update,) * 7
    if audience != "medimos" or not name or (n or 0) % MEASURE_EVERY:
        return skip
    with _state_lock:
        history = _histories.get(name)
        snap = _last_snapshots.get(name)
        if history is None or snap is None:
            return skip
        view = history.view(time.monotonic())
        counts = _funnel.counts
    iaf = snap.iaf_hz if _analyzer(name).has_baseline else None
    spectrogram = spectrogram_patch(view, iaf) if n % SPECTROGRAM_EVERY == 0 else no_update
    exponent = (
        ""
        if snap.exponent is None
        else text("medimos", "exponente").format(valor=f"{snap.exponent:.2f}")
    )
    funnel = no_update
    if counts != _funnel_shown[0]:
        _funnel_shown[0] = counts
        funnel = funnel_figure(counts)
    return (
        spectrogram,
        fit_patch(snap),
        track_patch(view),
        quality_children(view),
        exponent,
        funnel,
        _card_class(snap, view.clean_fraction is not None),
    )


@dash.callback(
    Output("mm-stage-live-1", "children"),
    Output("mm-stage-live-3", "children"),
    Output("mm-stage-live-4", "children"),
    Output("mm-flow", "className"),
    Output("mm-spark", "figure"),
    Input("mm-interval", "n_intervals"),
    State("mm-session", "data"),
    State("mm-audience", "data"),
    prevent_initial_call=True,
)
def refresh_organization(n, name, audience):
    """Datos en vivo del flujo "Cómo funciona", solo cuando la vista está a la vista."""
    if audience != "empresa" or (n or 0) % MEASURE_EVERY:
        return (no_update,) * 5
    session = registry.get(name) if name else None
    with _state_lock:
        snap = _last_snapshots.get(name) if name else None
        history = _histories.get(name) if name else None
        view = history.view(time.monotonic()) if history is not None else None
    ready = session is not None and session.quality.is_ready
    rate = session.quality.packet_rate_hz if session is not None else 0.0
    sensor, cleaning, analysis, flow = organization_live(snap, view, ready, rate)
    spark = no_update
    if snap is not None:
        keep = snap.trace_t >= -2.0
        spark = Patch()
        spark["data"][0]["x"] = _rounded(snap.trace_t[keep])
        spark["data"][0]["y"] = _rounded(snap.trace_uv[keep])
    return sensor, cleaning, analysis, flow, spark


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


def _card_class(snap: Snapshot, ready: bool) -> str:
    """Marco rojo ligero cuando lo que se dibuja no entró al análisis."""
    excluded = ready and snap.display_spectrum_db is not None and not snap.window_used
    return "mm-plot is-excluded" if excluded else "mm-plot"


def organization_live(snap: Snapshot | None, view, ready: bool, rate_hz: float) -> tuple:
    """Textos en vivo de las etapas 1, 3 y 4 del flujo, y la clase del flujo."""
    live = text("organizacion", "vivo")
    if snap is None or not ready:
        off = live["sin_conexion"]
        return off, off, off, "mm-flow"
    sensor = live["tramas"].format(n=f"{rate_hz:.0f}")
    useful = 0.0 if view is None or view.clean_fraction is None else 100 * view.clean_fraction
    cleaning = live["limpieza"].format(n=snap.blink_count, p=f"{useful:.0f}")
    analysis = live["sin_base"] if snap.zone is None else text("persona", "zonas", snap.zone.value)
    return sensor, cleaning, analysis, "mm-flow is-live"


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
    # Se dibuja la ventana completa, con artefactos: lo que se ve no se recorta.
    # Si el análisis la descartó, la tarjeta lo marca con un marco rojo.
    if snap.spectrum_hz is None or snap.display_spectrum_db is None:
        return no_update
    patch = Patch()
    patch["data"][0]["x"] = _rounded(snap.spectrum_hz)
    patch["data"][0]["y"] = _rounded(snap.display_spectrum_db)
    patch["layout"]["shapes"][0]["x0"] = snap.iaf_hz - ALPHA_HALF_WIDTH_HZ
    patch["layout"]["shapes"][0]["x1"] = snap.iaf_hz + ALPHA_HALF_WIDTH_HZ
    return patch


def _empty_figures() -> tuple:
    """Figuras y textos vacios: sin sesion no se muestra senal congelada."""
    return (
        trace_figure(),
        spectrum_figure(),
        spectrogram_figure(),
        fit_figure(),
        track_figure(),
        quality_children(None),
        "",
    )


def _serial(port: str):
    """Import diferido: pyserial solo hace falta con diadema real."""
    from neurodac.acquisition import SerialSource

    return SerialSource(port)


def _message(message: str, tone: str) -> html.Div:
    return html.Div(message, className=f"mm-op-message {tone}")
