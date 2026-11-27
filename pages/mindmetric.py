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

from neurodac import mindmetric_theme as theme
from neurodac.acquisition import registry
from neurodac.content import mindmetric as text
from neurodac.content import visitor
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
    funnel_children,
    measurement_view,
    quality_children,
    spectrogram_figure,
    spectrogram_patch,
    stimulus_figure,
    stimulus_patch,
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

TRACE_RANGE_UV = 100.0

_analyzers: dict[str, LiveAnalyzer] = {}
_analyzers_lock = threading.Lock()
_pilot = simulate_pilot()

#: Estado de las vistas en vivo. El candado protege el analizador y el
#: historial, porque varios callbacks los leen y escriben en hilos distintos.
_state_lock = threading.RLock()
_histories: dict[str, SessionHistory] = {}
_last_snapshots: dict[str, Snapshot] = {}
_funnel = StandFunnel()
_funnel_shown: list = [None]

AUDIENCES = ("persona", "empresa", "medimos")
#: Aplicaciones que abren una demostración en vivo.
DEMO_CARDS = {"control": "control", "consumidores": "estimulo"}
#: Las vistas secundarias se refrescan cada 2 ciclos y el espectrograma cada 4.
MEASURE_EVERY = 2
SPECTROGRAM_EVERY = 4


def _analyzer(name: str, *, reset: bool = False) -> LiveAnalyzer:
    """Un analizador por sesion; se reinicia al conectar."""
    with _analyzers_lock:
        if reset or name not in _analyzers:
            _analyzers[name] = LiveAnalyzer()
        return _analyzers[name]


# =============================================================== figuras
# Sistema visual: tema Console de Hallmark (mindmetric_theme.py). Azul frío
# para lo que mide el instrumento, ámbar para lo que hace una persona, rojo
# solo para lo que el análisis no usa.


def _base_layout(**axes) -> go.Layout:
    return go.Layout(
        margin={"t": 4, "l": 62, "r": 8, "b": 28},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": theme.FONT_MEASURED, "color": theme.MUTED, "size": 11},
        showlegend=False,
        uirevision="mm",
        **axes,
    )


def _axis(**extra) -> dict:
    return {
        "gridcolor": theme.RULE,
        "zeroline": False,
        "fixedrange": True,
        "linecolor": theme.RULE,
        **extra,
    }


def trace_figure() -> go.Figure:
    """Señal filtrada; el artefacto se dibuja encima en rojo."""
    return go.Figure(
        data=[
            go.Scatter(x=[], y=[], mode="lines", line={"color": theme.COOL, "width": 1.3}),
            go.Scatter(x=[], y=[], mode="lines", line={"color": theme.BAD, "width": 1.6}),
        ],
        layout=_base_layout(
            xaxis=_axis(range=[-DISPLAY_S, 0], ticksuffix=" s"),
            yaxis=_axis(
                range=[-TRACE_RANGE_UV, TRACE_RANGE_UV],
                ticksuffix=" µV",
                dtick=50,
                zeroline=True,
                zerolinecolor=theme.RULE_2,
            ),
        ),
    )


def spectrum_figure() -> go.Figure:
    """Espectro de la ventana completa con la banda alfa sombreada."""
    figure = go.Figure(
        data=[go.Scatter(x=[], y=[], mode="lines", line={"color": theme.INK_2, "width": 1.5})],
        layout=_base_layout(
            xaxis=_axis(range=list(SPECTRUM_DISPLAY_HZ), ticksuffix=" Hz"),
            yaxis=_axis(ticksuffix=" dB", nticks=4),
        ),
    )
    figure.add_vrect(
        x0=IAF_DEFAULT_HZ - ALPHA_HALF_WIDTH_HZ,
        x1=IAF_DEFAULT_HZ + ALPHA_HALF_WIDTH_HZ,
        fillcolor=theme.COOL_SOFT,
        opacity=1,
        line_width=0,
        layer="below",
    )
    return figure


def spark_figure() -> go.Figure:
    """Traza mínima de los últimos 2 s, para la etapa "Señal" del flujo."""
    hidden = {"visible": False, "fixedrange": True}
    return go.Figure(
        data=[go.Scatter(x=[], y=[], mode="lines", line={"color": theme.COOL, "width": 1.2})],
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
        className="mm-tab is-active" if active else "mm-tab",
        **{"aria-pressed": "true" if active else "false"},
    )


def _header() -> html.Header:
    return html.Header(
        className="mm-bar",
        children=[
            html.Div(
                className="mm-brand",
                children=[
                    html.Span(text("marca", "nombre"), className="mm-brand-name"),
                    html.Span(text("marca", "respaldo"), className="mm-brand-by"),
                ],
            ),
            html.Nav(
                className="mm-tabs",
                children=[_audience_button(key, key == "persona") for key in AUDIENCES],
            ),
            html.Div(
                className="mm-bar-end",
                children=[
                    html.Span(
                        text("fuente", "sin_fuente"), id="mm-source", className="mm-chip is-off"
                    ),
                    html.Button(text("operacion", "abrir"), id="mm-op-open", className="mm-btn"),
                ],
            ),
        ],
    )


def _panel_head(title: str, meta: str | None = None, extra: list | None = None) -> html.Div:
    children = [html.H2(title)]
    if meta:
        children.append(html.Span(meta, className="mm-meta"))
    return html.Div(className="mm-panel-head", children=children + (extra or []))


def _excluded_chip() -> html.Span:
    # Solo se ve cuando el panel trae la clase is-excluded.
    return html.Span(text("avisos", "excluida"), className="mm-chip is-bad mm-excluded-chip")


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


def _persona_view() -> html.Section:
    return html.Section(
        id="mm-view-persona",
        className="mm-view mm-persona",
        children=[
            html.Div(
                className="mm-panel mm-span-trace",
                children=[
                    _panel_head(
                        text("persona", "senal"),
                        text("persona", "senal_meta"),
                    ),
                    _graph("mm-trace", trace_figure(), "100%"),
                ],
            ),
            html.Div(
                id="mm-spectrum-card",
                className="mm-panel mm-span-spectrum",
                children=[
                    _panel_head(
                        text("persona", "espectro"),
                        text("persona", "espectro_meta"),
                        [_excluded_chip()],
                    ),
                    _graph("mm-spectrum", spectrum_figure(), "100%"),
                ],
            ),
            html.Div(
                className="mm-panel mm-span-reading mm-reading-panel",
                children=[
                    html.Div(visitor("desconectada"), id="mm-gate", className="mm-gate"),
                    _panel_head(text("persona", "indicador")),
                    html.Div(id="mm-zone", className="mm-zone is-empty"),
                    _gauge(),
                    html.P(id="mm-status", className="mm-status"),
                    html.Div(
                        id="mm-progress",
                        className="mm-progress is-hidden",
                        children=html.Div(id="mm-progress-fill", className="mm-progress-fill"),
                    ),
                    html.Div(
                        className="mm-reading-foot",
                        children=[
                            html.Button(
                                text("persona", "boton_medir"),
                                id="mm-baseline",
                                className="mm-btn is-primary",
                            ),
                            html.Span(text("persona", "linea_base_meta"), className="mm-meta"),
                        ],
                    ),
                ],
            ),
        ],
    )


def _group_button(key: str, label: str, active: bool) -> html.Button:
    return html.Button(
        label,
        id={"type": "mm-group", "key": key},
        className="mm-seg is-active" if active else "mm-seg",
    )


def _flow_row(key: str, value: object) -> html.Div:
    return html.Div(
        className="mm-flow-row",
        children=[html.Span(text("organizacion", "etapas", key), className="mm-flow-name"), value],
    )


def _live_value(index: int) -> html.Span:
    return html.Span(
        text("organizacion", "vivo", "sin_conexion"),
        id=f"mm-stage-live-{index}",
        className="mm-measured",
    )


def _app_row(key: str) -> html.Tr:
    app = text("organizacion", "apps", key)
    action = (
        html.Button(
            text("organizacion", "probar"),
            id=f"mm-try-{DEMO_CARDS[key]}",
            className="mm-btn is-small",
        )
        if key in DEMO_CARDS
        else None
    )
    return html.Tr(
        className=f"is-{app['estado']}",
        children=[
            html.Td(app["titulo"]),
            html.Td(text("organizacion", "estados", app["estado"]), className="mm-app-state"),
            html.Td(action, className="mm-app-action"),
        ],
    )


def _demo_overlay() -> html.Div:
    """Demostraciones en vivo que abren las filas con "Probar"."""
    control = text("organizacion", "demos", "control")
    stimulus = text("organizacion", "demos", "estimulo")
    return html.Div(
        id="mm-demo",
        className="mm-demo is-hidden",
        children=html.Div(
            className="mm-demo-card",
            children=[
                html.Button(
                    text("organizacion", "demos", "cerrar"), id="mm-demo-close", className="mm-btn"
                ),
                html.Div(
                    id="mm-demo-control",
                    className="mm-demo-panel is-hidden",
                    children=[
                        html.H2(control["titulo"]),
                        html.P(control["instruccion"], className="mm-lead"),
                        html.Div(
                            className="mm-lamp-row",
                            children=[
                                html.Div(id="mm-lamp", className="mm-lamp"),
                                html.Div(
                                    [
                                        html.Div(
                                            control["apagada"],
                                            id="mm-lamp-label",
                                            className="mm-big",
                                        ),
                                        html.Span(id="mm-double-count", className="mm-measured"),
                                    ]
                                ),
                            ],
                        ),
                        html.P(control["nota"], className="mm-meta"),
                    ],
                ),
                html.Div(
                    id="mm-demo-estimulo",
                    className="mm-demo-panel is-hidden",
                    children=[
                        html.H2(stimulus["titulo"]),
                        html.P(stimulus["instruccion"], className="mm-lead"),
                        html.Button(stimulus["boton"], id="mm-mark", className="mm-btn is-primary"),
                        _graph("mm-stimulus", stimulus_figure(), "38vh"),
                        html.Div(
                            className="mm-demo-foot",
                            children=[
                                html.Span(id="mm-stim-count", className="mm-measured"),
                                html.Span(stimulus["nota"], className="mm-meta"),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    )


def _company_view() -> html.Section:
    groups = text("empresa", "grupos")
    apps = list(text("organizacion", "apps").keys())
    return html.Section(
        id="mm-view-empresa",
        className="mm-view mm-org is-hidden",
        children=[
            html.Div(
                id="mm-flow",
                className="mm-panel mm-flow",
                children=[
                    _panel_head(text("organizacion", "flujo")),
                    _flow_row("sensor", _live_value(1)),
                    _flow_row("senal", _graph("mm-spark", spark_figure(), "34px")),
                    _flow_row("limpieza", _live_value(3)),
                    _flow_row("analisis", _live_value(4)),
                    _flow_row(
                        "resultado",
                        html.Span(text("organizacion", "vivo", "resultado"), className="mm-meta"),
                    ),
                ],
            ),
            html.Div(
                className="mm-panel mm-apps",
                children=[
                    _panel_head(text("organizacion", "apps_titulo")),
                    html.Table(
                        className="mm-table", children=html.Tbody([_app_row(k) for k in apps])
                    ),
                ],
            ),
            html.Div(
                className="mm-panel mm-privacy",
                children=[
                    _panel_head(
                        text("organizacion", "privacidad"),
                        text("empresa", "aviso_simulados"),
                    ),
                    html.Div(
                        className="mm-privacy-body",
                        children=[
                            html.Div(
                                className="mm-segs",
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
            _demo_overlay(),
        ],
    )


def _source_button(kind: str, label: str, active: bool) -> html.Button:
    return html.Button(
        label,
        id=f"mm-kind-{kind}",
        className="mm-seg is-active" if active else "mm-seg",
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
                    html.Button(op["cerrar"], id="mm-op-close", className="mm-btn"),
                ],
            ),
            html.Div(op["fuente"], className="mm-op-label"),
            html.Div(
                className="mm-segs",
                role="group",
                children=[
                    _source_button("real", op["diadema"], True),
                    _source_button("sim", op["simulada"], False),
                ],
            ),
            html.Label(op["puerto"], htmlFor="mm-port", className="mm-op-label"),
            dcc.Input(id="mm-port", type="text", placeholder="COM5", className="mm-input"),
            html.Div(
                className="mm-op-actions",
                children=[
                    html.Button(op["conectar"], id="mm-connect", className="mm-btn is-primary"),
                    html.Button(op["detener"], id="mm-stop", className="mm-btn"),
                ],
            ),
            html.Div(id="mm-op-status", className="mm-op-status"),
            html.Div(id="mm-op-quality", className="nd-quality mm-op-quality"),
            html.P(id="mm-op-iaf", className="mm-meta"),
            html.P(op["guion_simulado"], className="mm-meta"),
            html.Div(
                className="mm-op-actions",
                children=[html.Button(op["simular_doble"], id="mm-sim-double", className="mm-btn")],
            ),
            html.Div(
                className="mm-op-actions",
                children=[
                    # Lo controla assets/mindmetric.js; las etiquetas viven aquí, en el contenido.
                    html.Button(
                        op["rotar_activar"],
                        id="mm-autoplay",
                        className="mm-btn",
                        **{
                            "data-label-on": op["rotar_activar"],
                            "data-label-off": op["rotar_desactivar"],
                        },
                    ),
                ],
            ),
            html.P(op["rotar_nota"], className="mm-meta"),
        ],
    )


def group_card(group_key: str) -> list:
    """Lectura de un grupo con las reglas de agregación ya aplicadas."""
    size = next(g.size for g in GROUPS if g.key == group_key)
    result = report(size, _pilot.participants(group_key, CURRENT_WEEK))
    if result.kind is ReportKind.SUPPRESSED:
        value = html.Div(text("empresa", "oculto"), className="mm-big is-muted")
    elif result.kind is ReportKind.RANGE:
        value = html.Div(text("empresa", "rangos", result.range_key), className="mm-big is-text")
    else:
        value = html.Div(f"{result.percent} %", className="mm-reading")
    return [
        html.Span(text("empresa", "tamano").format(n=size), className="mm-meta"),
        value,
        html.Span(text("empresa", "reglas", result.kind.value), className="mm-meta"),
    ]


layout = html.Div(
    className="mm-root",
    children=[
        _header(),
        html.Main(
            className="mm-main",
            children=[_persona_view(), _company_view(), measurement_view(_funnel.counts)],
        ),
        html.Footer(className="mm-foot", children=html.Span(text("limites"))),
        _operator_panel(),
        dcc.Interval(id="mm-interval", interval=TICK_MS, disabled=True),
        dcc.Store(id="mm-session", data=None),
        dcc.Store(id="mm-source-kind", data="real"),
        dcc.Store(id="mm-audience", data="persona"),
        dcc.Store(id="mm-demo-kind", data=None),
        dcc.Store(id="mm-ack", data=None),
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
    view_class = {
        "persona": "mm-view mm-persona",
        "empresa": "mm-view mm-org",
        "medimos": "mm-view mm-medimos",
    }
    views = [view_class[k] if k == chosen else view_class[k] + " is-hidden" for k in AUDIENCES]
    buttons = ["mm-tab is-active" if key == chosen else "mm-tab" for key in AUDIENCES]
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
    on, off = "mm-seg is-active", "mm-seg"
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
    Input("mm-session", "data"),
)
def show_source(name):
    """La fuente se declara en pantalla; el aviso sale del nombre de la sesion."""
    if not name:
        return text("fuente", "sin_fuente"), "mm-chip is-off"
    if name == SIM_NAME:
        return text("fuente", "simulada"), "mm-chip is-sim"
    return text("fuente", "en_vivo"), "mm-chip is-live"


@dash.callback(
    Output("mm-spectrogram", "figure"),
    Output("mm-fit", "figure"),
    Output("mm-track", "figure"),
    Output("mm-quality", "children"),
    Output("mm-exponent", "children"),
    Output("mm-funnel", "children"),
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
        funnel = funnel_children(counts)
    return (
        spectrogram,
        fit_patch(snap),
        track_patch(view),
        quality_children(view),
        exponent,
        funnel,
        _card_class(snap, view.clean_fraction is not None, "mm-panel mm-span-fit"),
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
    Output("mm-demo", "className"),
    Output("mm-demo-control", "className"),
    Output("mm-demo-estimulo", "className"),
    Output("mm-demo-kind", "data"),
    Input("mm-try-control", "n_clicks"),
    Input("mm-try-estimulo", "n_clicks"),
    Input("mm-demo-close", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_demo(*_clicks):
    """Abre la demostración de la tarjeta elegida, o la cierra."""
    kind = str(ctx.triggered_id).removeprefix("mm-try-")
    if kind not in ("control", "estimulo"):
        hidden = "mm-demo-panel is-hidden"
        return "mm-demo is-hidden", hidden, hidden, None
    panels = [
        "mm-demo-panel" if k == kind else "mm-demo-panel is-hidden" for k in ("control", "estimulo")
    ]
    return "mm-demo", panels[0], panels[1], kind


@dash.callback(
    Output("mm-ack", "data", allow_duplicate=True),
    Input("mm-mark", "n_clicks"),
    State("mm-session", "data"),
    prevent_initial_call=True,
)
def mark_stimulus(clicks, name):
    """Registra el instante del estímulo en el historial de la sesión."""
    with _state_lock:
        history = _histories.get(name) if name else None
        if history is not None:
            history.mark_stimulus(time.monotonic())
    return clicks


@dash.callback(
    Output("mm-ack", "data", allow_duplicate=True),
    Input("mm-sim-double", "n_clicks"),
    State("mm-session", "data"),
    prevent_initial_call=True,
)
def simulate_double_blink(clicks, name):
    """Solo con la señal simulada: agenda un doble parpadeo para ensayar."""
    session = registry.get(name) if name == SIM_NAME else None
    if session is not None and hasattr(session.source, "request_double_blink"):
        session.source.request_double_blink()
    return clicks


@dash.callback(
    Output("mm-lamp", "className"),
    Output("mm-lamp-label", "children"),
    Output("mm-double-count", "children"),
    Output("mm-stimulus", "figure"),
    Output("mm-stim-count", "children"),
    Input("mm-interval", "n_intervals"),
    State("mm-session", "data"),
    State("mm-demo-kind", "data"),
    prevent_initial_call=True,
)
def refresh_demo(n, name, kind):
    """Actualiza la demostración abierta. La luz responde en cada ciclo."""
    if not kind or not name:
        return (no_update,) * 5
    with _state_lock:
        snap = _last_snapshots.get(name)
        history = _histories.get(name)
        response = history.stimulus_response(time.monotonic()) if history else None
    return demo_outputs(kind, snap, response, n or 0)


def demo_outputs(kind: str, snap: Snapshot | None, response, n: int) -> tuple:
    """Salidas de la demostración abierta; puro para poder probarlo."""
    control = text("organizacion", "demos", "control")
    lamp = label = count = graph = stim = no_update
    if kind == "control" and snap is not None:
        on = snap.double_blink_count % 2 == 1
        lamp = "mm-lamp is-on" if on else "mm-lamp"
        label = control["encendida"] if on else control["apagada"]
        count = control["contador"].format(n=snap.double_blink_count)
    if kind == "estimulo" and response is not None and n % MEASURE_EVERY == 0:
        graph = stimulus_patch(response)
        stim = text("organizacion", "demos", "estimulo", "conteo").format(
            n=len(response.curves), p=response.pending
        )
    return lamp, label, count, graph, stim


@dash.callback(
    Output("mm-group-card", "children"),
    Output({"type": "mm-group", "key": ALL}, "className"),
    Input({"type": "mm-group", "key": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def show_group(_clicks):
    """Aplica las reglas de agregacion al grupo elegido."""
    key = ctx.triggered_id["key"] if ctx.triggered_id else GROUPS[0].key
    classes = ["mm-seg is-active" if g.key == key else "mm-seg" for g in GROUPS]
    return group_card(key), classes


# ============================================================== auxiliares


def _card_class(snap: Snapshot, ready: bool, base: str = "mm-panel mm-span-spectrum") -> str:
    """Marco rojo ligero cuando lo que se dibuja no entró al análisis."""
    excluded = ready and snap.display_spectrum_db is not None and not snap.window_used
    return base + " is-excluded" if excluded else base


def organization_live(snap: Snapshot | None, view, ready: bool, rate_hz: float) -> tuple:
    """Textos en vivo de las etapas 1, 3 y 4 del flujo, y la clase del flujo."""
    live = text("organizacion", "vivo")
    if snap is None or not ready:
        off = live["sin_conexion"]
        return off, off, off, "mm-panel mm-flow"
    sensor = live["tramas"].format(n=f"{rate_hz:.0f}")
    useful = 0.0 if view is None or view.clean_fraction is None else 100 * view.clean_fraction
    cleaning = live["limpieza"].format(n=snap.blink_count, p=f"{useful:.0f}")
    analysis = live["sin_base"] if snap.zone is None else text("persona", "zonas", snap.zone.value)
    return sensor, cleaning, analysis, "mm-panel mm-flow is-live"


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
