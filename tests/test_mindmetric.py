"""Pruebas del demo provisional de MindMetric.

Siguen la regla de senal externa: ninguna necesita la diadema. El analisis se
prueba con senales de verdad conocida y reloj controlado, y el espectro se
valida contra una implementacion independiente (MNE-Python).
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import dash
import numpy as np
import pytest

from neurodac.content import mindmetric as text
from neurodac.mindmetric_analysis import (
    BASELINE_S,
    BLINK_CEIL_UV,
    BLINK_FLOOR_UV,
    FS,
    HOLD_S,
    IAF_DEFAULT_HZ,
    RAW_TO_UV,
    ZONE_Z,
    LiveAnalyzer,
    MetricStatus,
    Zone,
    aperiodic_residual,
    longest_clean_run,
    periodic_band_power,
    spectrum,
    zone_for,
)
from neurodac.mindmetric_panel import (
    CURRENT_WEEK,
    GROUPS,
    MIN_GROUP_SIZE,
    ReportKind,
    report,
    simulate_pilot,
)
from neurodac.mindmetric_sim import MindMetricSimulatedSource, eyes_closed_at
from neurodac.thinkgear import Code, ThinkGearParser

TICK_S = 0.15
PAGE = Path(__file__).resolve().parent.parent / "pages" / "mindmetric.py"


def _eeg(seconds: float, alpha_uv: float, *, seed: int = 0, alpha_hz: float = 10.0) -> np.ndarray:
    """EEG sintetico: ruido blanco mas una alfa senoidal de amplitud conocida."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * FS)) / FS
    return alpha_uv * np.sin(2 * np.pi * alpha_hz * t) + 4.0 * rng.standard_normal(t.size)


def _feed(analyzer: LiveAnalyzer, x: np.ndarray, t0: float, *, ready: bool = True):
    """Alimenta en bloques de TICK_S y devuelve (ultimo snapshot, tiempo final)."""
    block = int(TICK_S * FS)
    t, snap = t0, None
    for k in range(0, x.size, block):
        t += TICK_S
        snap = analyzer.update(x[k : k + block], t, ready)
    return snap, t


def _with_baseline(seed: int = 0) -> tuple[LiveAnalyzer, float]:
    analyzer = LiveAnalyzer()
    _, t = _feed(analyzer, _eeg(3, 5.0, seed=seed), 0.0)
    analyzer.start_baseline()
    _, t = _feed(analyzer, _eeg(BASELINE_S + 3, 5.0, seed=seed + 1), t)
    assert analyzer.has_baseline
    return analyzer, t


# ---------------------------------------------------------------- analisis


def test_conversion_a_microvoltios() -> None:
    assert RAW_TO_UV == pytest.approx(0.2197, abs=1e-4)


def test_filtro_por_bloques_igual_a_filtro_de_una_vez() -> None:
    x = _eeg(6, 10.0)
    whole = LiveAnalyzer().filter_block(x)
    blocks = LiveAnalyzer()
    pieces = [blocks.filter_block(x[k : k + 77]) for k in range(0, x.size, 77)]
    np.testing.assert_allclose(np.concatenate(pieces), whole, atol=1e-9)


def test_espectro_coincide_con_mne() -> None:
    mne_tf = pytest.importorskip("mne.time_frequency")
    x = _eeg(2, 10.0)
    x = x - x.mean()
    freqs, ours = spectrum(x)
    theirs, f_mne = mne_tf.psd_array_welch(
        x,
        FS,
        fmin=0,
        fmax=FS / 2,
        n_fft=1024,
        n_per_seg=1024,
        n_overlap=0,
        window="hann",
        verbose=False,
    )
    np.testing.assert_allclose(f_mne, freqs)
    band = (freqs >= 1) & (freqs <= 100)
    np.testing.assert_allclose(theirs[band], ours[band], rtol=1e-6)


def test_senoidal_de_10_hz_da_potencia_alfa_positiva() -> None:
    freqs, psd = spectrum(_eeg(2, 10.0))
    f, res = aperiodic_residual(freqs, psd, (8.0, 12.0))
    assert periodic_band_power(f, res, 10.0) > 3.0


def test_ruido_blanco_tiene_espectro_plano() -> None:
    # Promedio de muchos periodogramas: la prueba es sobre la resta aperiodica,
    # no sobre la varianza de un solo segmento.
    rng = np.random.default_rng(3)
    psds = [spectrum(rng.standard_normal(1024))[1] for _ in range(200)]
    freqs = spectrum(np.zeros(1024))[0]
    f, res = aperiodic_residual(freqs, np.mean(psds, axis=0), (8.0, 12.0))
    assert abs(periodic_band_power(f, res, 10.0)) < 0.5
    assert np.abs(res).max() < 0.15


def test_linea_base_estima_la_iaf() -> None:
    analyzer = LiveAnalyzer()
    _, t = _feed(analyzer, _eeg(3, 8.0, alpha_hz=9.5), 0.0)
    analyzer.start_baseline()
    snap, _ = _feed(analyzer, _eeg(BASELINE_S + 3, 8.0, seed=1, alpha_hz=9.5), t)
    assert snap.iaf_hz == pytest.approx(9.5, abs=0.5)


def test_sin_pico_claro_usa_10_hz() -> None:
    analyzer = LiveAnalyzer()
    _, t = _feed(analyzer, _eeg(3, 0.0), 0.0)
    analyzer.start_baseline()
    snap, _ = _feed(analyzer, _eeg(BASELINE_S + 3, 0.0, seed=2), t)
    assert snap.iaf_hz == IAF_DEFAULT_HZ


def _zones(analyzer: LiveAnalyzer, x: np.ndarray, t: float) -> list[Zone]:
    block = int(TICK_S * FS)
    zones = []
    for k in range(0, x.size, block):
        t += TICK_S
        snap = analyzer.update(x[k : k + block], t, True)
        if snap.status is MetricStatus.LIVE:
            zones.append(snap.zone)
    return zones


@pytest.mark.parametrize("seed", [0, 10, 20, 30])
def test_misma_senal_que_la_linea_base_queda_casi_siempre_en_la_zona(seed: int) -> None:
    analyzer, t = _with_baseline(seed)
    zones = _zones(analyzer, _eeg(30, 5.0, seed=seed + 9), t)
    assert zones.count(Zone.WITHIN) / len(zones) > 0.8


def test_mas_alfa_que_la_linea_base_queda_por_encima() -> None:
    analyzer, t = _with_baseline()
    zones = _zones(analyzer, _eeg(10, 15.0, seed=9), t)
    # Tras la ventana de 2 s y el suavizado, el indicador ya debe estar arriba.
    settled = zones[int(4.0 / TICK_S) :]
    assert settled.count(Zone.ABOVE) / len(settled) > 0.9


def test_sin_senal_lista_no_avanza_nada() -> None:
    analyzer = LiveAnalyzer()
    analyzer.start_baseline()
    snap, _ = _feed(analyzer, _eeg(10, 5.0), 0.0, ready=False)
    assert snap.status is MetricStatus.WAITING_SIGNAL
    assert snap.baseline_progress == 0.0


def test_artefacto_sostiene_y_luego_pausa() -> None:
    analyzer, t = _with_baseline()
    noisy = _eeg(HOLD_S + 2, 5.0, seed=4)
    # Parpadeos de 200 uV cada 0.6 s: ningun tramo limpio llega a 1 s.
    t_blink = np.arange(noisy.size) / FS
    for center in np.arange(0.3, t_blink[-1], 0.6):
        noisy += 200.0 * np.exp(-0.5 * ((t_blink - center) / 0.06) ** 2)
    block = int(TICK_S * FS)
    statuses = []
    for k in range(0, noisy.size, block):
        t += TICK_S
        statuses.append(analyzer.update(noisy[k : k + block], t, True).status)
    assert MetricStatus.HOLD in statuses
    assert statuses[-1] is MetricStatus.PAUSED


def _blinks(x: np.ndarray, centers_s: list[float], amp_uv: float, sigma_s: float) -> np.ndarray:
    """Suma parpadeos gaussianos de amplitud y duracion conocidas."""
    t = np.arange(x.size) / FS
    out = x.copy()
    for center in centers_s:
        out += amp_uv * np.exp(-0.5 * ((t - center) / sigma_s) ** 2)
    return out


def _marked_at(snap, total_s: float, centers_s: list[float]) -> list[bool]:
    """Si la traza mostrada marca artefacto en el centro de cada parpadeo."""
    marked = []
    for center in centers_s:
        idx = int(np.argmin(np.abs(snap.trace_t - (center - total_s))))
        marked.append(not np.isnan(snap.trace_artifact_uv[idx]))
    return marked


def test_parpadeo_suave_se_marca_aunque_el_filtro_lo_encoja() -> None:
    total, centers = 6.0, [2.5, 3.6, 4.8]
    x = _blinks(_eeg(total, 5.0, seed=5), centers, amp_uv=100.0, sigma_s=0.08)
    # El motivo del cambio: en 1 a 45 Hz este parpadeo ya no llega a 100 uV.
    assert np.abs(LiveAnalyzer().filter_block(x)).max() < 100.0
    snap, _ = _feed(LiveAnalyzer(), x, 0.0)
    assert all(_marked_at(snap, total, centers))


def test_fondo_limpio_con_alfa_no_se_marca() -> None:
    analyzer = LiveAnalyzer()
    snap, _ = _feed(analyzer, _eeg(6, 15.0, seed=6), 0.0)
    assert analyzer.blink_threshold_uv == BLINK_FLOOR_UV
    assert np.isnan(snap.trace_artifact_uv).mean() > 0.98


def test_fondo_ruidoso_sube_el_umbral_sin_pasar_el_techo() -> None:
    rng = np.random.default_rng(8)
    # Fondo lento y grande, como un electrodo seco con mal contacto.
    noisy = np.cumsum(rng.standard_normal(int(6 * FS))) * 2.0
    total, centers = 6.0, [3.0, 4.5]
    x = _blinks(noisy - noisy.mean(), centers, amp_uv=400.0, sigma_s=0.08)
    analyzer = LiveAnalyzer()
    snap, _ = _feed(analyzer, x, 0.0)
    assert BLINK_FLOOR_UV < analyzer.blink_threshold_uv <= BLINK_CEIL_UV
    assert all(_marked_at(snap, total, centers))


def test_simulador_marca_sus_parpadeos() -> None:
    source = MindMetricSimulatedSource()
    x = np.array([source.sample_uv((i + 1) / FS) for i in range(int(20 * FS))])
    snap, _ = _feed(LiveAnalyzer(), x, 0.0)
    assert not np.isnan(snap.trace_artifact_uv).all()


def test_tramo_limpio_mas_largo() -> None:
    assert longest_clean_run(np.array([1, 1, 0, 1, 1, 1, 0], dtype=bool)) == (3, 6)
    assert longest_clean_run(np.zeros(4, dtype=bool)) == (0, 0)
    assert longest_clean_run(np.ones(4, dtype=bool)) == (0, 4)


@pytest.mark.parametrize(
    ("z", "zone"),
    [
        (-ZONE_Z - 0.1, Zone.BELOW),
        (0.0, Zone.WITHIN),
        (ZONE_Z, Zone.WITHIN),
        (ZONE_Z + 0.1, Zone.ABOVE),
    ],
)
def test_bandas_cualitativas(z: float, zone: Zone) -> None:
    assert zone_for(z) is zone


# ------------------------------------------------------------------ panel


def test_grupo_chico_no_se_muestra() -> None:
    for participants in range(MIN_GROUP_SIZE):
        assert report(MIN_GROUP_SIZE - 1, participants).kind is ReportKind.SUPPRESSED


@pytest.mark.parametrize(("participants", "key"), [(0, "menos"), (3, "mitad"), (6, "mas")])
def test_grupo_mediano_va_en_rangos_y_oculta_los_extremos(participants: int, key: str) -> None:
    result = report(6, participants)
    assert result.kind is ReportKind.RANGE
    assert result.range_key == key
    assert result.percent is None


@pytest.mark.parametrize(("size", "participants", "percent"), [(20, 13, 70), (11, 7, 60)])
def test_grupo_grande_en_multiplos_de_10(size: int, participants: int, percent: int) -> None:
    assert report(size, participants).percent == percent


def test_reporte_rechaza_conteos_imposibles() -> None:
    with pytest.raises(ValueError):
        report(5, 6)
    with pytest.raises(ValueError):
        report(0, 0)


def test_piloto_simulado_es_reproducible() -> None:
    a, b = simulate_pilot(), simulate_pilot()
    np.testing.assert_array_equal(a.participated, b.participated)
    assert sum(g.size for g in GROUPS[1:]) == GROUPS[0].size
    assert len(a.weekly_report("todos")) == CURRENT_WEEK


def test_el_demo_recorre_las_tres_reglas() -> None:
    pilot = simulate_pilot()
    kinds = {report(g.size, pilot.participants(g.key, CURRENT_WEEK)).kind for g in GROUPS}
    assert kinds == set(ReportKind)


# -------------------------------------------------------------- simulador


def test_simulador_emite_tramas_thinkgear_validas() -> None:
    source = MindMetricSimulatedSource()
    source.open()
    time.sleep(0.12)
    events = ThinkGearParser().feed(source.read(0.05))
    source.close()
    raw = [e for e in events if e.code == Code.RAW_VALUE]
    assert raw, "el simulador debe emitir muestras crudas"


def test_simulador_sube_la_alfa_con_ojos_cerrados() -> None:
    source = MindMetricSimulatedSource()
    times = np.arange(1, int(70 * FS)) / FS
    x = np.array([source.sample_uv(t) for t in times])

    def alpha(t0: float) -> float:
        seg = x[int(t0 * FS) : int((t0 + 2) * FS)]
        freqs, psd = spectrum(seg - seg.mean())
        return float(psd[(freqs >= 8) & (freqs <= 12)].mean())

    assert eyes_closed_at(55.0) and not eyes_closed_at(45.0)
    assert alpha(55.0) > 4 * alpha(20.0)


# -------------------------------------------------------------- contenido


def test_cada_estado_y_zona_tiene_texto() -> None:
    for status in MetricStatus:
        assert text("persona", "estados", status.value)
    for zone in Zone:
        assert text("persona", "zonas", zone.value)


def test_cada_regla_grupo_y_rango_tiene_texto() -> None:
    for kind in ReportKind:
        assert text("empresa", "reglas", kind.value)
    for group in GROUPS:
        assert text("empresa", "grupos", group.key)
    for key in ("menos", "mitad", "mas"):
        assert text("empresa", "rangos", key)


# ----------------------------------------------------------------- pagina


@pytest.fixture(scope="module")
def page():
    dash.Dash(__name__, use_pages=True, pages_folder="")
    spec = importlib.util.spec_from_file_location("mindmetric_page", PAGE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pagina_arma_su_layout(page) -> None:
    assert page.layout is not None


def test_pagina_aplica_las_reglas_a_cada_grupo(page) -> None:
    for group in GROUPS:
        card = page.group_card(group.key)
        assert len(card) == 3


def _components(node) -> list:
    """Recorre el arbol de componentes de Dash."""
    found = [node]
    children = getattr(node, "children", None)
    if isinstance(children, list | tuple):
        for child in children:
            found.extend(_components(child))
    elif children is not None and hasattr(children, "children"):
        found.extend(_components(children))
    return found


def test_pagina_no_depende_de_bootstrap(page) -> None:
    # NeuroDAC carga su tema de Bootstrap desde una CDN: sin internet, sus
    # componentes pierden el estilo. Esta pagina debe verse igual sin red.
    modules = {type(c).__module__ for c in _components(page.layout)}
    assert not [m for m in modules if m.startswith("dash_bootstrap_components")]


def test_pagina_declara_la_senal_simulada(page) -> None:
    _, css, notice_css = page.show_source(page.SIM_NAME)
    assert "is-sim" in css
    assert "is-hidden" not in notice_css
