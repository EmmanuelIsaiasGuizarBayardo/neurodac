"""Historial en memoria para la vista "Como lo medimos" del demo de MindMetric.

Guarda lo necesario para dibujar el ultimo minuto de espectrograma, los
ultimos minutos del indicador y la calidad de la senal. Todo vive en memoria y
se descarta al desconectar: no se escribe a disco, como el resto del demo.

`StandFunnel` lleva conteos del stand (cuantas lineas base se iniciaron, cuantas
se completaron y cuantas llegaron a un minuto de practica). Son conteos sin
ningun dato de la persona, y se pierden al cerrar la aplicacion.

Puro: no lee el reloj ni abre puertos; recibe el tiempo y los Snapshot.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .mindmetric_analysis import MetricStatus, Snapshot

__all__ = [
    "HISTORY_STEP_S",
    "PRACTICE_GOAL_S",
    "QUALITY_WINDOW_S",
    "SPECTROGRAM_S",
    "TRACK_S",
    "FunnelCounts",
    "HistoryView",
    "SessionHistory",
    "StandFunnel",
]

HISTORY_STEP_S = 0.25
SPECTROGRAM_S = 60.0
TRACK_S = 120.0
QUALITY_WINDOW_S = 60.0
#: Un minuto de practica con senal util tras la linea base. El estudio de
#: referencia usa 120 s de tarea; en un stand, 60 s es lo realista.
PRACTICE_GOAL_S = 60.0

#: Estados en los que la ventana actual se uso (no se descarto).
_USABLE = (MetricStatus.LIVE, MetricStatus.BASELINE, MetricStatus.NO_BASELINE)


@dataclass(frozen=True)
class HistoryView:
    """Lo que la vista dibuja, con el tiempo relativo al presente (0 = ahora).

    Attributes
    ----------
    spec_t : numpy.ndarray
        Tiempo de cada columna del espectrograma, en segundos (negativos).
    spec_hz : numpy.ndarray or None
        Frecuencias de las filas; None si aun no hay espectro.
    spec_db : numpy.ndarray or None
        Matriz frecuencia por tiempo, en dB; NaN donde la ventana se descarto.
    track_t, track_z : numpy.ndarray
        Indicador respecto a la linea base; NaN donde no hubo dato.
    baseline_spans : list of tuple of float
        Intervalos (inicio, fin) de las lineas base, relativos al presente.
    clean_fraction : float or None
        Fraccion del ultimo minuto con ventana util, entre los instantes con
        senal lista. None si no hubo senal lista.
    blinks : int
        Parpadeos detectados desde que se conecto.
    time_to_ready_s : float or None
        Segundos desde conectar hasta tener senal lista por primera vez.
    """

    spec_t: np.ndarray
    spec_hz: np.ndarray | None
    spec_db: np.ndarray | None
    track_t: np.ndarray
    track_z: np.ndarray
    baseline_spans: list[tuple[float, float]]
    clean_fraction: float | None
    blinks: int
    time_to_ready_s: float | None


class SessionHistory:
    """Muestras espaciadas cada `HISTORY_STEP_S` de una conexion.

    Parameters
    ----------
    connected_at : float
        Tiempo monotonico de la conexion.
    """

    def __init__(self, connected_at: float) -> None:
        self._connected_at = connected_at
        self._ready_at: float | None = None
        self._last_at: float | None = None
        self._spec: deque[tuple[float, np.ndarray | None]] = deque()
        self._track: deque[tuple[float, float | None]] = deque()
        self._usable: deque[tuple[float, bool]] = deque()
        self._spans: list[list[float]] = []
        self._baseline_was_active = False
        self._spec_hz: np.ndarray | None = None
        self._blinks = 0

    def record(self, snap: Snapshot, now: float, ready: bool) -> None:
        """Incorpora el Snapshot del instante `now`, a lo mas cada `HISTORY_STEP_S`."""
        if ready and self._ready_at is None:
            self._ready_at = now
        self._blinks = snap.blink_count
        self._track_baseline(snap, now)
        if self._last_at is not None and now - self._last_at < HISTORY_STEP_S:
            return
        self._last_at = now

        usable = ready and snap.status in _USABLE
        column = None
        if usable and snap.spectrum_db is not None:
            column = np.asarray(snap.spectrum_db, dtype=float)
            self._spec_hz = np.asarray(snap.spectrum_hz, dtype=float)
        self._spec.append((now, column))

        z = snap.z_display if snap.status is MetricStatus.LIVE else None
        self._track.append((now, z))
        if ready:
            self._usable.append((now, usable))
        self._trim(now)

    def view(self, now: float) -> HistoryView:
        """Arma las matrices y series para dibujar en el instante `now`."""
        spec_t = np.array([t - now for t, _ in self._spec])
        spec_db = None
        if self._spec_hz is not None and self._spec:
            n_hz = self._spec_hz.size
            columns = [
                c if c is not None and c.size == n_hz else np.full(n_hz, np.nan)
                for _, c in self._spec
            ]
            spec_db = np.column_stack(columns)
        track_t = np.array([t - now for t, _ in self._track])
        track_z = np.array([np.nan if z is None else z for _, z in self._track])
        spans = [
            (start - now, (end if end is not None else now) - now) for start, end in self._spans
        ]
        recent = [ok for t, ok in self._usable if now - t <= QUALITY_WINDOW_S]
        clean = float(np.mean(recent)) if recent else None
        ready_s = None if self._ready_at is None else self._ready_at - self._connected_at
        return HistoryView(
            spec_t=spec_t,
            spec_hz=self._spec_hz,
            spec_db=spec_db,
            track_t=track_t,
            track_z=track_z,
            baseline_spans=spans,
            clean_fraction=clean,
            blinks=self._blinks,
            time_to_ready_s=ready_s,
        )

    def _track_baseline(self, snap: Snapshot, now: float) -> None:
        """Abre un intervalo al empezar una linea base y lo cierra al terminar."""
        if snap.baseline_active and not self._baseline_was_active:
            self._spans.append([now, None])
        if not snap.baseline_active and self._baseline_was_active and self._spans:
            self._spans[-1][1] = now
        self._baseline_was_active = snap.baseline_active

    def _trim(self, now: float) -> None:
        while self._spec and now - self._spec[0][0] > SPECTROGRAM_S:
            self._spec.popleft()
        while self._track and now - self._track[0][0] > TRACK_S:
            self._track.popleft()
        while self._usable and now - self._usable[0][0] > QUALITY_WINDOW_S:
            self._usable.popleft()
        self._spans = [s for s in self._spans if s[1] is None or now - s[1] <= TRACK_S]


@dataclass(frozen=True)
class FunnelCounts:
    """Conteos del stand en cada etapa."""

    started: int
    completed: int
    practiced: int


class StandFunnel:
    """Embudo de lineas base del stand, solo con conteos.

    Cada "Medir linea base" abre un intento. El intento cuenta como completo
    cuando la linea base termina y como practicado cuando acumula
    `PRACTICE_GOAL_S` de indicador en vivo. Volver a medir abandona el intento
    anterior en la etapa en que iba, como en un diagrama de flujo de estudio.
    """

    def __init__(self) -> None:
        self._started = 0
        self._completed = 0
        self._practiced = 0
        self._stage: str | None = None
        self._practice_s = 0.0
        self._last_at: float | None = None

    def baseline_started(self) -> None:
        """Registra un intento nuevo."""
        self._started += 1
        self._stage = "started"
        self._practice_s = 0.0
        self._last_at = None

    def observe(self, snap: Snapshot, now: float) -> None:
        """Avanza el intento actual segun el Snapshot."""
        dt = 0.0 if self._last_at is None else min(now - self._last_at, 1.0)
        self._last_at = now
        live = snap.status is MetricStatus.LIVE
        if self._stage == "started" and not snap.baseline_active and live:
            self._completed += 1
            self._stage = "completed"
        elif self._stage == "completed" and live:
            self._practice_s += dt
            if self._practice_s >= PRACTICE_GOAL_S:
                self._practiced += 1
                self._stage = "done"

    @property
    def counts(self) -> FunnelCounts:
        """Conteos actuales."""
        return FunnelCounts(self._started, self._completed, self._practiced)
