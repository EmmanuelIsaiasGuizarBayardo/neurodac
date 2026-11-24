"""Analisis en vivo del demo provisional de MindMetric.

Version reducida de las decisiones de senal de CortiThym (filtro causal con
estado, espectro de 2 s suavizado, ajuste aperiodico, potencia periodica en la
IAF +/- 2 Hz y z robusto contra la linea base de la propia persona). Es puro:
no abre puertos, no lanza hilos y no lee el reloj; quien llama pasa el tiempo,
de modo que se prueba con un reloj controlado.

Simplificaciones declaradas frente al plan original:

- Linea base abreviada de 30 s con ojos abiertos (el plan pide 60 + 60 s).
- El espectro usa el tramo limpio contiguo mas largo de la ventana, no la
  union de subsegmentos limpios.
- El ajuste aperiodico es una recta en log-log que excluye la banda alfa,
  no `specparam`.
- La IAF sale de la linea base con ojos abiertos; sin pico claro se usa 10 Hz.
- Los parpadeos se detectan con un umbral adaptativo en 0.5 a 6 Hz, no con el
  detector de artefactos del plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from scipy import signal

__all__ = [
    "FS",
    "RAW_TO_UV",
    "LiveAnalyzer",
    "MetricStatus",
    "Snapshot",
    "Zone",
    "aperiodic_residual",
    "estimate_iaf",
    "exponential_smoothing",
    "longest_clean_run",
    "periodic_band_power",
    "spectrum",
    "zone_for",
]

FS = 512.0

#: Cuentas ThinkGear a microvoltios: raw * (1.8 / 4096) / 2000 * 1e6.
#: Pendiente de verificar contra la documentacion de NeuroSky.
RAW_TO_UV = 1.8 / 4096 / 2000 * 1e6

BANDPASS_HZ = (1.0, 45.0)
BANDPASS_ORDER = 4
NOTCH_HZ = 60.0
NOTCH_Q = 30.0

DISPLAY_S = 4.0
DISPLAY_DECIMATION = 2
WINDOW_S = 2.0
NFFT = 1024
MIN_CLEAN_S = 1.0
SMOOTHING_TAU_S = 0.5

#: Criterio de banda ancha (1 a 45 Hz): movimientos y rafagas grandes.
ARTIFACT_UV = 100.0
ARTIFACT_PAD_S = 0.2

#: Detector de parpadeos, aparte del filtro de 1 a 45 Hz. Ese filtro deja un
#: parpadeo de 150 uV en 64 a 96 uV segun su duracion, asi que el umbral de
#: 100 uV se perdia los parpadeos suaves. En 0.5 a 6 Hz el parpadeo conserva mas
#: amplitud y la alfa de ojos cerrados queda fuera de la banda.
BLINK_BAND_HZ = (0.5, 6.0)
BLINK_ORDER = 2
#: Umbral adaptativo: BLINK_K desviaciones robustas del fondo de la propia
#: persona, acotado entre un piso y un techo. El piso evita marcar fondo limpio;
#: el techo garantiza que un parpadeo grande se marque aunque el fondo sea ruidoso.
BLINK_K = 4.0
BLINK_FLOOR_UV = 40.0
BLINK_CEIL_UV = 150.0
#: Escala de MAD a desviacion tipica bajo normalidad.
MAD_TO_SD = 1.4826

HOLD_S = 2.0
BASELINE_S = 30.0
#: Con +/- 1.5 desviaciones robustas, en reposo la senal queda fuera de la zona
#: cerca de 13% del tiempo si se distribuye normal; con +/- 1, un tercio.
ZONE_Z = 1.5
#: Suavizado del indicador, aplicado igual en la linea base y en vivo.
METRIC_TAU_S = 1.0
Z_DISPLAY_LIMIT = 3.0
#: Piso de la dispersion de la linea base, en dB. Con 30 s autocorrelacionados
#: el IQR se subestima a menudo y la zona parpadearia. Con este piso, salir de
#: la zona exige 1.5 * 0.75 = 1.1 dB, menos que el +50% de alfa (1.8 dB) que
#: busca el hito del efecto Berger.
MIN_ROBUST_SD_DB = 0.75

APERIODIC_RANGE_HZ = (2.0, 40.0)
IAF_SEARCH_HZ = (7.0, 13.0)
IAF_DEFAULT_HZ = 10.0
#: Umbral del pico alfa sobre el residuo suavizado (log10). Medido con ruido
#: blanco sin alfa y 30 semillas: el maximo en 7 a 13 Hz tuvo percentil 95 de
#: 0.18. Con 0.25, un pico tiene que ser casi el doble de la recta aperiodica.
IAF_MIN_PEAK_LOG10 = 0.25
#: Suavizado del residuo antes de buscar la IAF: 1 Hz con pesos de Hann. Una
#: caja de 2 Hz aplana un pico angosto y la IAF saltaria +/- 0.5 Hz.
IAF_SMOOTHING_KERNEL = (0.25, 0.5, 0.25)
ALPHA_HALF_WIDTH_HZ = 2.0
SPECTRUM_DISPLAY_HZ = (1.0, 40.0)

#: Escala del IQR a desviacion tipica bajo normalidad.
IQR_TO_SD = 1.349


class MetricStatus(StrEnum):
    """Que puede mostrar el indicador en este momento."""

    WAITING_SIGNAL = "esperando"
    NO_BASELINE = "sin_linea_base"
    BASELINE = "midiendo"
    LIVE = "en_vivo"
    HOLD = "ajustando"
    PAUSED = "pausa"


class Zone(StrEnum):
    """Bandas cualitativas respecto a la linea base de la propia persona."""

    BELOW = "debajo"
    WITHIN = "zona"
    ABOVE = "encima"


def zone_for(z: float) -> Zone:
    """Traduce un z robusto a una banda cualitativa; nunca a un puntaje."""
    if z < -ZONE_Z:
        return Zone.BELOW
    if z > ZONE_Z:
        return Zone.ABOVE
    return Zone.WITHIN


def longest_clean_run(clean: np.ndarray) -> tuple[int, int]:
    """Inicio y fin (exclusivo) del tramo contiguo limpio mas largo.

    Parameters
    ----------
    clean : numpy.ndarray of bool
        True donde la muestra no tiene artefacto.

    Returns
    -------
    tuple of int
        ``(inicio, fin)``; ``(0, 0)`` si no hay ninguna muestra limpia.
    """
    padded = np.concatenate(([False], clean.astype(bool), [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    if edges.size == 0:
        return 0, 0
    starts, ends = edges[0::2], edges[1::2]
    best = int(np.argmax(ends - starts))
    return int(starts[best]), int(ends[best])


def spectrum(segment: np.ndarray, fs: float = FS) -> tuple[np.ndarray, np.ndarray]:
    """Periodograma con ventana de Hann, rellenado a `NFFT` puntos.

    Un solo segmento, sin promediar: no es Welch. La varianza se reduce despues
    con el suavizado exponencial entre pasos.
    """
    freqs, psd = signal.periodogram(
        segment, fs=fs, window="hann", nfft=NFFT, detrend="constant", scaling="density"
    )
    return freqs, psd


def aperiodic_residual(
    freqs: np.ndarray, psd: np.ndarray, exclude: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Resta al espectro una recta aperiodica ajustada en log-log.

    Parameters
    ----------
    freqs, psd : numpy.ndarray
        Espectro lineal.
    exclude : tuple of float
        Banda que no participa del ajuste (la alfa esperada).

    Returns
    -------
    tuple of numpy.ndarray
        Frecuencias de `APERIODIC_RANGE_HZ` y residuo en log10, es decir, la
        parte periodica por encima (o por debajo) de la recta.

    Notes
    -----
    Sin pasada de exclusion de picos: con un periodograma de un segmento, la
    varianza por bin hace que esa pasada recorte ruido y no picos, y sesga la
    recta hacia abajo. El sesgo que queda (el logaritmo de un estimador
    ruidoso) es constante y se cancela al comparar contra la linea base.
    """
    lo, hi = APERIODIC_RANGE_HZ
    in_range = (freqs >= lo) & (freqs <= hi)
    f = freqs[in_range]
    log_f = np.log10(f)
    log_p = np.log10(psd[in_range] + 1e-20)
    fit_mask = ~((f >= exclude[0]) & (f <= exclude[1]))
    coef = np.polyfit(log_f[fit_mask], log_p[fit_mask], 1)
    return f, log_p - np.polyval(coef, log_f)


def periodic_band_power(f: np.ndarray, residual: np.ndarray, center_hz: float) -> float:
    """Potencia en ``center_hz +/- 2 Hz`` relativa a la recta aperiodica, en dB.

    Promedia en escala lineal y convierte al final. Promediar el logaritmo
    diluiria un pico angosto entre los bins vecinos que no lo tienen.
    """
    band = (f >= center_hz - ALPHA_HALF_WIDTH_HZ) & (f <= center_hz + ALPHA_HALF_WIDTH_HZ)
    return float(10.0 * np.log10(np.mean(10.0 ** residual[band])))


def estimate_iaf(f: np.ndarray, mean_residual: np.ndarray) -> float:
    """Pico alfa individual; `IAF_DEFAULT_HZ` si no hay pico claro.

    El residuo se suaviza a 1 Hz antes de buscar el maximo, para que un bin
    ruidoso no pase por pico.
    """
    smooth = np.convolve(mean_residual, IAF_SMOOTHING_KERNEL, mode="same")
    search = (f >= IAF_SEARCH_HZ[0]) & (f <= IAF_SEARCH_HZ[1])
    if not search.any():
        return IAF_DEFAULT_HZ
    idx = np.flatnonzero(search)[int(np.argmax(smooth[search]))]
    if smooth[idx] < IAF_MIN_PEAK_LOG10:
        return IAF_DEFAULT_HZ
    return float(f[idx])


def exponential_smoothing(values: np.ndarray, times: np.ndarray, tau_s: float) -> np.ndarray:
    """Media movil exponencial con paso de tiempo irregular."""
    out = np.empty_like(values, dtype=float)
    for i, value in enumerate(values):
        if i == 0:
            out[i] = value
            continue
        weight = 1.0 - np.exp(-max(0.0, times[i] - times[i - 1]) / tau_s)
        out[i] = (1.0 - weight) * out[i - 1] + weight * value
    return out


@dataclass
class _Baseline:
    """Linea base en curso o terminada."""

    residuals: list[np.ndarray] = field(default_factory=list)
    times: list[float] = field(default_factory=list)
    collected_s: float = 0.0
    done: bool = False
    iaf_hz: float = IAF_DEFAULT_HZ
    median_db: float = 0.0
    robust_sd_db: float = 1.0


@dataclass(frozen=True)
class Snapshot:
    """Lo que la pagina necesita dibujar en un instante.

    Attributes
    ----------
    trace_t, trace_uv : numpy.ndarray
        Senal filtrada de los ultimos `DISPLAY_S` segundos, diezmada para la
        vista. El analisis trabaja a `FS`.
    trace_artifact_uv : numpy.ndarray
        La misma senal solo donde hay artefacto, NaN en lo demas.
    spectrum_hz, spectrum_db : numpy.ndarray or None
        Espectro suavizado en dB, o None si aun no hay ventana.
    status : MetricStatus
        Estado del indicador.
    zone : Zone or None
        Banda cualitativa; None sin linea base o sin dato.
    z_display : float or None
        Posicion del marcador, recortada a +/- `Z_DISPLAY_LIMIT`.
    baseline_progress : float
        Avance de la linea base, de 0 a 1.
    baseline_active : bool
        True mientras se mide una linea base, aunque una ventana se descarte.
    blink_threshold_uv : float
        Umbral de parpadeo vigente; se muestra solo al personal.
    iaf_hz : float
        IAF en uso (10 Hz hasta terminar la linea base).
    """

    trace_t: np.ndarray
    trace_uv: np.ndarray
    trace_artifact_uv: np.ndarray
    spectrum_hz: np.ndarray | None
    spectrum_db: np.ndarray | None
    status: MetricStatus
    zone: Zone | None
    z_display: float | None
    baseline_progress: float
    baseline_active: bool
    blink_threshold_uv: float
    iaf_hz: float


class LiveAnalyzer:
    """Procesa bloques de muestras en uV y mantiene el estado entre ellos.

    Parameters
    ----------
    fs : float, optional
        Frecuencia de muestreo de la entrada.
    """

    def __init__(self, fs: float = FS) -> None:
        if fs <= 0:
            raise ValueError(f"fs debe ser > 0, no {fs}")
        self.fs = fs
        bandpass = signal.butter(BANDPASS_ORDER, BANDPASS_HZ, btype="bandpass", fs=fs, output="sos")
        b, a = signal.iirnotch(NOTCH_HZ, NOTCH_Q, fs=fs)
        self._sos = np.vstack([bandpass, signal.tf2sos(b, a)])
        self._zi: np.ndarray | None = None
        self._sos_blink = signal.butter(
            BLINK_ORDER, BLINK_BAND_HZ, btype="bandpass", fs=fs, output="sos"
        )
        self._zi_blink: np.ndarray | None = None
        self._blink_band = np.zeros(0)
        self._blink_threshold = BLINK_FLOOR_UV

        self._display_n = int(DISPLAY_S * fs)
        self._window_n = int(WINDOW_S * fs)
        self._pad_n = int(ARTIFACT_PAD_S * fs)
        self._filtered = np.zeros(0)
        self._over = np.zeros(0, dtype=bool)

        self._psd: np.ndarray | None = None
        self._freqs: np.ndarray | None = None
        self._last_psd_at: float | None = None
        self._last_good_at: float | None = None
        self._last_z: float | None = None
        self._metric: float | None = None
        self._last_metric_at: float | None = None
        self._baseline = _Baseline()
        self._baseline_active = False
        self._last_baseline_at: float | None = None

    # ---------------------------------------------------------- linea base

    def start_baseline(self) -> None:
        """Descarta la linea base anterior y empieza a medir una nueva."""
        self._baseline = _Baseline()
        self._baseline_active = True
        self._last_baseline_at = None
        self._last_z = None

    @property
    def has_baseline(self) -> bool:
        """True cuando ya hay una linea base terminada."""
        return self._baseline.done

    # -------------------------------------------------------------- datos

    def filter_block(self, samples_uv: np.ndarray) -> np.ndarray:
        """Filtra un bloque conservando el estado del filtro entre bloques."""
        x = np.asarray(samples_uv, dtype=float)
        if x.size == 0:
            return x
        if self._zi is None:
            # Arranca en el primer valor para no inyectar un escalon.
            self._zi = signal.sosfilt_zi(self._sos) * x[0]
        y, self._zi = signal.sosfilt(self._sos, x, zi=self._zi)
        return y

    def blink_band_block(self, samples_uv: np.ndarray) -> np.ndarray:
        """Filtra un bloque en la banda del detector de parpadeos, con estado."""
        x = np.asarray(samples_uv, dtype=float)
        if x.size == 0:
            return x
        if self._zi_blink is None:
            self._zi_blink = signal.sosfilt_zi(self._sos_blink) * x[0]
        y, self._zi_blink = signal.sosfilt(self._sos_blink, x, zi=self._zi_blink)
        return y

    @property
    def blink_threshold_uv(self) -> float:
        """Umbral de parpadeo vigente, en uV sobre la banda de 0.5 a 6 Hz."""
        return self._blink_threshold

    def update(self, samples_uv: np.ndarray, now: float, signal_ready: bool) -> Snapshot:
        """Incorpora un bloque y devuelve lo que hay que mostrar.

        Parameters
        ----------
        samples_uv : numpy.ndarray
            Muestras nuevas en uV, en orden.
        now : float
            Tiempo monotonico en segundos.
        signal_ready : bool
            True si el enlace con la diadema esta en estado listo. Mientras no
            lo este, nada que dependa de la senal avanza.
        """
        self._append(self.filter_block(samples_uv), self.blink_band_block(samples_uv))
        status, zone, z = self._evaluate(now, signal_ready)
        trace_t, trace, artifact = self._display_trace()
        spectrum_hz, spectrum_db = self._display_spectrum()
        return Snapshot(
            trace_t=trace_t,
            trace_uv=trace,
            trace_artifact_uv=artifact,
            spectrum_hz=spectrum_hz,
            spectrum_db=spectrum_db,
            status=status,
            zone=zone,
            z_display=None if z is None else float(np.clip(z, -Z_DISPLAY_LIMIT, Z_DISPLAY_LIMIT)),
            baseline_progress=min(1.0, self._baseline.collected_s / BASELINE_S),
            baseline_active=self._baseline_active,
            blink_threshold_uv=self._blink_threshold,
            iaf_hz=self._baseline.iaf_hz,
        )

    # ----------------------------------------------------------- internos

    def _append(self, filtered: np.ndarray, blink_band: np.ndarray) -> None:
        keep = max(self._display_n, self._window_n) + self._pad_n
        self._filtered = np.concatenate([self._filtered, filtered])[-keep:]
        self._blink_band = np.concatenate([self._blink_band, blink_band])[-keep:]
        self._blink_threshold = self._adaptive_blink_threshold()
        self._over = (np.abs(self._filtered) > ARTIFACT_UV) | (
            np.abs(self._blink_band) > self._blink_threshold
        )

    def _adaptive_blink_threshold(self) -> float:
        """BLINK_K desviaciones robustas del fondo, entre el piso y el techo.

        La MAD casi no se mueve con parpadeos esporadicos, que ocupan una
        fraccion chica del bufer; por eso describe el fondo y no al parpadeo.
        Con menos de 1 s de datos se usa el piso.
        """
        if self._blink_band.size < int(self.fs):
            return BLINK_FLOOR_UV
        band = self._blink_band
        robust_sd = MAD_TO_SD * float(np.median(np.abs(band - np.median(band))))
        return float(np.clip(BLINK_K * robust_sd, BLINK_FLOOR_UV, BLINK_CEIL_UV))

    def _clean_mask(self) -> np.ndarray:
        """Muestras sin artefacto, con margen alrededor de cada excedencia."""
        if not self._over.any():
            return np.ones(self._filtered.size, dtype=bool)
        kernel = np.ones(2 * self._pad_n + 1)
        dilated = np.convolve(self._over.astype(float), kernel, mode="same") > 0
        return ~dilated

    def _evaluate(self, now: float, ready: bool) -> tuple[MetricStatus, Zone | None, float | None]:
        if not ready:
            return MetricStatus.WAITING_SIGNAL, None, None
        if self._filtered.size < self._window_n:
            return MetricStatus.WAITING_SIGNAL, None, None

        residual = self._window_residual(now)
        if residual is None:
            return self._hold(now)
        f, res = residual
        self._last_good_at = now

        if self._baseline_active and not self._baseline.done:
            return self._collect_baseline(now, res)
        if not self._baseline.done:
            return MetricStatus.NO_BASELINE, None, None

        value = periodic_band_power(f, res, self._baseline.iaf_hz)
        if self._metric is None or self._last_metric_at is None:
            self._metric = value
        else:
            dt = max(0.0, now - self._last_metric_at)
            weight = 1.0 - np.exp(-dt / METRIC_TAU_S)
            self._metric = (1.0 - weight) * self._metric + weight * value
        self._last_metric_at = now
        z = (self._metric - self._baseline.median_db) / self._baseline.robust_sd_db
        self._last_z = z
        return MetricStatus.LIVE, zone_for(z), z

    def _window_residual(self, now: float) -> tuple[np.ndarray, np.ndarray] | None:
        """Actualiza el espectro suavizado; None si la ventana no sirve."""
        window = self._filtered[-self._window_n :]
        clean = self._clean_mask()[-self._window_n :]
        start, end = longest_clean_run(clean)
        if end - start < int(MIN_CLEAN_S * self.fs):
            return None
        freqs, psd = spectrum(window[start:end], self.fs)
        if self._psd is None or self._last_psd_at is None:
            self._psd = psd
        else:
            dt = max(0.0, now - self._last_psd_at)
            weight = 1.0 - np.exp(-dt / SMOOTHING_TAU_S)
            self._psd = (1.0 - weight) * self._psd + weight * psd
        self._freqs = freqs
        self._last_psd_at = now
        iaf = self._baseline.iaf_hz
        exclude = (iaf - ALPHA_HALF_WIDTH_HZ, iaf + ALPHA_HALF_WIDTH_HZ)
        return aperiodic_residual(freqs, self._psd, exclude)

    def _hold(self, now: float) -> tuple[MetricStatus, Zone | None, float | None]:
        """Sin ventana util: sostiene el ultimo valor un rato y luego pausa."""
        recent = self._last_good_at is not None and now - self._last_good_at <= HOLD_S
        if recent and self._last_z is not None:
            return MetricStatus.HOLD, zone_for(self._last_z), self._last_z
        if recent:
            return MetricStatus.HOLD, None, None
        return MetricStatus.PAUSED, None, None

    def _collect_baseline(
        self, now: float, residual: np.ndarray
    ) -> tuple[MetricStatus, Zone | None, float | None]:
        base = self._baseline
        step = 0.0 if self._last_baseline_at is None else now - self._last_baseline_at
        self._last_baseline_at = now
        base.residuals.append(residual)
        base.times.append(now)
        base.collected_s += min(step, HOLD_S)
        if base.collected_s >= BASELINE_S:
            self._finish_baseline()
        return MetricStatus.BASELINE, None, None

    def _finish_baseline(self) -> None:
        base = self._baseline
        lo, hi = APERIODIC_RANGE_HZ
        f = self._freqs[(self._freqs >= lo) & (self._freqs <= hi)]
        stack = np.vstack(base.residuals)
        base.iaf_hz = estimate_iaf(f, stack.mean(axis=0))
        raw = np.array([periodic_band_power(f, r, base.iaf_hz) for r in stack])
        values = exponential_smoothing(raw, np.asarray(base.times), METRIC_TAU_S)
        # El arranque del suavizado no representa el reposo: se descarta.
        settled = values[np.asarray(base.times) - base.times[0] >= METRIC_TAU_S]
        q25, q50, q75 = np.percentile(settled if settled.size else values, [25, 50, 75])
        base.median_db = float(q50)
        base.robust_sd_db = max(float(q75 - q25) / IQR_TO_SD, MIN_ROBUST_SD_DB)
        base.done = True
        self._metric = float(values[-1])
        self._last_metric_at = base.times[-1]
        self._baseline_active = False
        self._last_baseline_at = None

    def _display_trace(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        data = self._filtered[-self._display_n :]
        clean = self._clean_mask()[-self._display_n :]
        step = DISPLAY_DECIMATION
        shown = data[::step]
        artifact = np.where(clean[::step], np.nan, shown)
        # Eje en segundos respecto al presente: el ultimo punto es 0.
        t = (np.arange(shown.size) - shown.size + 1) * step / self.fs
        return t, shown, artifact

    def _display_spectrum(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self._psd is None or self._freqs is None:
            return None, None
        lo, hi = SPECTRUM_DISPLAY_HZ
        band = (self._freqs >= lo) & (self._freqs <= hi)
        return self._freqs[band], 10.0 * np.log10(self._psd[band] + 1e-20)
