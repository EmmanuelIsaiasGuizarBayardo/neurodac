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
    "aperiodic_exponent",
    "aperiodic_fit",
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
#: Escala del percentil 25 de |x| a desviacion tipica bajo normalidad:
#: P(|x| < 0.3186 sigma) = 0.25. Se usa el percentil 25 y no la MAD (percentil
#: 50) porque con parpadeos seguidos la mitad del bufer puede ser parpadeo, y
#: entonces la MAD describe al parpadeo y el umbral sube hasta el techo.
LOW_QUARTILE_TO_SD = 1.0 / 0.31863936
BLINK_REFRACTORY_S = 0.25
#: Firma del parpadeo en la banda de 0.5 a 6 Hz: un lobulo hacia la polaridad
#: del parpadeo y, en menos de BLINK_CONFIRM_S, un lobulo de signo contrario
#: de al menos BLINK_CONFIRM_FACTOR del umbral (el que deja el filtro). Solo un
#: parpadeo tiene esa forma; el ruido o la cola de un parpadeo previo no. Asi
#: se evitan los dobles falsos sin pedir parpadeos exagerados.
BLINK_CONFIRM_S = 0.5
BLINK_CONFIRM_FACTOR = 0.5
#: El conteo usa un umbral mas bajo que el de marcar artefactos: en un doble
#: parpadeo rapido, el contralobulo del primero resta amplitud al segundo. La
#: confirmacion por forma es la que evita los falsos positivos.
BLINK_COUNT_FACTOR = 0.7
#: La polaridad se aprende de los ultimos parpadeos (el signo del primer
#: lobulo); al inicio se supone positiva, como en Fp1 con referencia en oreja.
BLINK_POLARITY_MEMORY = 15
BLINK_EVENT_GAP_S = 0.6
#: Doble parpadeo voluntario: dos parpadeos contados con no más de 1 s entre sí
#: (el mínimo lo pone el refractario de 0.4 s). Tras uno, 1.5 s sin detectar
#: otro, para que un tercer parpadeo no encadene una segunda orden.
DOUBLE_BLINK_MAX_S = 1.0
DOUBLE_BLINK_COOLDOWN_S = 1.5

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


def aperiodic_fit(
    freqs: np.ndarray, psd: np.ndarray, exclude: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recta aperiodica en log-log sobre `APERIODIC_RANGE_HZ`.

    Parameters
    ----------
    freqs, psd : numpy.ndarray
        Espectro lineal.
    exclude : tuple of float
        Banda que no participa del ajuste (la alfa esperada).

    Returns
    -------
    tuple of numpy.ndarray
        Frecuencias del rango, log10 del espectro y log10 de la recta.

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
    return f, log_p, np.polyval(coef, log_f)


def aperiodic_residual(
    freqs: np.ndarray, psd: np.ndarray, exclude: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Parte periodica: log10 del espectro menos la recta aperiodica.

    Returns
    -------
    tuple of numpy.ndarray
        Frecuencias de `APERIODIC_RANGE_HZ` y residuo en log10.
    """
    f, log_p, log_fit = aperiodic_fit(freqs, psd, exclude)
    return f, log_p - log_fit


def aperiodic_exponent(f: np.ndarray, log_fit: np.ndarray) -> float:
    """Exponente de la ley de potencia 1/f^x a partir de la recta ajustada."""
    slope = (log_fit[-1] - log_fit[0]) / (np.log10(f[-1]) - np.log10(f[0]))
    return float(-slope)


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
    blink_count : int
        Parpadeos detectados desde que se creo el analizador, con senal lista.
    double_blink_count : int
        Dobles parpadeos detectados: la orden de la demostracion de control.
    alpha_db : float or None
        Potencia alfa sobre el fondo aperiodico, suavizada, en dB, aunque no
        haya linea base. None si la ventana actual no se uso. Sirve para ver
        cambios relativos, como la respuesta a un estimulo.
    fit_hz, fit_db : numpy.ndarray or None
        Recta aperiodica del ultimo espectro util, en dB, sobre 2 a 40 Hz.
    exponent : float or None
        Exponente aperiodico de esa recta. Descriptivo: lo mueven el musculo y
        el contacto, no solo el cerebro.
    window_used : bool
        True si la ventana de este instante entro al analisis; False si se
        descarto por parpadeo, movimiento o falta de senal.
    display_spectrum_db : numpy.ndarray or None
        Espectro de la ventana completa, con artefactos incluidos, en dB y en
        las frecuencias de `spectrum_hz`. Solo para dibujar: lo que se ve no se
        recorta, y lo que no se usa se marca aparte.
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
    blink_count: int
    double_blink_count: int
    alpha_db: float | None
    fit_hz: np.ndarray | None
    fit_db: np.ndarray | None
    exponent: float | None
    window_used: bool
    display_spectrum_db: np.ndarray | None
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
        self._blink_count = 0
        self._double_count = 0
        self._double_cooldown_until = -(10**9)
        self._alpha_db: float | None = None
        self._last_alpha_at: float | None = None
        self._blink_prev_over = False
        self._polarity_votes: list[int] = []
        self._last_abs_cross = -(10**9)
        self._abs_prev_over = False
        self._candidate: int | None = None
        self._samples_seen = 0
        self._last_blink_sample = -(10**9)
        self._fit_hz: np.ndarray | None = None
        self._fit_log: np.ndarray | None = None
        self._exponent: float | None = None
        self._window_used = False
        self._display_psd: np.ndarray | None = None
        self._display_freqs: np.ndarray | None = None
        self._last_display_at: float | None = None

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
        blink_band = self.blink_band_block(samples_uv)
        self._append(self.filter_block(samples_uv), blink_band)
        self._count_blinks(blink_band, signal_ready)
        self._update_display_psd(now)
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
            blink_count=self._blink_count,
            double_blink_count=self._double_count,
            alpha_db=self._alpha_db if self._window_used else None,
            fit_hz=self._fit_hz,
            fit_db=None if self._fit_log is None else 10.0 * self._fit_log,
            exponent=self._exponent,
            window_used=self._window_used,
            display_spectrum_db=self._display_db(),
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

    def _count_blinks(self, blink_band: np.ndarray, ready: bool) -> None:
        """Cuenta parpadeos por su forma: lobulo y contralobulo.

        Un cruce del umbral hacia la polaridad del parpadeo abre un candidato;
        se confirma si en menos de `BLINK_CONFIRM_S` aparece el lobulo
        contrario. La polaridad se aprende del signo del primer cruce de cada
        evento, por mayoria.
        """
        n = blink_band.size
        if n == 0:
            return
        threshold = BLINK_COUNT_FACTOR * self._blink_threshold
        confirm_level = BLINK_CONFIRM_FACTOR * threshold
        confirm_n = int(BLINK_CONFIRM_S * self.fs)
        refractory = int(BLINK_REFRACTORY_S * self.fs)
        event_gap = int(BLINK_EVENT_GAP_S * self.fs)
        for offset, value in enumerate(blink_band):
            index = self._samples_seen + offset
            over_abs = abs(value) > threshold
            if over_abs and not self._abs_prev_over:
                if index - self._last_abs_cross > event_gap:
                    self._polarity_votes.append(1 if value > 0 else -1)
                    self._polarity_votes = self._polarity_votes[-BLINK_POLARITY_MEMORY:]
                self._last_abs_cross = index
            self._abs_prev_over = over_abs
            polarity = -1 if sum(self._polarity_votes) < 0 else 1
            signed = polarity * value
            over = signed > threshold
            if (
                over
                and not self._blink_prev_over
                and self._candidate is None
                and index - self._last_blink_sample > refractory
            ):
                self._candidate = index
            self._blink_prev_over = over
            if self._candidate is None:
                continue
            if index - self._candidate > confirm_n:
                self._candidate = None
            elif signed < -confirm_level:
                self._confirm_blink(self._candidate, ready)
                self._candidate = None
        self._samples_seen += n

    def _confirm_blink(self, index: int, ready: bool) -> None:
        if not ready:
            self._last_blink_sample = index
            return
        gap = index - self._last_blink_sample
        self._blink_count += 1
        if gap <= DOUBLE_BLINK_MAX_S * self.fs and index >= self._double_cooldown_until:
            self._double_count += 1
            self._double_cooldown_until = index + int(DOUBLE_BLINK_COOLDOWN_S * self.fs)
        self._last_blink_sample = index

    def _adaptive_blink_threshold(self) -> float:
        """BLINK_K desviaciones robustas del fondo, entre el piso y el techo.

        La desviacion sale del percentil 25 de |x|, que sigue describiendo el
        fondo aunque hasta tres cuartas partes del bufer sean parpadeos. Con
        menos de 1 s de datos se usa el piso.
        """
        return self._robust_threshold(self._blink_band)

    def _robust_threshold(self, band: np.ndarray) -> float:
        """BLINK_K desviaciones robustas de `band`, entre el piso y el techo."""
        if band.size < int(self.fs):
            return BLINK_FLOOR_UV
        deviation = np.abs(band - np.median(band))
        robust_sd = LOW_QUARTILE_TO_SD * float(np.percentile(deviation, 25))
        return float(np.clip(BLINK_K * robust_sd, BLINK_FLOOR_UV, BLINK_CEIL_UV))

    def _clean_mask(self) -> np.ndarray:
        """Muestras sin artefacto, con margen alrededor de cada excedencia."""
        if not self._over.any():
            return np.ones(self._filtered.size, dtype=bool)
        kernel = np.ones(2 * self._pad_n + 1)
        dilated = np.convolve(self._over.astype(float), kernel, mode="same") > 0
        return ~dilated

    def _evaluate(self, now: float, ready: bool) -> tuple[MetricStatus, Zone | None, float | None]:
        self._window_used = False
        if not ready:
            return MetricStatus.WAITING_SIGNAL, None, None
        if self._filtered.size < self._window_n:
            return MetricStatus.WAITING_SIGNAL, None, None

        residual = self._window_residual(now)
        if residual is None:
            return self._hold(now)
        f, res = residual
        self._window_used = True
        self._last_good_at = now
        self._update_alpha(periodic_band_power(f, res, self._baseline.iaf_hz), now)

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
        f, log_p, log_fit = aperiodic_fit(freqs, self._psd, exclude)
        self._fit_hz, self._fit_log = f, log_fit
        self._exponent = aperiodic_exponent(f, log_fit)
        return f, log_p - log_fit

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

    def _update_alpha(self, value: float, now: float) -> None:
        """Alfa suavizada igual que el indicador, pero sin depender de la linea base."""
        if self._alpha_db is None or self._last_alpha_at is None:
            self._alpha_db = value
        else:
            weight = 1.0 - np.exp(-max(0.0, now - self._last_alpha_at) / METRIC_TAU_S)
            self._alpha_db = (1.0 - weight) * self._alpha_db + weight * value
        self._last_alpha_at = now

    def _update_display_psd(self, now: float) -> None:
        """Espectro de la ventana completa, suavizado igual que el del analisis."""
        if self._filtered.size < self._window_n:
            return
        freqs, psd = spectrum(self._filtered[-self._window_n :], self.fs)
        if self._display_psd is None or self._last_display_at is None:
            self._display_psd = psd
        else:
            dt = max(0.0, now - self._last_display_at)
            weight = 1.0 - np.exp(-dt / SMOOTHING_TAU_S)
            self._display_psd = (1.0 - weight) * self._display_psd + weight * psd
        self._display_freqs = freqs
        self._last_display_at = now

    def _display_db(self) -> np.ndarray | None:
        if self._display_psd is None or self._display_freqs is None:
            return None
        lo, hi = SPECTRUM_DISPLAY_HZ
        band = (self._display_freqs >= lo) & (self._display_freqs <= hi)
        return 10.0 * np.log10(self._display_psd[band] + 1e-20)

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
            if self._display_freqs is None:
                return None, None
            lo, hi = SPECTRUM_DISPLAY_HZ
            band = (self._display_freqs >= lo) & (self._display_freqs <= hi)
            return self._display_freqs[band], None
        lo, hi = SPECTRUM_DISPLAY_HZ
        band = (self._freqs >= lo) & (self._freqs <= hi)
        return self._freqs[band], 10.0 * np.log10(self._psd[band] + 1e-20)
