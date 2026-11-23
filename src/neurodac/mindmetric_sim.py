"""Diadema simulada para ensayar el demo provisional de MindMetric.

Hereda de `SimulatedSource`, asi que sigue emitiendo tramas ThinkGear reales
que pasan por el mismo parser y el mismo hilo lector que la diadema. Solo
cambia la muestra cruda: en lugar de senoidales fijas, produce EEG con escala
fisiologica en uV, ruido de baja frecuencia, un ritmo alfa que crece en los
tramos de "ojos cerrados" y parpadeos en los de ojos abiertos.

Guion fijo, pensado para ensayar: 50 s con ojos abiertos y despues, cada
40 s, 12 s con ojos cerrados. Los 50 s cubren los ~12 s que tarda la sesion en
quedar lista mas los 30 s de la linea base, si se pide en cuanto se puede.
"""

from __future__ import annotations

import math

import numpy as np

from .mindmetric_analysis import RAW_TO_UV
from .simulator import SimulatedSource

__all__ = ["MindMetricSimulatedSource", "eyes_closed_at"]

FIRST_CLOSED_S = 50.0
CYCLE_S = 40.0
CLOSED_S = 12.0
RAMP_S = 1.0

ALPHA_HZ = 10.0
ALPHA_OPEN_UV = 5.0
ALPHA_CLOSED_UV = 15.0
BETA_HZ = 18.0
BETA_UV = 2.0
LINE_HZ = 60.0
LINE_UV = 1.5
RED_NOISE_COEF = 0.995
RED_NOISE_SD_UV = 8.0
WHITE_NOISE_UV = 3.0
BLINK_UV = 180.0
BLINK_SIGMA_S = 0.06
BLINK_INTERVAL_S = (2.5, 5.5)

RAW_LIMIT = 2047


def eyes_closed_at(t: float) -> bool:
    """True si el guion tiene los ojos cerrados en el instante `t`."""
    return t >= FIRST_CLOSED_S and (t - FIRST_CLOSED_S) % CYCLE_S < CLOSED_S


def _closed_weight(t: float) -> float:
    """Peso de 0 a 1 con rampas suaves, para que la alfa no salte."""
    if t < FIRST_CLOSED_S:
        return 0.0
    phase = (t - FIRST_CLOSED_S) % CYCLE_S
    if phase < RAMP_S:
        return phase / RAMP_S
    if phase < CLOSED_S:
        return 1.0
    if phase < CLOSED_S + RAMP_S:
        return 1.0 - (phase - CLOSED_S) / RAMP_S
    return 0.0


class MindMetricSimulatedSource(SimulatedSource):
    """Fuente simulada con fisiologia plausible para el demo.

    Parameters
    ----------
    name : str, optional
        Clave en el registro de sesiones; distinta de la de NeuroDAC para que
        las dos paginas no compartan simulador.
    seed : int, optional
        Semilla: la misma semilla produce la misma corrida.
    """

    def __init__(self, name: str = "mindmetric-simulada", *, seed: int = 7) -> None:
        super().__init__(name, seed=seed)
        self._rng = np.random.default_rng(seed)
        self._red = 0.0
        self._red_drive = RED_NOISE_SD_UV * math.sqrt(1.0 - RED_NOISE_COEF**2)
        self._next_blink = float(self._rng.uniform(*BLINK_INTERVAL_S))
        self._blink_at: float | None = None

    def sample_uv(self, t: float) -> float:
        """Muestra en uV para el instante `t`; avanza el estado del ruido."""
        self._red = RED_NOISE_COEF * self._red + self._red_drive * self._rng.standard_normal()
        weight = _closed_weight(t)
        alpha_uv = ALPHA_OPEN_UV + (ALPHA_CLOSED_UV - ALPHA_OPEN_UV) * weight
        value = (
            alpha_uv * math.sin(2 * math.pi * ALPHA_HZ * t)
            + BETA_UV * math.sin(2 * math.pi * BETA_HZ * t + 1.1)
            + LINE_UV * math.sin(2 * math.pi * LINE_HZ * t)
            + self._red
            + WHITE_NOISE_UV * self._rng.standard_normal()
        )
        return value + self._blink(t, weight)

    def _blink(self, t: float, closed_weight: float) -> float:
        """Parpadeos solo con ojos abiertos, a intervalos irregulares."""
        if self._blink_at is None and t >= self._next_blink:
            if closed_weight == 0.0:
                self._blink_at = t + 3 * BLINK_SIGMA_S
            self._next_blink = t + float(self._rng.uniform(*BLINK_INTERVAL_S))
        if self._blink_at is None:
            return 0.0
        offset = t - self._blink_at
        if offset > 3 * BLINK_SIGMA_S:
            self._blink_at = None
            return 0.0
        return BLINK_UV * math.exp(-0.5 * (offset / BLINK_SIGMA_S) ** 2)

    def _raw_sample(self, t: float) -> int:
        counts = round(self.sample_uv(t) / RAW_TO_UV)
        return max(-RAW_LIMIT - 1, min(RAW_LIMIT, counts))
