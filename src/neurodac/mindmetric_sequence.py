"""Secuencia guiada de la demostracion "Respuesta a un estimulo".

En lugar de que el presentador consiga un estimulo y lo marque a mano, la
pantalla dirige la prueba: prepararse, cerrar los ojos, abrirlos, tres veces.
Cerrar los ojos es el estimulo mas confiable para la alfa (aumenta en
segundos), asi que la curva promedio muestra el efecto aunque la senal en la
frente sea modesta. Cada cierre se marca solo.

Puro: recibe los segundos transcurridos y devuelve la fase.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["CLOSED_S", "OPEN_S", "PREP_S", "ROUNDS", "Phase", "onsets", "phase_at", "total_s"]

PREP_S = 3.0
CLOSED_S = 6.0
OPEN_S = 6.0
ROUNDS = 3


@dataclass(frozen=True)
class Phase:
    """Fase en curso de la secuencia.

    Attributes
    ----------
    key : str
        "preparate", "cierra", "abre" o "listo".
    round : int
        Ronda en curso, de 1 a `ROUNDS` (0 durante la preparacion).
    remaining_s : float
        Segundos que faltan para la siguiente fase.
    """

    key: str
    round: int
    remaining_s: float


def total_s() -> float:
    """Duracion total de la secuencia."""
    return PREP_S + ROUNDS * (CLOSED_S + OPEN_S)


def onsets() -> list[float]:
    """Segundos, desde el inicio, en que empieza cada cierre de ojos."""
    return [PREP_S + k * (CLOSED_S + OPEN_S) for k in range(ROUNDS)]


def phase_at(elapsed: float) -> Phase:
    """Fase de la secuencia a los `elapsed` segundos de empezar."""
    if elapsed < PREP_S:
        return Phase("preparate", 0, PREP_S - elapsed)
    if elapsed >= total_s():
        return Phase("listo", ROUNDS, 0.0)
    into = elapsed - PREP_S
    period = CLOSED_S + OPEN_S
    index = int(into // period)
    within = into - index * period
    if within < CLOSED_S:
        return Phase("cierra", index + 1, CLOSED_S - within)
    return Phase("abre", index + 1, period - within)
