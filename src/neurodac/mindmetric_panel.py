"""Panel de empresa del demo provisional de MindMetric, con datos simulados.

Lo que importa aqui no son los datos, que son inventados y se declaran como
tales en pantalla, sino las reglas que decide la arquitectura y que el demo
muestra en vivo:

1. Un grupo con menos de `MIN_GROUP_SIZE` personas no se muestra.
2. Un grupo con menos de `EXACT_RATE_MIN_SIZE` personas se muestra en rangos,
   porque un 0% o un 100% revelaria la conducta de cada integrante.
3. Los porcentajes se redondean a multiplos de `RATE_STEP_PCT`.
4. Los grupos se fijan al inicio del piloto; no hay filtros libres con los que
   se pueda restar un grupo de otro para aislar a alguien.

La empresa nunca ve senal cerebral ni resultados individuales: este modulo ni
siquiera los recibe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

__all__ = [
    "EXACT_RATE_MIN_SIZE",
    "GROUPS",
    "MIN_GROUP_SIZE",
    "Group",
    "GroupReport",
    "ReportKind",
    "SimulatedPilot",
    "report",
    "simulate_pilot",
]

MIN_GROUP_SIZE = 5
EXACT_RATE_MIN_SIZE = 10
RATE_STEP_PCT = 10
LOW_RANGE_MAX = 0.4
MID_RANGE_MAX = 0.6

PILOT_WEEKS = 12
CURRENT_WEEK = 7


@dataclass(frozen=True)
class Group:
    """Grupo predefinido del piloto. La etiqueta visible vive en el contenido."""

    key: str
    size: int


#: Piloto fundador de 20 personas. Los tamanos estan elegidos para que el
#: demo recorra las tres reglas: uno exacto, uno en rangos y uno oculto.
GROUPS: tuple[Group, ...] = (
    Group("todos", 20),
    Group("operaciones", 11),
    Group("administracion", 6),
    Group("direccion", 3),
)


class ReportKind(StrEnum):
    """Como se reporta un grupo."""

    SUPPRESSED = "oculto"
    RANGE = "rango"
    PERCENT = "porcentaje"


@dataclass(frozen=True)
class GroupReport:
    """Reporte de participacion de un grupo en una semana.

    Attributes
    ----------
    kind : ReportKind
        Regla aplicada.
    percent : int or None
        Porcentaje redondeado, solo si `kind` es PERCENT.
    range_key : str or None
        "menos", "mitad" o "mas", solo si `kind` es RANGE.
    """

    kind: ReportKind
    percent: int | None = None
    range_key: str | None = None


def report(size: int, participants: int) -> GroupReport:
    """Aplica las reglas de agregacion a un conteo.

    Parameters
    ----------
    size : int
        Personas del grupo.
    participants : int
        Cuantas participaron en la semana.
    """
    if size <= 0:
        raise ValueError(f"size debe ser > 0, no {size}")
    if not 0 <= participants <= size:
        raise ValueError(f"participants debe estar entre 0 y {size}, no {participants}")
    if size < MIN_GROUP_SIZE:
        return GroupReport(ReportKind.SUPPRESSED)
    rate = participants / size
    if size < EXACT_RATE_MIN_SIZE:
        if rate < LOW_RANGE_MAX:
            key = "menos"
        elif rate <= MID_RANGE_MAX:
            key = "mitad"
        else:
            key = "mas"
        return GroupReport(ReportKind.RANGE, range_key=key)
    # Redondeo hacia arriba en el medio: round() de Python iria al par, y 65%
    # quedaria en 60% pero 75% en 80%.
    percent = RATE_STEP_PCT * math.floor(100 * rate / RATE_STEP_PCT + 0.5)
    return GroupReport(ReportKind.PERCENT, percent=percent)


@dataclass(frozen=True)
class SimulatedPilot:
    """Participacion simulada por persona y semana.

    Attributes
    ----------
    membership : dict of str to numpy.ndarray
        Indices de las personas de cada grupo.
    participated : numpy.ndarray of bool
        Persona por semana: si hizo al menos una sesion.
    sessions : numpy.ndarray of int
        Sesiones por persona y semana.
    """

    membership: dict[str, np.ndarray]
    participated: np.ndarray
    sessions: np.ndarray

    def participants(self, group_key: str, week: int) -> int:
        """Personas del grupo que participaron en la semana (1 a `CURRENT_WEEK`)."""
        members = self.membership[group_key]
        return int(self.participated[members, week - 1].sum())

    def weekly_report(self, group_key: str) -> list[GroupReport]:
        """Reporte de cada semana transcurrida para un grupo."""
        size = self.membership[group_key].size
        return [
            report(size, self.participants(group_key, week)) for week in range(1, CURRENT_WEEK + 1)
        ]

    def median_sessions(self, week: int) -> float:
        """Mediana de sesiones entre quienes participaron en la semana."""
        column = self.sessions[:, week - 1]
        active = column[column > 0]
        return float(np.median(active)) if active.size else 0.0


def simulate_pilot(seed: int = 11) -> SimulatedPilot:
    """Genera un piloto plausible: la participacion empieza alta y decae.

    La caida es deliberada. Un demo con participacion perfecta seria menos
    creible que uno que muestra lo que suele pasar en los programas
    voluntarios.
    """
    rng = np.random.default_rng(seed)
    total = GROUPS[0].size
    subgroups = GROUPS[1:]
    if sum(g.size for g in subgroups) != total:
        raise ValueError("los subgrupos deben sumar el total del piloto")

    weeks = np.arange(1, CURRENT_WEEK + 1)
    propensity = rng.uniform(0.55, 1.0, size=total)
    decay = np.exp(-0.07 * (weeks - 1))
    probability = np.clip(propensity[:, None] * decay[None, :], 0.0, 1.0)
    participated = rng.random((total, CURRENT_WEEK)) < probability
    sessions = np.where(participated, rng.integers(1, 5, size=(total, CURRENT_WEEK)), 0)

    membership: dict[str, np.ndarray] = {GROUPS[0].key: np.arange(total)}
    start = 0
    for group in subgroups:
        membership[group.key] = np.arange(start, start + group.size)
        start += group.size
    return SimulatedPilot(membership, participated, sessions)
