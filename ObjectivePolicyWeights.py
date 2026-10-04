"""
Policy preference weights for the second-stage composite objective.

These multiply the EXISTING instance/model coefficients. They are NOT a
normalization of objective components and do NOT change physical parameters
(travel times, risk factors, shortage costs, etc.).

Baseline = (1, 1, 1, 1) reproduces the current Gurobi formulation exactly.

OAT grids are component-aware because instance coefficient scales differ by
orders of magnitude (e.g. shortage ~150k vs mean travel time ~35).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union


COMPONENT_NAMES: Tuple[str, ...] = (
    "travel_time",
    "evacuation_risk",
    "unmet_demand",
    "threat_risk",
)


# Component-specific OAT grids sized to the instance coefficient scales:
#   travel ~35, evac/threat ~1e4, unmet ~1.5e5
# Shared 0.1–10 grids do not move travel; travel needs hundreds–thousands.
DEFAULT_OAT_MULTIPLIERS_BY_COMPONENT: Dict[str, Tuple[float, ...]] = {
    "travel_time": (1.0, 10.0, 20.0, 50.0, 100.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0),
    "evacuation_risk": (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
    "unmet_demand": (0.001, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
    "threat_risk": (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
}


@dataclass(frozen=True)
class ObjectivePolicyWeights:
    """Relative policy preference weights for the four second-stage criteria."""

    travel_time: float = 1.0
    evacuation_risk: float = 1.0
    unmet_demand: float = 1.0
    threat_risk: float = 1.0

    def __post_init__(self) -> None:
        for name in COMPONENT_NAMES:
            value = getattr(self, name)
            if not isinstance(value, (int, float)):
                raise TypeError(f"Weight '{name}' must be numeric, got {type(value)!r}")
            if value < 0:
                raise ValueError(f"Weight '{name}' must be non-negative, got {value}")

    @classmethod
    def baseline(cls) -> "ObjectivePolicyWeights":
        return cls()

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ObjectivePolicyWeights":
        kwargs = {name: float(mapping[name]) for name in COMPONENT_NAMES if name in mapping}
        missing = [name for name in COMPONENT_NAMES if name not in kwargs]
        if missing:
            raise ValueError(f"Missing weight keys: {missing}")
        return cls(**kwargs)

    def scaled(self, component: str, multiplier: float) -> "ObjectivePolicyWeights":
        if component not in COMPONENT_NAMES:
            raise ValueError(f"Unknown component '{component}'. Expected one of {COMPONENT_NAMES}")
        data = self.as_dict()
        data[component] = float(data[component]) * float(multiplier)
        return ObjectivePolicyWeights(**data)

    def as_dict(self) -> Dict[str, float]:
        return {f.name: float(getattr(self, f.name)) for f in fields(self)}

    def weight_shares(self) -> Dict[str, float]:
        """Normalized shares for reporting the policy mix only (not used in optimization)."""
        values = self.as_dict()
        total = sum(values.values())
        if total <= 0:
            return {name: 0.0 for name in COMPONENT_NAMES}
        return {name: values[name] / total for name in COMPONENT_NAMES}

    def label(self) -> str:
        return (
            f"tt={self.travel_time:g}_er={self.evacuation_risk:g}_"
            f"ud={self.unmet_demand:g}_tr={self.threat_risk:g}"
        )


def parse_objective_weights_string(text: str) -> ObjectivePolicyWeights:
    """
    Parse a CLI weight string.

    Accepted forms:
      travel_time=1,evacuation_risk=1,unmet_demand=1,threat_risk=1
      1,1,1,1
    """
    text = (text or "").strip()
    if not text:
        return ObjectivePolicyWeights.baseline()

    if "=" in text:
        parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]
        mapping: Dict[str, float] = {}
        for part in parts:
            if "=" not in part:
                raise ValueError(f"Invalid weight token '{part}' in '{text}'")
            key, value = part.split("=", 1)
            key = key.strip().lower()
            aliases = {
                "tt": "travel_time",
                "travel": "travel_time",
                "travel_time": "travel_time",
                "er": "evacuation_risk",
                "evac": "evacuation_risk",
                "evacuation_risk": "evacuation_risk",
                "ud": "unmet_demand",
                "unmet": "unmet_demand",
                "unmet_demand": "unmet_demand",
                "tr": "threat_risk",
                "threat": "threat_risk",
                "threat_risk": "threat_risk",
            }
            if key not in aliases:
                raise ValueError(f"Unknown weight key '{key}' in '{text}'")
            mapping[aliases[key]] = float(value)
        return ObjectivePolicyWeights.from_mapping(mapping)

    parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError(
            "ObjectiveWeights must have 4 values "
            "(travel_time, evacuation_risk, unmet_demand, threat_risk)"
        )
    return ObjectivePolicyWeights(
        travel_time=float(parts[0]),
        evacuation_risk=float(parts[1]),
        unmet_demand=float(parts[2]),
        threat_risk=float(parts[3]),
    )


def default_weight_schemes() -> Dict[str, ObjectivePolicyWeights]:
    """
    Primary journal-paper weighting schemes.

    Multipliers are relative to the existing instance coefficients (baseline = 1).
    Absolute coefficient magnitudes already embed physical units and the paper's
    calibrated preference levels; schemes therefore vary relative priority only.
    """
    return {
        "Baseline": ObjectivePolicyWeights.baseline(),
        "TravelTime_Priority": ObjectivePolicyWeights(
            travel_time=2.0, evacuation_risk=1.0, unmet_demand=1.0, threat_risk=1.0
        ),
        "EvacuationRisk_Priority": ObjectivePolicyWeights(
            travel_time=1.0, evacuation_risk=2.0, unmet_demand=1.0, threat_risk=1.0
        ),
        "UnmetDemand_Priority": ObjectivePolicyWeights(
            travel_time=1.0, evacuation_risk=1.0, unmet_demand=2.0, threat_risk=1.0
        ),
        "ThreatRisk_Priority": ObjectivePolicyWeights(
            travel_time=1.0, evacuation_risk=1.0, unmet_demand=1.0, threat_risk=2.0
        ),
    }


def oat_multipliers_default() -> Dict[str, Tuple[float, ...]]:
    """Component-aware default OAT grids (copy)."""
    return {k: tuple(v) for k, v in DEFAULT_OAT_MULTIPLIERS_BY_COMPONENT.items()}


def _normalize_component_key(key: str) -> str:
    aliases = {
        "tt": "travel_time",
        "travel": "travel_time",
        "travel_time": "travel_time",
        "er": "evacuation_risk",
        "evac": "evacuation_risk",
        "evacuation_risk": "evacuation_risk",
        "ud": "unmet_demand",
        "unmet": "unmet_demand",
        "unmet_demand": "unmet_demand",
        "tr": "threat_risk",
        "threat": "threat_risk",
        "threat_risk": "threat_risk",
    }
    k = key.strip().lower()
    if k not in aliases:
        raise ValueError(
            f"Unknown OAT component '{key}'. Expected one of {list(COMPONENT_NAMES)}"
        )
    return aliases[k]


def parse_oat_multipliers_string(
    text: Optional[str],
) -> Dict[str, Tuple[float, ...]]:
    """
    Parse OAT multiplier specification.

    Forms:
      None / "" / "auto" / "intelligent"
        -> component-aware defaults
      "0.5,1,2"
        -> same multipliers for every component (legacy)
      "travel_time=1,10,100;unmet_demand=0.05,0.1,1"
        -> per-component overrides (unlisted components keep defaults)
    """
    result = oat_multipliers_default()
    if text is None:
        return result
    raw = str(text).strip()
    if not raw or raw.lower() in {"auto", "intelligent", "default", "smart"}:
        return result

    # Per-component form uses '=' and ';' (or newlines) as component separators.
    if "=" in raw:
        chunks = [c.strip() for c in raw.replace("\n", ";").split(";") if c.strip()]
        for chunk in chunks:
            if "=" not in chunk:
                raise ValueError(
                    f"Invalid OAT token '{chunk}'. Use "
                    "'travel_time=1,10,100;unmet_demand=0.05,1' or a shared list '0.5,1,2'."
                )
            key, values = chunk.split("=", 1)
            component = _normalize_component_key(key)
            mults = tuple(float(x) for x in values.replace(" ", "").split(",") if x)
            if not mults:
                raise ValueError(f"No multipliers for component '{component}'")
            result[component] = mults
        return result

    # Shared list for all components
    shared = tuple(float(x) for x in raw.replace(";", ",").split(",") if x.strip())
    if not shared:
        return result
    return {name: shared for name in COMPONENT_NAMES}


def build_oat_schemes(
    baseline: Optional[ObjectivePolicyWeights] = None,
    multipliers: Optional[Union[Iterable[float], Mapping[str, Iterable[float]]]] = None,
) -> Dict[str, ObjectivePolicyWeights]:
    """
    Build one-at-a-time schemes.

    ``multipliers`` may be:
      - None -> intelligent per-component defaults
      - a single iterable applied to every component
      - a mapping component -> iterable of multipliers
    """
    baseline = baseline or ObjectivePolicyWeights.baseline()

    if multipliers is None:
        by_component = oat_multipliers_default()
    elif isinstance(multipliers, Mapping):
        by_component = oat_multipliers_default()
        for key, vals in multipliers.items():
            component = _normalize_component_key(str(key))
            by_component[component] = tuple(float(v) for v in vals)
    else:
        shared = tuple(float(v) for v in multipliers)
        by_component = {name: shared for name in COMPONENT_NAMES}

    schemes: Dict[str, ObjectivePolicyWeights] = {}
    for component in COMPONENT_NAMES:
        for mult in by_component[component]:
            name = f"OAT_{component}_x{mult:g}"
            schemes[name] = baseline.scaled(component, mult)
    return schemes


# ---------------------------------------------------------------------------
# Factorial / priority-combination design (scale-aware)
# ---------------------------------------------------------------------------
#
# OAT often leaves first-stage decisions unchanged because shortage (~1.5e5)
# and risk (~1e4) dwarf travel (~35). A full 4-way 3-level grid would be 81
# solves; we therefore treat evacuation_risk and threat_risk as one joint
# "risk" factor (same multiplier) → 3 × 3 × 3 = 27 combinations.
#
# Levels are chosen so corners can actually flip dominance:
#   TT_H * 35 ≈ 3.5e5   vs   UD_L * 1.5e5 ≈ 150     → travel can dominate
#   TT_L * 35 ≈ 350     vs   UD_H * 1.5e5 ≈ 7.5e5   → shortage can dominate
#   RK_M * 1e4 ≈ 2e4    sits between travel-M and shortage-M

FACTORIAL_LEVEL_LABELS: Tuple[str, ...] = ("L", "M", "H")

DEFAULT_FACTORIAL_LEVELS: Dict[str, Dict[str, float]] = {
    # Travel must use large multipliers to compete with shortage.
    "travel_time": {"L": 10.0, "M": 500.0, "H": 10000.0},
    # Joint multiplier for evacuation_risk AND threat_risk.
    "risk": {"L": 0.1, "M": 2.0, "H": 10.0},
    # Shortage / unmet demand.
    "unmet_demand": {"L": 0.001, "M": 0.25, "H": 5.0},
}


def factorial_levels_default() -> Dict[str, Dict[str, float]]:
    return {
        factor: dict(levels) for factor, levels in DEFAULT_FACTORIAL_LEVELS.items()
    }


def build_factorial_schemes(
    levels: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> Dict[str, ObjectivePolicyWeights]:
    """
    Full factorial of travel × joint-risk × unmet priority levels.

    Scheme names look like: FAC_TT-H_RK-M_UD-L
    """
    from itertools import product

    lvl = factorial_levels_default()
    if levels is not None:
        for key, mapping in levels.items():
            k = str(key).strip().lower()
            if k in ("travel", "tt", "travel_time"):
                lvl["travel_time"] = {str(a).upper(): float(b) for a, b in mapping.items()}
            elif k in ("risk", "rk"):
                lvl["risk"] = {str(a).upper(): float(b) for a, b in mapping.items()}
            elif k in ("unmet", "ud", "unmet_demand", "shortage"):
                lvl["unmet_demand"] = {str(a).upper(): float(b) for a, b in mapping.items()}
            else:
                raise ValueError(f"Unknown factorial factor '{key}'")

    tt_labels = list(lvl["travel_time"].keys())
    rk_labels = list(lvl["risk"].keys())
    ud_labels = list(lvl["unmet_demand"].keys())

    schemes: Dict[str, ObjectivePolicyWeights] = {}
    for tt, rk, ud in product(tt_labels, rk_labels, ud_labels):
        name = f"FAC_TT-{tt}_RK-{rk}_UD-{ud}"
        risk_w = float(lvl["risk"][rk])
        schemes[name] = ObjectivePolicyWeights(
            travel_time=float(lvl["travel_time"][tt]),
            evacuation_risk=risk_w,
            unmet_demand=float(lvl["unmet_demand"][ud]),
            threat_risk=risk_w,
        )
    return schemes


def parse_factorial_levels_string(text: Optional[str]) -> Dict[str, Dict[str, float]]:
    """
    Optional override, e.g.:
      travel_time=L:10,M:500,H:10000;risk=L:0.1,M:2,H:10;unmet_demand=L:0.001,M:0.25,H:5
    """
    result = factorial_levels_default()
    if text is None or not str(text).strip() or str(text).strip().lower() in {"auto", "default"}:
        return result
    raw = str(text).strip()
    for chunk in raw.replace("\n", ";").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(
                "FactorialLevels must look like "
                "'travel_time=L:10,M:500,H:10000;risk=L:0.1,M:2,H:10;unmet_demand=L:0.001,M:0.25,H:5'"
            )
        key, values = chunk.split("=", 1)
        k = key.strip().lower()
        mapping: Dict[str, float] = {}
        for token in values.split(","):
            token = token.strip()
            if not token:
                continue
            if ":" not in token:
                raise ValueError(f"Invalid level token '{token}' (expected L:10)")
            lab, val = token.split(":", 1)
            mapping[lab.strip().upper()] = float(val)
        if k in ("travel", "tt", "travel_time"):
            result["travel_time"] = mapping
        elif k in ("risk", "rk"):
            result["risk"] = mapping
        elif k in ("unmet", "ud", "unmet_demand", "shortage"):
            result["unmet_demand"] = mapping
        else:
            raise ValueError(f"Unknown factorial factor '{key}'")
    return result
