"""Rank factorial (+ Baseline) OOS KPIs for one instance."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Change this to rank another instance's results.
INSTANCE = "4_20_5_20_3_1_CRP"

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / f"{INSTANCE}_objective_function_sensitivity_results.csv"
OUT = ROOT / "results" / f"{INSTANCE}_factorial_oos_kpi_ranking.csv"


def main() -> None:
    df = pd.read_csv(RESULTS)
    base = df[df.weight_scheme == "Baseline"].iloc[-1:]
    fac = df[df.sensitivity_mode == "factorial"].copy()
    allr = pd.concat([base, fac], ignore_index=True)

    cols = [
        "oos_pct_on_time_transfer",
        "oos_pct_on_time_evacuation",
        "oos_pct_not_evacuated",
    ]

    def norm_high(x: pd.Series) -> pd.Series:
        r = x.max() - x.min()
        return (x - x.min()) / r if r > 1e-12 else pd.Series(1.0, index=x.index)

    def norm_low(x: pd.Series) -> pd.Series:
        r = x.max() - x.min()
        return (x.max() - x) / r if r > 1e-12 else pd.Series(1.0, index=x.index)

    t = allr["oos_pct_on_time_transfer"].astype(float)
    e = allr["oos_pct_on_time_evacuation"].astype(float)
    n = allr["oos_pct_not_evacuated"].astype(float)
    allr["composite"] = (norm_high(t) + norm_high(e) + norm_low(n)) / 3.0

    lex = allr.sort_values(by=cols, ascending=[False, False, True]).reset_index(drop=True)
    lex["lex_rank"] = np.arange(1, len(lex) + 1)
    comp = allr.sort_values("composite", ascending=False).reset_index(drop=True)
    comp["comp_rank"] = np.arange(1, len(comp) + 1)

    m = lex[["weight_scheme"] + cols + ["composite", "lex_rank"]].merge(
        comp[["weight_scheme", "comp_rank"]], on="weight_scheme"
    )
    m["label"] = m["weight_scheme"].astype(str).str.replace(r"^FAC_", "", regex=True)
    m = m.sort_values("comp_rank")
    m.to_csv(OUT, index=False)
    print(m[["comp_rank", "lex_rank", "label"] + cols + ["composite"]].round(4).to_string(index=False))
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
