"""Plotting utilities for objective-function sensitivity analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def _prefixed_filename(filename: str, instance_name: Optional[str]) -> str:
    """Ensure output artifact names start with the instance name."""
    name = Path(filename).name
    if not instance_name:
        return name
    prefix = f"{instance_name}_"
    if name.startswith(prefix):
        return name
    return f"{prefix}{name}"


def _out(plot_dir: Path, filename: str, instance_name: Optional[str]) -> Path:
    return Path(plot_dir) / _prefixed_filename(filename, instance_name)


def _savefig(fig, path: Path, dpi: int = 300) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def generate_all_plots(df: pd.DataFrame, plot_dir: Path, instance_name: Optional[str] = None) -> None:
    if df is None or df.empty:
        return
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    success = df[df["success"] == True].copy() if "success" in df.columns else df.copy()
    if success.empty:
        return

    scheme_df = success[success["sensitivity_mode"].isin(["schemes", "baseline", "custom"])].copy()
    if scheme_df.empty:
        scheme_df = success.copy()

    # Figure 1 — Objective by scheme
    if "weight_scheme" in scheme_df.columns and "total_objective" in scheme_df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        ordered = scheme_df.sort_values("total_objective")
        ax.bar(ordered["weight_scheme"], ordered["total_objective"])
        ax.set_xlabel("Weighting scheme")
        ax.set_ylabel("Total objective value")
        ax.set_title(
            "Objective value by policy scheme\n"
            "(values are under different preference structures; not directly comparable welfare scores)"
        )
        ax.tick_params(axis="x", rotation=30)
        _savefig(fig, _out(plot_dir, "objective_by_weight_scheme.png", instance_name))

    # Figure 2 — Raw components by scheme
    comp_cols = [
        "travel_time_raw",
        "evacuation_risk_raw",
        "unmet_demand_raw",
        "threat_risk_raw",
    ]
    if all(c in scheme_df.columns for c in comp_cols):
        fig, ax = plt.subplots(figsize=(11, 5))
        x = range(len(scheme_df))
        width = 0.2
        for i, col in enumerate(comp_cols):
            ax.bar([xi + i * width for xi in x], scheme_df[col], width=width, label=col.replace("_raw", ""))
        ax.set_xticks([xi + 1.5 * width for xi in x])
        ax.set_xticklabels(scheme_df["weight_scheme"], rotation=30, ha="right")
        ax.set_ylabel("Raw component contribution (baseline coefficients)")
        ax.set_title("Objective component trade-offs across weighting schemes")
        ax.legend()
        _savefig(fig, _out(plot_dir, "objective_components_by_scheme.png", instance_name))

    # Prefer out-of-sample KPIs for paper-facing plots when available
    on_time_col = "oos_pct_on_time_transfer" if "oos_pct_on_time_transfer" in scheme_df.columns else "on_time_transfer_percentage"
    on_time_evac_col = "oos_pct_on_time_evacuation" if "oos_pct_on_time_evacuation" in scheme_df.columns else "on_time_evacuation_percentage"
    nonevac_col = "oos_pct_not_evacuated" if "oos_pct_not_evacuated" in success.columns else "non_evacuated_patients"

    # Figure 3 — Unmet / not-evacuated (prefer OOS %)
    if {"unmet_demand", "weight_scheme"}.issubset(success.columns) and nonevac_col in success.columns:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(success["unmet_demand"], success[nonevac_col])
        for _, r in success.iterrows():
            ax.annotate(str(r["weight_scheme"]), (r["unmet_demand"], r[nonevac_col]), fontsize=7)
        ax.set_xlabel("In-sample unmet casualty demand (total over scenarios)")
        ax.set_ylabel(nonevac_col)
        ax.set_title("Unmet demand vs unevacuated patients")
        _savefig(fig, _out(plot_dir, "unmet_demand_vs_threat_risk.png", instance_name))

    # Figure 4 — Transport time vs evacuation risk
    if {"total_transport_time", "evacuation_risk_raw", "weight_scheme"}.issubset(success.columns):
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(success["total_transport_time"], success["evacuation_risk_raw"])
        for _, r in success.iterrows():
            ax.annotate(str(r["weight_scheme"]), (r["total_transport_time"], r["evacuation_risk_raw"]), fontsize=7)
        ax.set_xlabel("Total transport time (flow-weighted)")
        ax.set_ylabel("Evacuation-risk component (raw)")
        ax.set_title("Transportation time vs evacuation risk")
        _savefig(fig, _out(plot_dir, "transport_time_vs_evacuation_risk.png", instance_name))

    # Figure 5 — On-time performance (prefer OOS)
    if on_time_col in scheme_df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(scheme_df["weight_scheme"], scheme_df[on_time_col], label=on_time_col)
        if on_time_evac_col in scheme_df.columns:
            ax.plot(
                scheme_df["weight_scheme"],
                scheme_df[on_time_evac_col],
                marker="o",
                color="C1",
                label=on_time_evac_col,
            )
        ax.set_ylabel("Percentage")
        ax.set_title("On-time performance across schemes (prefer out-of-sample KPIs)")
        ax.tick_params(axis="x", rotation=30)
        ax.legend()
        _savefig(fig, _out(plot_dir, "on_time_vs_weight.png", instance_name))

    # Figure 6 — Threat-risk OAT
    oat_threat = success[success["weight_scheme"].astype(str).str.startswith("OAT_threat_risk")].copy()
    if not oat_threat.empty and "weight_threat_risk" in oat_threat.columns:
        oat_threat = oat_threat.sort_values("weight_threat_risk")
        y_nonevac = "oos_pct_not_evacuated" if "oos_pct_not_evacuated" in oat_threat.columns else "non_evacuated_patients"
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(oat_threat["weight_threat_risk"], oat_threat[y_nonevac], marker="o", label=y_nonevac)
        ax1.set_xlabel("Threat-risk policy weight")
        ax1.set_ylabel(y_nonevac)
        ax2 = ax1.twinx()
        ax2.plot(oat_threat["weight_threat_risk"], oat_threat["threat_risk_raw"], marker="s", color="C1", label="Threat risk raw")
        ax2.set_ylabel("Threat-risk component (raw)")
        ax1.set_title("Threat-risk weight sensitivity")
        _savefig(fig, _out(plot_dir, "non_evacuated_vs_weight.png", instance_name))

    # Figure 7 — Unmet-demand OAT
    oat_unmet = success[success["weight_scheme"].astype(str).str.startswith("OAT_unmet_demand")].copy()
    if not oat_unmet.empty and "weight_unmet_demand" in oat_unmet.columns:
        oat_unmet = oat_unmet.sort_values("weight_unmet_demand")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(oat_unmet["weight_unmet_demand"], oat_unmet["unmet_demand"], marker="o", label="Unmet demand")
        ax.plot(
            oat_unmet["weight_unmet_demand"],
            oat_unmet["non_evacuated_patients"],
            marker="s",
            label="Non-evacuated patients",
        )
        ax.set_xlabel("Unmet-demand policy weight")
        ax.set_ylabel("Count (total over scenarios)")
        ax.set_title("Unmet-demand weight sensitivity")
        ax.legend()
        _savefig(fig, _out(plot_dir, "unmet_demand_sensitivity.png", instance_name))

    # Figure 8 — Priority-class effects
    if {"high_priority_unmet", "low_priority_unmet"}.issubset(scheme_df.columns):
        fig, ax = plt.subplots(figsize=(10, 5))
        x = range(len(scheme_df))
        ax.bar([xi - 0.2 for xi in x], scheme_df["high_priority_unmet"], width=0.4, label="High priority unmet")
        ax.bar([xi + 0.2 for xi in x], scheme_df["low_priority_unmet"], width=0.4, label="Low priority unmet")
        ax.set_xticks(list(x))
        ax.set_xticklabels(scheme_df["weight_scheme"], rotation=30, ha="right")
        ax.set_ylabel("Unmet casualties (total)")
        ax.set_title("Priority-class unmet demand by scheme")
        ax.legend()
        _savefig(fig, _out(plot_dir, "priority_class_effects.png", instance_name))

    # Figure 9 — Operational decision changes
    if "number_of_changed_binary_decisions" in scheme_df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(scheme_df["weight_scheme"], scheme_df["number_of_changed_binary_decisions"])
        ax.set_ylabel("Changed binary decisions vs baseline")
        ax.set_title("Operational decision changes relative to baseline")
        ax.tick_params(axis="x", rotation=30)
        _savefig(fig, _out(plot_dir, "decision_changes_by_scheme.png", instance_name))

    # Empirical trade-off cloud (all successful runs)
    if {"unmet_demand_raw", "threat_risk_raw"}.issubset(success.columns):
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(success["unmet_demand_raw"], success["threat_risk_raw"])
        ax.set_xlabel("Unmet-demand component (raw)")
        ax.set_ylabel("Threat-risk component (raw)")
        ax.set_title(
            "Empirical trade-offs under weighted scalarizations\n"
            "(not a formally generated Pareto frontier)"
        )
        _savefig(fig, _out(plot_dir, "empirical_tradeoff_unmet_vs_threat.png", instance_name))

    # Figure 10 — Travel-time OAT: ACF proximity to demand
    oat_travel = success[success["weight_scheme"].astype(str).str.startswith("OAT_travel_time")].copy()
    if oat_travel.empty:
        # Include baseline + travel OAT if present under schemes naming
        oat_travel = success[
            success["weight_scheme"].astype(str).isin(["Baseline"])
            | success["weight_scheme"].astype(str).str.startswith("OAT_travel_time")
        ].copy()
    if (
        not oat_travel.empty
        and "weight_travel_time" in oat_travel.columns
        and "mean_nearest_open_acf_distance_to_demand" in oat_travel.columns
    ):
        oat_travel = oat_travel.sort_values("weight_travel_time")
        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax1.plot(
            oat_travel["weight_travel_time"],
            oat_travel["mean_nearest_open_acf_distance_to_demand"],
            marker="o",
            label="Mean nearest open-ACF distance to demand",
        )
        if "mean_pairwise_open_acf_distance" in oat_travel.columns:
            ax1.plot(
                oat_travel["weight_travel_time"],
                oat_travel["mean_pairwise_open_acf_distance"],
                marker="s",
                label="Mean pairwise distance among open ACFs",
            )
        ax1.set_xscale("log")
        ax1.set_xlabel("Travel-time policy weight (log scale)")
        ax1.set_ylabel("Distance (instance plane units)")
        ax1.set_title(
            "Established ACF geometry vs travel-time weight\n"
            "(Euclidean distances from instance ACF/demand positions)"
        )
        ax1.legend(loc="best")
        _savefig(fig, _out(plot_dir, "acf_distance_vs_travel_weight.png", instance_name))

        # Map of open-ACF centroids
        if {"open_acf_centroid_x", "open_acf_centroid_y"}.issubset(oat_travel.columns):
            fig, ax = plt.subplots(figsize=(7, 6))
            sc = ax.scatter(
                oat_travel["open_acf_centroid_x"],
                oat_travel["open_acf_centroid_y"],
                c=oat_travel["weight_travel_time"],
                cmap="viridis",
                s=60,
            )
            for _, r in oat_travel.iterrows():
                if pd.notna(r.get("open_acf_centroid_x")):
                    ax.annotate(
                        f"w={r['weight_travel_time']:g}",
                        (r["open_acf_centroid_x"], r["open_acf_centroid_y"]),
                        fontsize=7,
                    )
            fig.colorbar(sc, ax=ax, label="Travel-time weight")
            ax.set_xlabel("Open-ACF centroid X")
            ax.set_ylabel("Open-ACF centroid Y")
            ax.set_title("Centroid of established ACFs as travel-time weight changes")
            _savefig(fig, _out(plot_dir, "acf_centroid_vs_travel_weight.png", instance_name))

    # Figure 11 — Travel-time OAT: transport time
    if not oat_travel.empty and "total_transport_time" in oat_travel.columns:
        oat_travel = oat_travel.sort_values("weight_travel_time")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(oat_travel["weight_travel_time"], oat_travel["total_transport_time"], marker="o")
        ax.set_xscale("log")
        ax.set_xlabel("Travel-time policy weight (log scale)")
        ax.set_ylabel("Total transport time (flow-weighted)")
        ax.set_title("Transport time vs travel-time policy weight")
        _savefig(fig, _out(plot_dir, "transport_time_vs_travel_weight.png", instance_name))


def _parse_fac_levels(scheme: str) -> Optional[dict]:
    """Parse FAC_TT-H_RK-M_UD-L -> {travel: H, risk: M, unmet: L}."""
    if not isinstance(scheme, str) or not scheme.startswith("FAC_"):
        return None
    out = {}
    for token in scheme.split("_")[1:]:
        if token.startswith("TT-"):
            out["travel"] = token.split("-", 1)[1]
        elif token.startswith("RK-"):
            out["risk"] = token.split("-", 1)[1]
        elif token.startswith("UD-"):
            out["unmet"] = token.split("-", 1)[1]
    return out if len(out) == 3 else None


def generate_factorial_plots(df: pd.DataFrame, plot_dir: Path, instance_name: Optional[str] = None) -> None:
    """
    Plots specific to the factorial priority-combination section.
    Written under plots/factorial/ so they do not overwrite OAT/scheme figures.
    """
    if df is None or df.empty:
        return
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    success = df[df["success"] == True].copy() if "success" in df.columns else df.copy()
    if "sensitivity_mode" in success.columns:
        fac = success[success["sensitivity_mode"] == "factorial"].copy()
    else:
        fac = success[success["weight_scheme"].astype(str).str.startswith("FAC_")].copy()
    if fac.empty:
        return

    parsed = fac["weight_scheme"].map(_parse_fac_levels)
    fac = fac.assign(
        travel_level=parsed.map(lambda d: None if d is None else d.get("travel")),
        risk_level=parsed.map(lambda d: None if d is None else d.get("risk")),
        unmet_level=parsed.map(lambda d: None if d is None else d.get("unmet")),
    )

    level_order = ["L", "M", "H"]

    # 1) Number of ACFs / vehicles / backups by scheme
    count_cols = [
        ("number_of_active_acfs", "Active ACFs"),
        ("total_land_vehicles_assigned", "Vehicles Assigned"),
        ("number_of_backup_hospital_links", "Backup Hospital Links"),
    ]
    present_counts = [(c, lab) for c, lab in count_cols if c in fac.columns]
    if present_counts:
        fig, axes = plt.subplots(1, len(present_counts), figsize=(5 * len(present_counts), 4), sharex=False)
        if len(present_counts) == 1:
            axes = [axes]
        ordered = fac.sort_values("weight_scheme")
        x_labels = (
            ordered["weight_scheme"]
            .astype(str)
            .str.replace(r"^FAC_", "", regex=True)
        )
        x = range(len(ordered))
        for ax, (col, lab) in zip(axes, present_counts):
            ax.bar(x, ordered[col])
            ax.set_xticks(list(x))
            ax.set_xticklabels(x_labels, rotation=90, fontsize=6)
            ax.set_ylabel(lab)
        _savefig(fig, _out(plot_dir, "factorial_first_stage_counts.png", instance_name))

    # 2) Heatmap: #ACFs vs travel×unmet, one panel per risk level
    if {"number_of_active_acfs", "travel_level", "unmet_level", "risk_level"}.issubset(fac.columns):
        risk_levels = [r for r in level_order if r in set(fac["risk_level"].dropna())]
        if risk_levels:
            fig, axes = plt.subplots(1, len(risk_levels), figsize=(4.5 * len(risk_levels), 4), sharey=True)
            if len(risk_levels) == 1:
                axes = [axes]
            for ax, rk in zip(axes, risk_levels):
                sub = fac[fac["risk_level"] == rk]
                pivot = sub.pivot_table(
                    index="unmet_level",
                    columns="travel_level",
                    values="number_of_active_acfs",
                    aggfunc="mean",
                )
                pivot = pivot.reindex(index=[i for i in level_order if i in pivot.index],
                                     columns=[c for c in level_order if c in pivot.columns])
                im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
                ax.set_xticks(range(len(pivot.columns)))
                ax.set_xticklabels(pivot.columns)
                ax.set_yticks(range(len(pivot.index)))
                ax.set_yticklabels(pivot.index)
                ax.set_xlabel("Travel Time Priority")
                ax.set_ylabel("Unmet Demand Priority")
                ax.set_title(f"Evacuation Risk Priority={rk}")
                for i in range(pivot.shape[0]):
                    for j in range(pivot.shape[1]):
                        val = pivot.values[i, j]
                        if pd.notna(val):
                            ax.text(j, i, f"{val:.0f}", ha="center", va="center", color="w", fontsize=8)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.suptitle("Number of Established ACFs")
            _savefig(fig, _out(plot_dir, "factorial_acf_count_heatmap.png", instance_name))

    # 3) Distance metrics vs travel weight, colored by unmet level
    # Distinct markers + log-x dodge so overlapping UD series stay visible.
    ud_style = {
        "L": {"color": "C1", "marker": "o", "dodge": 0.85, "zorder": 3},
        "M": {"color": "C2", "marker": "s", "dodge": 1.00, "zorder": 2},
        "H": {"color": "C0", "marker": "^", "dodge": 1.18, "zorder": 4},
    }
    if {"weight_travel_time", "mean_nearest_open_acf_distance_to_demand"}.issubset(fac.columns):
        fig, ax = plt.subplots(figsize=(8, 5))
        for ud in ("L", "M", "H"):
            sub = fac[fac["unmet_level"] == ud]
            if sub.empty:
                continue
            style = ud_style[ud]
            sub = sub.sort_values("weight_travel_time")
            ax.scatter(
                sub["weight_travel_time"] * style["dodge"],
                sub["mean_nearest_open_acf_distance_to_demand"],
                label=f"UD={ud}",
                s=90,
                color=style["color"],
                marker=style["marker"],
                edgecolors="black",
                linewidths=0.7,
                alpha=0.95,
                zorder=style["zorder"],
            )
        ax.set_xscale("log")
        ax.set_xlabel("Travel Time Weight (log)")
        ax.set_ylabel("Mean Nearest open-ACF Distance to Demand Points")
        ax.legend()
        _savefig(fig, _out(plot_dir, "factorial_acf_demand_distance.png", instance_name))

    if {"weight_travel_time", "mean_pairwise_open_acf_distance"}.issubset(fac.columns):
        fig, ax = plt.subplots(figsize=(8, 5))
        for ud in ("L", "M", "H"):
            sub = fac[fac["unmet_level"] == ud]
            if sub.empty:
                continue
            style = ud_style[ud]
            sub = sub.sort_values("weight_travel_time")
            ax.scatter(
                sub["weight_travel_time"] * style["dodge"],
                sub["mean_pairwise_open_acf_distance"],
                label=f"UD={ud}",
                s=90,
                color=style["color"],
                marker=style["marker"],
                edgecolors="black",
                linewidths=0.7,
                alpha=0.95,
                zorder=style["zorder"],
            )
        ax.set_xscale("log")
        ax.set_xlabel("Travel Time Weight (log)")
        ax.set_ylabel("Mean Pairwise Distance among Open ACFs")
        ax.legend()
        _savefig(fig, _out(plot_dir, "factorial_acf_pairwise_distance.png", instance_name))

    # 4) Unique ACF location sets (how often decisions change)
    if "acf_locations" in fac.columns:
        n_unique = fac["acf_locations"].nunique(dropna=True)
        counts = fac["acf_locations"].value_counts()
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(range(len(counts)), counts.values)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index.astype(str), rotation=90, fontsize=6)
        ax.set_ylabel("Number of factorial schemes")
        ax.set_title(f"Distinct established-ACF sets ({n_unique} unique patterns)")
        _savefig(fig, _out(plot_dir, "factorial_unique_acf_patterns.png", instance_name))

    # 5) OOS operational KPIs
    oos_cols = [
        ("oos_pct_on_time_transfer", "% On-Time Transfer"),
        ("oos_pct_on_time_evacuation", "% On-Time Evacuation"),
        ("oos_pct_not_evacuated", "% Not Evacuated"),
    ]
    present_oos = [(c, lab) for c, lab in oos_cols if c in fac.columns]
    if present_oos:
        fig, axes = plt.subplots(1, len(present_oos), figsize=(5 * len(present_oos), 4))
        if len(present_oos) == 1:
            axes = [axes]
        ordered = fac.sort_values("weight_scheme")
        x_labels = (
            ordered["weight_scheme"]
            .astype(str)
            .str.replace(r"^FAC_", "", regex=True)
        )
        x = range(len(ordered))
        for ax, (col, lab) in zip(axes, present_oos):
            ax.bar(x, ordered[col])
            ax.set_xticks(list(x))
            ax.set_xticklabels(x_labels, rotation=90, fontsize=6)
            ax.set_ylabel(lab)
        _savefig(fig, _out(plot_dir, "factorial_oos_kpis.png", instance_name))


def _extract_baseline_row(df: pd.DataFrame) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    success = df[df["success"] == True].copy() if "success" in df.columns else df.copy()
    if "weight_scheme" in success.columns:
        hits = success[success["weight_scheme"].astype(str) == "Baseline"]
        if not hits.empty:
            return hits.iloc[-1]
    if "sensitivity_mode" in success.columns:
        hits = success[success["sensitivity_mode"].astype(str) == "baseline"]
        if not hits.empty:
            return hits.iloc[-1]
    return None


def generate_factorial_baseline_comparison_plots(df: pd.DataFrame, plot_dir: Path, instance_name: Optional[str] = None) -> None:
    """
    Factorial plots with Baseline overlaid for direct comparison.
    Default output: plots/factorial+Baseline/
    """
    if df is None or df.empty:
        return
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    baseline = _extract_baseline_row(df)
    if baseline is None:
        raise ValueError("No Baseline row found in results CSV; cannot build factorial+Baseline plots.")

    success = df[df["success"] == True].copy() if "success" in df.columns else df.copy()
    if "sensitivity_mode" in success.columns:
        fac = success[success["sensitivity_mode"] == "factorial"].copy()
    else:
        fac = success[success["weight_scheme"].astype(str).str.startswith("FAC_")].copy()
    if fac.empty:
        return

    parsed = fac["weight_scheme"].map(_parse_fac_levels)
    fac = fac.assign(
        travel_level=parsed.map(lambda d: None if d is None else d.get("travel")),
        risk_level=parsed.map(lambda d: None if d is None else d.get("risk")),
        unmet_level=parsed.map(lambda d: None if d is None else d.get("unmet")),
    )
    level_order = ["L", "M", "H"]
    base_n_acf = float(baseline.get("number_of_active_acfs")) if pd.notna(baseline.get("number_of_active_acfs")) else None
    base_locs = str(baseline.get("acf_locations") or "")
    base_color = "#C0392B"

    # 1) First-stage counts with Baseline bar first
    count_cols = [
        ("number_of_active_acfs", "Active ACFs"),
        ("total_land_vehicles_assigned", "Vehicles Assigned"),
        ("number_of_backup_hospital_links", "Backup Hospital Links"),
    ]
    present_counts = [(c, lab) for c, lab in count_cols if c in fac.columns and c in baseline.index]
    if present_counts:
        fig, axes = plt.subplots(1, len(present_counts), figsize=(5.5 * len(present_counts), 4.2), sharex=False)
        if len(present_counts) == 1:
            axes = [axes]
        ordered = fac.sort_values("weight_scheme")
        x_labels = ["Baseline"] + (
            ordered["weight_scheme"].astype(str).str.replace(r"^FAC_", "", regex=True).tolist()
        )
        x = range(len(x_labels))
        for ax, (col, lab) in zip(axes, present_counts):
            values = [float(baseline[col])] + ordered[col].astype(float).tolist()
            colors = [base_color] + ["#4C72B0"] * len(ordered)
            ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.4)
            ax.set_xticks(list(x))
            ax.set_xticklabels(x_labels, rotation=90, fontsize=6)
            ax.set_ylabel(lab)
            ax.axhline(float(baseline[col]), color=base_color, linestyle="--", linewidth=1.0, alpha=0.7)
        _savefig(fig, _out(plot_dir, "factorial_first_stage_counts.png", instance_name))

    # 2) Heatmap with Baseline ACF-count / same-set markers
    if {"number_of_active_acfs", "travel_level", "unmet_level", "risk_level"}.issubset(fac.columns):
        risk_levels = [r for r in level_order if r in set(fac["risk_level"].dropna())]
        if risk_levels:
            fig, axes = plt.subplots(1, len(risk_levels), figsize=(4.8 * len(risk_levels), 4.4), sharey=True)
            if len(risk_levels) == 1:
                axes = [axes]
            for ax, rk in zip(axes, risk_levels):
                sub = fac[fac["risk_level"] == rk]
                pivot = sub.pivot_table(
                    index="unmet_level",
                    columns="travel_level",
                    values="number_of_active_acfs",
                    aggfunc="mean",
                )
                pivot = pivot.reindex(
                    index=[i for i in level_order if i in pivot.index],
                    columns=[c for c in level_order if c in pivot.columns],
                )
                im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
                ax.set_xticks(range(len(pivot.columns)))
                ax.set_xticklabels(pivot.columns)
                ax.set_yticks(range(len(pivot.index)))
                ax.set_yticklabels(pivot.index)
                ax.set_xlabel("Travel Time Priority")
                ax.set_ylabel("Unmet Demand Priority")
                ax.set_title(f"Evacuation Risk Priority={rk}")
                for i, ud in enumerate(pivot.index):
                    for j, tt in enumerate(pivot.columns):
                        val = pivot.values[i, j]
                        if pd.isna(val):
                            continue
                        cell = sub[(sub["unmet_level"] == ud) & (sub["travel_level"] == tt)]
                        same_set = (
                            (not cell.empty)
                            and base_locs
                            and str(cell.iloc[0].get("acf_locations")) == base_locs
                        )
                        same_count = base_n_acf is not None and abs(float(val) - base_n_acf) < 0.5
                        label = f"{val:.0f}"
                        if same_set:
                            label = f"{val:.0f}\n★Base"
                        elif same_count:
                            label = f"{val:.0f}\n(=Base #)"
                        ax.text(j, i, label, ha="center", va="center", color="w", fontsize=7, fontweight="bold")
                        if same_set:
                            ax.add_patch(
                                plt.Rectangle(
                                    (j - 0.48, i - 0.48),
                                    0.96,
                                    0.96,
                                    fill=False,
                                    edgecolor="gold",
                                    linewidth=2.5,
                                )
                            )
                        elif same_count:
                            ax.add_patch(
                                plt.Rectangle(
                                    (j - 0.48, i - 0.48),
                                    0.96,
                                    0.96,
                                    fill=False,
                                    edgecolor="white",
                                    linewidth=1.5,
                                    linestyle="--",
                                )
                            )
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.suptitle(
                f"Number of Established ACFs  |  Baseline = {base_n_acf:.0f} ACFs"
                + (f" at {{{base_locs}}}" if base_locs else "")
            )
            fig.text(
                0.5,
                0.02,
                "★Base / gold box = same open-ACF set as Baseline;  (=Base #) / dashed = same ACF count only",
                ha="center",
                fontsize=8,
            )
            fig.subplots_adjust(bottom=0.14)
            _savefig(fig, _out(plot_dir, "factorial_acf_count_heatmap.png", instance_name))

    # 3) Distance scatters with Baseline marker
    ud_style = {
        "L": {"color": "C1", "marker": "o", "dodge": 0.85, "zorder": 3},
        "M": {"color": "C2", "marker": "s", "dodge": 1.00, "zorder": 2},
        "H": {"color": "C0", "marker": "^", "dodge": 1.18, "zorder": 4},
    }
    base_tt = float(baseline.get("weight_travel_time", 1.0) or 1.0)
    if base_tt <= 0:
        base_tt = 1.0

    if {"weight_travel_time", "mean_nearest_open_acf_distance_to_demand"}.issubset(fac.columns):
        fig, ax = plt.subplots(figsize=(8, 5))
        for ud in ("L", "M", "H"):
            sub = fac[fac["unmet_level"] == ud]
            if sub.empty:
                continue
            style = ud_style[ud]
            sub = sub.sort_values("weight_travel_time")
            ax.scatter(
                sub["weight_travel_time"] * style["dodge"],
                sub["mean_nearest_open_acf_distance_to_demand"],
                label=f"UD={ud}",
                s=90,
                color=style["color"],
                marker=style["marker"],
                edgecolors="black",
                linewidths=0.7,
                alpha=0.95,
                zorder=style["zorder"],
            )
        if pd.notna(baseline.get("mean_nearest_open_acf_distance_to_demand")):
            ax.scatter(
                [base_tt],
                [float(baseline["mean_nearest_open_acf_distance_to_demand"])],
                label="Baseline",
                s=160,
                color=base_color,
                marker="*",
                edgecolors="black",
                linewidths=0.8,
                zorder=6,
            )
            ax.axhline(
                float(baseline["mean_nearest_open_acf_distance_to_demand"]),
                color=base_color,
                linestyle="--",
                linewidth=1.0,
                alpha=0.6,
            )
        ax.set_xscale("log")
        ax.set_xlabel("Travel Time Weight (log)")
        ax.set_ylabel("Mean Nearest open-ACF Distance to Demand Points")
        ax.legend()
        _savefig(fig, _out(plot_dir, "factorial_acf_demand_distance.png", instance_name))

    if {"weight_travel_time", "mean_pairwise_open_acf_distance"}.issubset(fac.columns):
        fig, ax = plt.subplots(figsize=(8, 5))
        for ud in ("L", "M", "H"):
            sub = fac[fac["unmet_level"] == ud]
            if sub.empty:
                continue
            style = ud_style[ud]
            sub = sub.sort_values("weight_travel_time")
            ax.scatter(
                sub["weight_travel_time"] * style["dodge"],
                sub["mean_pairwise_open_acf_distance"],
                label=f"UD={ud}",
                s=90,
                color=style["color"],
                marker=style["marker"],
                edgecolors="black",
                linewidths=0.7,
                alpha=0.95,
                zorder=style["zorder"],
            )
        if pd.notna(baseline.get("mean_pairwise_open_acf_distance")):
            ax.scatter(
                [base_tt],
                [float(baseline["mean_pairwise_open_acf_distance"])],
                label="Baseline",
                s=160,
                color=base_color,
                marker="*",
                edgecolors="black",
                linewidths=0.8,
                zorder=6,
            )
            ax.axhline(
                float(baseline["mean_pairwise_open_acf_distance"]),
                color=base_color,
                linestyle="--",
                linewidth=1.0,
                alpha=0.6,
            )
        ax.set_xscale("log")
        ax.set_xlabel("Travel Time Weight (log)")
        ax.set_ylabel("Mean Pairwise Distance among Open ACFs")
        ax.legend()
        _savefig(fig, _out(plot_dir, "factorial_acf_pairwise_distance.png", instance_name))

    # 4) OOS KPIs with Baseline bar
    oos_cols = [
        ("oos_pct_on_time_transfer", "% On-Time Transfer"),
        ("oos_pct_on_time_evacuation", "% On-Time Evacuation"),
        ("oos_pct_not_evacuated", "% Not Evacuated"),
    ]
    present_oos = [(c, lab) for c, lab in oos_cols if c in fac.columns and c in baseline.index]
    if present_oos:
        fig, axes = plt.subplots(1, len(present_oos), figsize=(5.5 * len(present_oos), 4.2))
        if len(present_oos) == 1:
            axes = [axes]
        ordered = fac.sort_values("weight_scheme")
        x_labels = ["Baseline"] + (
            ordered["weight_scheme"].astype(str).str.replace(r"^FAC_", "", regex=True).tolist()
        )
        x = range(len(x_labels))
        for ax, (col, lab) in zip(axes, present_oos):
            values = [float(baseline[col])] + ordered[col].astype(float).tolist()
            colors = [base_color] + ["#4C72B0"] * len(ordered)
            ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.4)
            ax.set_xticks(list(x))
            ax.set_xticklabels(x_labels, rotation=90, fontsize=6)
            ax.set_ylabel(lab)
            ax.axhline(float(baseline[col]), color=base_color, linestyle="--", linewidth=1.0, alpha=0.7)
        _savefig(fig, _out(plot_dir, "factorial_oos_kpis.png", instance_name))
