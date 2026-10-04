"""Core runner for objective-function sensitivity analysis."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from Constants import Constants
from Instance import Instance
from ObjectivePolicyWeights import (
    ObjectivePolicyWeights,
    build_factorial_schemes,
    build_oat_schemes,
    default_weight_schemes,
    parse_factorial_levels_string,
    parse_oat_multipliers_string,
    parse_objective_weights_string,
)
from ScenarioTree import ScenarioTree
from Solver import Solver
from TestIdentificator import TestIdentificator
from objective_function_sensitivity_analysis.exceptions import (
    InstanceNotFoundError,
    InvalidWeightConfigurationError,
    ModelConfigurationError,
    OptimizationFailedError,
)
from objective_function_sensitivity_analysis.metrics import build_result_record
from objective_function_sensitivity_analysis.plots import (
    _prefixed_filename,
    generate_all_plots,
    generate_factorial_baseline_comparison_plots,
    generate_factorial_plots,
)
LOGGER = logging.getLogger("objective_function_sensitivity")


def package_root() -> Path:
    return Path(__file__).resolve().parent


def default_output_dir() -> Path:
    return package_root()


def artifact_path(directory: Path, filename: str, instance_name: Optional[str]) -> Path:
    """Build an output path with the instance name as a filename prefix."""
    return Path(directory) / _prefixed_filename(filename, instance_name)


def get_git_metadata() -> Dict[str, Optional[str]]:
    meta = {"git_commit": None, "git_branch": None}
    try:
        meta["git_commit"] = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(package_root().parent),
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
        meta["git_branch"] = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=str(package_root().parent),
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
    except Exception:
        pass
    return meta


def load_instance_or_raise(instance_name: str) -> Instance:
    pickle_path = Path(Constants.PathInstances) / f"{instance_name}.pkl"
    if not pickle_path.exists():
        raise InstanceNotFoundError(
            f"Instance '{instance_name}' not found at '{pickle_path.resolve()}'. "
            "Generate it first with --Action GenerateInstances, or choose an existing instance."
        )
    instance = Instance(instance_name)
    instance.LoadInstanceFromPickle(instance_name)
    return instance


def build_test_identifier(args) -> TestIdentificator:
    if args.Model != "2Stage" or args.Solver != "MIP":
        raise ModelConfigurationError(
            "Objective-function sensitivity analysis requires Model=2Stage and Solver=MIP "
            f"(got Model={args.Model}, Solver={args.Solver})."
        )
    seed_index = int(getattr(args, "ScenarioSeed", -1))
    if seed_index < 0 or seed_index >= len(Constants.SeedArray):
        seed = Constants.SeedArray[0]
    else:
        seed = Constants.SeedArray[seed_index]

    bbc = args.bbcsetting
    if isinstance(bbc, str) and ":" in bbc:
        bbc = bbc.split(":", 1)[0].strip()

    return TestIdentificator(
        instance_name=args.Instance,
        model=args.Model,
        solver=args.Solver,
        nrScenario=str(args.NrScenario),
        seed=seed,
        sampling=args.ScenarioGeneration,
        phaobj=args.PHAObj,
        phapenalty=args.PHAPenalty,
        alnsRL=int(args.ALNSRL),
        alnsRL_DeepQ=int(args.ALNSRL_DeepQ),
        rlSelectionMethod=getattr(args, "RLSelectionMethod", "e-greedy"),
        bbcsetting=bbc,
        clustering=args.ClusteringMethod,
    )


def configure_runtime_constants(test_identifier: TestIdentificator, run_evaluation: bool = True) -> None:
    """
    Match the main Solve path for Gurobi/scenario settings.
    Enable solution pickles so Evaluator can reload first-stage decisions.
    Evaluation itself is toggled per run (Constants.Evaluation_Part).

    Also activates the objective-function sensitivity flags so ObjW_* naming,
    policy weights, and alternate Test output paths take effect for this script only.
    """
    Constants.SensitivityAnalysis = True
    Constants.SensitivityAnalysis_ObjectiveFunction = True
    Constants.ClusteringMethod = test_identifier.Clustering
    Constants.UserInterface = False
    Constants.LauchEvalAfterSolve = bool(run_evaluation)
    Constants.PrintSolutionFileToExcel = True
    Constants.PrintSolutionFileToPickle = True
    Constants.PrintDetailsExcelFiles = False
    # Keep False until we enter evaluation, matching main.py Solve() timing.
    Constants.Evaluation_Part = False


INSAMPLE_KPI_NAMES = [
    "grb_cost",
    "grb_time",
    "grb_gap",
    "grb_nr_constraints",
    "grb_nr_variables",
    "pha_cost",
    "pha_nr_iteration",
    "total_time",
    "acf_establishment_cost",
    "land_rescue_vehicle_cost",
    "backup_hospital_cost",
    "casualty_transfer_cost",
    "unsatisfied_casualties_cost",
    "discharged_patients_cost",
    "land_evacuated_patients_cost",
    "aerial_evacuated_patients_cost",
    "unevacuated_patient_cost",
    "available_cap_facility_cost",
    "pct_on_time_transfer",
    "pct_on_time_evacuation",
    "pct_not_evacuated",
    "nr_acf_established",
    "nr_land_res_vehicle_assigned",
    "nr_backup_hospitals",
    "evaluation_duration",
]

OOS_RESULT_NAMES = [
    "oos_mean",
    "oos_lb",
    "oos_ub",
    "oos_min_average",
    "oos_max_average",
    "oos_nrerror",
] + [f"oos_{name}" for name in INSAMPLE_KPI_NAMES]


def _zip_named(values, names) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if values is None:
        return out
    for i, name in enumerate(names):
        if i < len(values):
            out[name] = values[i]
    return out


def evaluate_like_main(
    instance: Instance,
    scheme_test_identifier: TestIdentificator,
    solver: Solver,
    evaluator_identifier,
    result_output_dir: Path,
) -> Dict[str, Any]:
    """
    Mirror main.py post-solve evaluation:
      Constants.Evaluation_Part = True
      Evaluator(...).RunEvaluation()
    Writes TestResult_*.xlsx under ./Test/ and under result_output_dir.
    Out-of-sample scenarios use Constants.EvaluationScenarioSeed + NrEvaluation,
    so they are identical across all weight schemes.
    """
    from Evaluator import Evaluator

    os.makedirs("./Test", exist_ok=True)
    os.makedirs("./Evaluations", exist_ok=True)
    os.makedirs(result_output_dir, exist_ok=True)

    Constants.Evaluation_Part = True
    try:
        evaluator = Evaluator(instance, scheme_test_identifier, evaluator_identifier, solver)
        evaluator.ResultOutputDir = str(result_output_dir)
        evaluator.RunEvaluation()

        record: Dict[str, Any] = {
            "evaluation_completed": True,
            "test_result_path": getattr(evaluator, "LastTestResultPath", None),
            "test_result_path_sensitivity": getattr(evaluator, "LastAlternateTestResultPath", None),
        }
        record.update({f"insample_{k}": v for k, v in _zip_named(evaluator.InSampleTestResult, INSAMPLE_KPI_NAMES).items()})
        record.update(_zip_named(evaluator.OutOfSampleTestResult, OOS_RESULT_NAMES))
        return record
    finally:
        Constants.Evaluation_Part = False


def solve_with_weights(
    instance: Instance,
    test_identifier: TestIdentificator,
    scenario_tree: ScenarioTree,
    weights: ObjectivePolicyWeights,
):
    solver = Solver(instance, test_identifier)
    solution, mipsolver = solver.LocationAllocation(
        treestructur=solver.TreeStructure,
        averagescenario=False,
        recordsolveinfo=True,
        scenariotree=scenario_tree,
        objective_policy_weights=weights,
        accept_feasible=True,
        quiet=True,
    )
    if solution is None:
        status = getattr(getattr(mipsolver, "LocAloc", None), "status", None)
        raise OptimizationFailedError(f"No usable Gurobi solution (status={status}).")
    # Persist solution pickles/excel so Evaluator can reload first-stage decisions.
    solver.PrintSolutionToFile(solution)
    return solution, mipsolver, solver


def create_frozen_scenario_tree(instance: Instance, test_identifier: TestIdentificator) -> ScenarioTree:
    int_part = "".join(ch for ch in str(test_identifier.NrScenario) if ch.isdigit())
    nr_scenario = int(int_part) if int_part else 50
    tree_structure = [1, nr_scenario]
    LOGGER.info(
        "Generating frozen scenario tree: n=%s, method=%s, seed=%s, clustering=%s",
        nr_scenario,
        test_identifier.ScenarioSampling,
        test_identifier.ScenarioSeed,
        Constants.ClusteringMethod,
    )
    return ScenarioTree(
        instance=instance,
        tree_structure=tree_structure,
        scenario_seed=test_identifier.ScenarioSeed,
        averagescenariotree=False,
        scenariogenerationmethod=test_identifier.ScenarioSampling,
    )


def select_weight_schemes(args) -> Dict[str, Tuple[str, ObjectivePolicyWeights]]:
    """
    Returns mapping scheme_name -> (sensitivity_mode, weights).
    """
    schemes: Dict[str, Tuple[str, ObjectivePolicyWeights]] = {}
    mode = (args.SensitivityMode or "all").lower()

    if args.ObjectiveWeights:
        try:
            custom = parse_objective_weights_string(args.ObjectiveWeights)
        except Exception as exc:
            raise InvalidWeightConfigurationError(str(exc)) from exc
        schemes["Custom"] = ("custom", custom)

    run_schemes = bool(args.RunAllSchemes) or mode in ("all", "schemes", "baseline")
    run_oat = mode in ("all", "oat")
    run_factorial = mode in ("factorial", "priority_combo")

    if mode == "baseline" and not args.ObjectiveWeights:
        schemes["Baseline"] = ("baseline", ObjectivePolicyWeights.baseline())
        return schemes

    if run_schemes:
        for name, weights in default_weight_schemes().items():
            if mode == "baseline" and name != "Baseline":
                continue
            schemes[name] = ("schemes", weights)

    if run_oat:
        multipliers = parse_oat_multipliers_string(getattr(args, "OATMultipliers", "auto"))
        for name, weights in build_oat_schemes(multipliers=multipliers).items():
            # Skip exact baseline duplicate if already present
            if name.endswith("_x1") and "Baseline" in schemes:
                continue
            schemes[name] = ("oat", weights)

    if run_factorial:
        # Always include Baseline with factorial so OOS/first-stage comparisons
        # and factorial+Baseline plots have a same-instance reference.
        if "Baseline" not in schemes:
            schemes["Baseline"] = ("baseline", ObjectivePolicyWeights.baseline())
        levels = parse_factorial_levels_string(getattr(args, "FactorialLevels", "auto"))
        for name, weights in build_factorial_schemes(levels=levels).items():
            schemes[name] = ("factorial", weights)

    if not schemes:
        raise InvalidWeightConfigurationError("No weight schemes selected.")
    return schemes


def _strip_private(row: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in row.items() if not str(k).startswith("_")}


def _load_existing_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as exc:
            LOGGER.warning("Could not read existing CSV %s (%s); starting fresh.", path, exc)
    return pd.DataFrame()


def _append_or_replace_by_scheme(
    existing: pd.DataFrame,
    new_df: pd.DataFrame,
    key: str = "weight_scheme",
) -> pd.DataFrame:
    """Keep prior rows; replace any schemes that reappear in new_df."""
    if new_df is None or new_df.empty:
        return existing
    if existing is None or existing.empty:
        return new_df.copy()
    if key not in new_df.columns:
        return pd.concat([existing, new_df], ignore_index=True, sort=False)
    drop_keys = set(new_df[key].astype(str).tolist())
    if key in existing.columns:
        keep = existing[~existing[key].astype(str).isin(drop_keys)].copy()
    else:
        keep = existing.copy()
    return pd.concat([keep, new_df], ignore_index=True, sort=False)


def _successful_schemes_from_results(results_csv: Path) -> set:
    df = _load_existing_csv(results_csv)
    if df.empty or "weight_scheme" not in df.columns:
        return set()
    if "success" in df.columns:
        ok = df[df["success"] == True]
    else:
        ok = df
    return set(ok["weight_scheme"].astype(str).tolist())


def write_outputs(
    rows: List[Dict[str, Any]],
    output_dir: Path,
    config: Dict[str, Any],
    generate_plots: bool,
    append: bool = False,
    plot_scope: str = "all",
    instance_name: Optional[str] = None,
) -> Dict[str, Path]:
    """
    Write / update result CSVs.

    append=True (typical for SensitivityMode=factorial):
      merge new rows into existing CSVs by weight_scheme (replace duplicates,
      keep other modes intact). Also writes a factorial-only snapshot CSV.
    plot_scope:
      "all"       -> generate_all_plots (may rewrite existing plot files)
      "factorial" -> only factorial-specific plots under plots/factorial/
      "none"      -> no plots

    All artifact filenames are prefixed with ``instance_name`` when provided.
    """
    output_dir = Path(output_dir)
    results_dir = output_dir / "results"
    plots_dir = output_dir / "plots"
    config_dir = output_dir / "config"
    solutions_dir = output_dir / "solutions"
    for d in (results_dir, plots_dir, config_dir, solutions_dir):
        d.mkdir(parents=True, exist_ok=True)

    if instance_name is None:
        instance_name = config.get("instance")

    clean_rows = [_strip_private(r) for r in rows]
    new_df = pd.DataFrame(clean_rows)

    results_csv = artifact_path(results_dir, "objective_function_sensitivity_results.csv", instance_name)
    detailed_csv = artifact_path(results_dir, "objective_function_sensitivity_detailed.csv", instance_name)
    long_csv = artifact_path(results_dir, "objective_function_sensitivity_long.csv", instance_name)
    summary_csv = artifact_path(results_dir, "objective_function_sensitivity_summary.csv", instance_name)
    acf_csv = artifact_path(results_dir, "acf_spatial_geometry_by_scheme.csv", instance_name)
    factorial_csv = artifact_path(results_dir, "factorial_priority_combinations.csv", instance_name)

    if append and not new_df.empty:
        merged = _append_or_replace_by_scheme(_load_existing_csv(results_csv), new_df)
    elif append and new_df.empty:
        merged = _load_existing_csv(results_csv)
    else:
        merged = new_df

    merged.to_csv(results_csv, index=False)
    merged.to_csv(detailed_csv, index=False)

    # Long format for plotting components (rebuild from merged)
    long_rows = []
    for _, r in merged.iterrows():
        if not bool(r.get("success", False)):
            continue
        for comp in ("travel_time", "evacuation_risk", "unmet_demand", "threat_risk"):
            long_rows.append(
                {
                    "run_id": r.get("run_id"),
                    "weight_scheme": r.get("weight_scheme"),
                    "sensitivity_mode": r.get("sensitivity_mode"),
                    "component": comp,
                    "weight": r.get(f"weight_{comp}"),
                    "weight_share": r.get(f"{comp}_weight_share"),
                    "raw": r.get(f"{comp}_raw"),
                    "weighted": r.get(f"{comp}_weighted"),
                    "total_objective": r.get("total_objective"),
                    "unmet_demand": r.get("unmet_demand"),
                    "non_evacuated_patients": r.get("non_evacuated_patients"),
                    "oos_mean": r.get("oos_mean"),
                    "oos_pct_on_time_transfer": r.get("oos_pct_on_time_transfer"),
                    "oos_pct_on_time_evacuation": r.get("oos_pct_on_time_evacuation"),
                    "oos_pct_not_evacuated": r.get("oos_pct_not_evacuated"),
                    "oos_nr_acf_established": r.get("oos_nr_acf_established"),
                }
            )
    pd.DataFrame(long_rows).to_csv(long_csv, index=False)

    # Paper-ready summary: schemes/baseline/custom (+ factorial in its own file)
    summary_cols = [
        "weight_scheme",
        "weight_travel_time",
        "weight_evacuation_risk",
        "weight_unmet_demand",
        "weight_threat_risk",
        "total_objective",
        "insample_grb_cost",
        "oos_mean",
        "oos_pct_on_time_transfer",
        "oos_pct_on_time_evacuation",
        "oos_pct_not_evacuated",
        "oos_nr_acf_established",
        "oos_nr_land_res_vehicle_assigned",
        "oos_nr_backup_hospitals",
        "oos_evaluation_duration",
        "travel_time_raw",
        "evacuation_risk_raw",
        "unmet_demand_raw",
        "threat_risk_raw",
        "acf_locations",
        "number_of_active_acfs",
        "mean_nearest_open_acf_distance_to_demand",
        "max_nearest_open_acf_distance_to_demand",
        "mean_pairwise_open_acf_distance",
        "mean_open_acf_distance_to_demand_centroid",
        "open_acf_centroid_x",
        "open_acf_centroid_y",
        "runtime",
        "mip_gap",
        "test_result_path",
        "test_result_path_sensitivity",
        "success",
        "solution_valid",
        "evaluation_completed",
    ]
    present = [c for c in summary_cols if c in merged.columns]
    summary_df = merged[merged["sensitivity_mode"].isin(["schemes", "baseline", "custom"])][present].copy() if "sensitivity_mode" in merged.columns else merged[present].copy()
    if summary_df.empty:
        summary_df = merged[present].copy()
    summary_df.to_csv(summary_csv, index=False)

    # ACF spatial geometry across all successful runs
    acf_cols = [
        "weight_scheme",
        "sensitivity_mode",
        "weight_travel_time",
        "weight_evacuation_risk",
        "weight_unmet_demand",
        "weight_threat_risk",
        "acf_locations",
        "number_of_active_acfs",
        "total_land_vehicles_assigned",
        "number_of_backup_hospital_links",
        "mean_nearest_open_acf_distance_to_demand",
        "max_nearest_open_acf_distance_to_demand",
        "min_nearest_open_acf_distance_to_demand",
        "mean_allpair_demand_to_open_acf_distance",
        "mean_pairwise_open_acf_distance",
        "mean_open_acf_distance_to_demand_centroid",
        "open_acf_centroid_x",
        "open_acf_centroid_y",
        "total_transport_time",
        "average_transport_time",
        "oos_nr_acf_established",
        "oos_nr_land_res_vehicle_assigned",
        "oos_nr_backup_hospitals",
        "success",
    ]
    acf_present = [c for c in acf_cols if c in merged.columns]
    if acf_present and "success" in merged.columns:
        acf_df = merged[merged["success"] == True][acf_present].copy()
    elif acf_present:
        acf_df = merged[acf_present].copy()
    else:
        acf_df = pd.DataFrame()
    acf_df.to_csv(acf_csv, index=False)

    # Factorial-only snapshot (always rewrite from merged factorial rows)
    if "sensitivity_mode" in merged.columns:
        fac_df = merged[merged["sensitivity_mode"] == "factorial"].copy()
    else:
        fac_df = pd.DataFrame()
    fac_df.to_csv(factorial_csv, index=False)

    # Config: for append mode, merge weight_schemes into prior config if present
    config_path = artifact_path(config_dir, "sensitivity_config.json", instance_name)
    if append and config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as fh:
                prior = json.load(fh)
            prior_schemes = prior.get("weight_schemes", {})
            prior_schemes.update(config.get("weight_schemes", {}))
            config["weight_schemes"] = prior_schemes
            config["prior_timestamp"] = prior.get("timestamp")
        except Exception:
            pass
    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, default=str)

    schemes_path = artifact_path(config_dir, "weight_schemes.json", instance_name)
    with open(schemes_path, "w", encoding="utf-8") as fh:
        json.dump(config.get("weight_schemes", {}), fh, indent=2)

    if generate_plots and plot_scope == "all":
        generate_all_plots(merged, plots_dir, instance_name=instance_name)
    elif generate_plots and plot_scope == "factorial":
        generate_factorial_plots(merged, plots_dir / "factorial", instance_name=instance_name)
        try:
            generate_factorial_baseline_comparison_plots(
                merged, plots_dir / "factorial+Baseline", instance_name=instance_name
            )
        except ValueError as exc:
            LOGGER.warning("Skipped factorial+Baseline plots: %s", exc)

    return {
        "results_csv": results_csv,
        "long_csv": long_csv,
        "summary_csv": summary_csv,
        "detailed_csv": detailed_csv,
        "acf_spatial_csv": acf_csv,
        "factorial_csv": factorial_csv,
        "config_path": config_path,
        "plots_dir": plots_dir,
    }


def generate_plots_only(args) -> Dict[str, Any]:
    """Regenerate plots from existing CSVs without re-solving."""
    output_dir = Path(args.OutputDir) if args.OutputDir else default_output_dir()
    instance_name = getattr(args, "Instance", None)
    results_csv = artifact_path(
        output_dir / "results",
        "objective_function_sensitivity_results.csv",
        instance_name,
    )
    if not results_csv.exists():
        raise FileNotFoundError(f"No results CSV found at {results_csv}")
    df = pd.read_csv(results_csv)
    if instance_name and "instance" in df.columns:
        df = df[df["instance"].astype(str) == str(instance_name)].copy()
    scope = (getattr(args, "PlotsOnly", None) or getattr(args, "SensitivityMode", "all") or "all").lower()
    plots_dir = output_dir / "plots"
    if scope in ("factorial", "priority_combo"):
        generate_factorial_plots(df, plots_dir / "factorial", instance_name=instance_name)
        target = plots_dir / "factorial"
    elif scope in ("factorial+baseline", "factorial_baseline", "factiorial+baseline"):
        generate_factorial_baseline_comparison_plots(
            df, plots_dir / "factorial+Baseline", instance_name=instance_name
        )
        target = plots_dir / "factorial+Baseline"
    else:
        generate_all_plots(df, plots_dir, instance_name=instance_name)
        target = plots_dir
    return {
        "instance": instance_name,
        "model": getattr(args, "Model", None),
        "solver": getattr(args, "Solver", None),
        "nr_scenario": getattr(args, "NrScenario", None),
        "scenario_generation": getattr(args, "ScenarioGeneration", None),
        "clustering": getattr(args, "ClusteringMethod", None),
        "nrevaluation": getattr(args, "nrevaluation", None),
        "run_evaluation": False,
        "runs_completed": 0,
        "successful": 0,
        "failed": 0,
        "runtime_seconds": 0.0,
        "output_dir": str(output_dir),
        "paths": {"plots_dir": str(target), "results_csv": str(results_csv)},
        "baseline_objective": None,
        "baseline_oos_mean": None,
        "baseline_valid": None,
    }


def run_sensitivity_analysis(args) -> Dict[str, Any]:
    started = time.time()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    output_dir = Path(args.OutputDir) if args.OutputDir else default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    test_dir = output_dir / "Test"
    test_dir.mkdir(parents=True, exist_ok=True)

    mode = (args.SensitivityMode or "all").lower()
    append = bool(getattr(args, "AppendResults", False)) or mode in ("factorial", "priority_combo")
    plot_scope = "factorial" if mode in ("factorial", "priority_combo") else "all"
    if not bool(getattr(args, "GeneratePlots", True)):
        plot_scope = "none"

    run_evaluation = bool(getattr(args, "RunEvaluation", True))
    test_identifier = build_test_identifier(args)
    configure_runtime_constants(test_identifier, run_evaluation=run_evaluation)
    instance = load_instance_or_raise(test_identifier.InstanceName)
    scenario_tree = create_frozen_scenario_tree(instance, test_identifier)
    schemes = select_weight_schemes(args)

    from EvaluatorIdentificator import EvaluatorIdentificator

    evaluator_identifier = EvaluatorIdentificator(
        getattr(args, "policy", "_"),
        int(getattr(args, "nrevaluation", 500)),
        int(getattr(args, "timehorizon", 1)),
        int(getattr(args, "allscenario", 0)),
    )

    # Skip schemes already successfully present when appending (resume support)
    results_csv = artifact_path(
        output_dir / "results",
        "objective_function_sensitivity_results.csv",
        test_identifier.InstanceName,
    )
    already_done = set()
    if append and bool(getattr(args, "SkipExisting", True)):
        already_done = _successful_schemes_from_results(results_csv)
        if already_done:
            before = len(schemes)
            schemes = {k: v for k, v in schemes.items() if k not in already_done}
            LOGGER.info(
                "SkipExisting: %d/%d schemes already in results CSV; %d remaining",
                before - len(schemes),
                before,
                len(schemes),
            )

    # Ensure Baseline is first when present
    ordered_names = list(schemes.keys())
    if "Baseline" in schemes:
        ordered_names = ["Baseline"] + [n for n in ordered_names if n != "Baseline"]

    rows: List[Dict[str, Any]] = []
    baseline_row = None
    baseline_decisions = None
    # Try to load Baseline from existing CSV for deltas when running factorial alone
    if append and baseline_row is None:
        prior = _load_existing_csv(results_csv)
        if not prior.empty and "weight_scheme" in prior.columns:
            base_hits = prior[prior["weight_scheme"].astype(str) == "Baseline"]
            if not base_hits.empty:
                baseline_row = base_hits.iloc[-1].to_dict()
    n_success = 0
    n_failed = 0

    LOGGER.info(
        "Starting sensitivity analysis with %d runs (mode=%s, append=%s, evaluation=%s, nrevaluation=%s)",
        len(ordered_names),
        mode,
        append,
        run_evaluation,
        evaluator_identifier.NrEvaluation,
    )
    if not ordered_names:
        LOGGER.warning("No schemes left to run (all skipped as existing). Writing/updating outputs only.")

    for idx, name in enumerate(ordered_names, start=1):
        mode_name, weights = schemes[name]
        scheme_test_id = test_identifier.with_sensitivity_scheme(name)
        run_id = f"{test_identifier.InstanceName}_{name}_{idx:03d}"
        LOGGER.info("[%d/%d] Solving scheme=%s weights=%s", idx, len(ordered_names), name, weights.as_dict())
        print(f"\n=== Sensitivity run {idx}/{len(ordered_names)}: {name} ===")
        try:
            # Restore in-sample clustering before each optimize (OOS eval sets ClusteringMethod='NoC').
            Constants.ClusteringMethod = test_identifier.Clustering
            Constants.Evaluation_Part = False

            solution, mipsolver, solver = solve_with_weights(
                instance, scheme_test_id, scenario_tree, weights
            )
            row = build_result_record(
                run_id=run_id,
                weight_scheme=name,
                sensitivity_mode=mode_name if name != "Baseline" else "baseline",
                test_identifier=scheme_test_id,
                weights=weights,
                solution=solution,
                mipsolver=mipsolver,
                baseline_row=baseline_row,
                baseline_decisions=baseline_decisions,
                timestamp=timestamp,
            )

            if run_evaluation and row.get("success"):
                print(f"--- Evaluating scheme {name} on shared out-of-sample scenarios "
                      f"(n={evaluator_identifier.NrEvaluation}) ---")
                eval_record = evaluate_like_main(
                    instance=instance,
                    scheme_test_identifier=scheme_test_id,
                    solver=solver,
                    evaluator_identifier=evaluator_identifier,
                    result_output_dir=test_dir,
                )
                row.update(eval_record)
                # Prefer OOS operational KPIs for paper-facing deltas when available
                if baseline_row is not None:
                    oos_keys = [
                        "oos_mean",
                        "oos_pct_on_time_transfer",
                        "oos_pct_on_time_evacuation",
                        "oos_pct_not_evacuated",
                        "oos_nr_acf_established",
                        "oos_nr_land_res_vehicle_assigned",
                        "oos_nr_backup_hospitals",
                    ]
                    for key in oos_keys:
                        if key in row and key in baseline_row and row[key] is not None and baseline_row[key] is not None:
                            try:
                                new_v = float(row[key])
                                base_v = float(baseline_row[key])
                                row[f"delta_{key}"] = new_v - base_v
                                row[f"pct_change_{key}"] = (
                                    None if abs(base_v) < 1e-12 else 100.0 * (new_v - base_v) / base_v
                                )
                            except (TypeError, ValueError):
                                pass
            elif run_evaluation:
                row["evaluation_completed"] = False

            if name == "Baseline" and row.get("success"):
                baseline_row = _strip_private(row)
                baseline_decisions = row.get("_decisions")
            if row.get("success"):
                n_success += 1
            else:
                n_failed += 1

            if args.SaveDetailedResults and solution is not None:
                snap = {
                    "run_id": run_id,
                    "weight_scheme": name,
                    "weights": weights.as_dict(),
                    "GRBCost": getattr(solution, "GRBCost", None),
                    "GRBGap": getattr(solution, "GRBGap", None),
                    "ACF": list(map(float, solution.ACFEstablishment_x_wi[0])),
                    "acf_locations": row.get("acf_locations"),
                    "mean_nearest_open_acf_distance_to_demand": row.get(
                        "mean_nearest_open_acf_distance_to_demand"
                    ),
                    "mean_pairwise_open_acf_distance": row.get("mean_pairwise_open_acf_distance"),
                    "mean_open_acf_distance_to_demand_centroid": row.get(
                        "mean_open_acf_distance_to_demand_centroid"
                    ),
                    "open_acf_centroid_x": row.get("open_acf_centroid_x"),
                    "open_acf_centroid_y": row.get("open_acf_centroid_y"),
                    "number_of_active_acfs": row.get("number_of_active_acfs"),
                    "total_land_vehicles_assigned": row.get("total_land_vehicles_assigned"),
                    "number_of_backup_hospital_links": row.get("number_of_backup_hospital_links"),
                    "oos_mean": row.get("oos_mean"),
                    "oos_pct_on_time_transfer": row.get("oos_pct_on_time_transfer"),
                    "oos_pct_on_time_evacuation": row.get("oos_pct_on_time_evacuation"),
                    "oos_pct_not_evacuated": row.get("oos_pct_not_evacuated"),
                    "oos_nr_acf_established": row.get("oos_nr_acf_established"),
                    "test_result_path": row.get("test_result_path"),
                    "test_result_path_sensitivity": row.get("test_result_path_sensitivity"),
                }
                snap_path = output_dir / "solutions" / f"{run_id}.json"
                snap_path.parent.mkdir(parents=True, exist_ok=True)
                with open(snap_path, "w", encoding="utf-8") as fh:
                    json.dump(snap, fh, indent=2)
            rows.append(row)
        except Exception as exc:
            n_failed += 1
            LOGGER.exception("Run failed: %s", name)
            rows.append(
                build_result_record(
                    run_id=run_id,
                    weight_scheme=name,
                    sensitivity_mode=mode_name,
                    test_identifier=scheme_test_id,
                    weights=weights,
                    solution=None,
                    mipsolver=None,
                    baseline_row=baseline_row,
                    baseline_decisions=baseline_decisions,
                    timestamp=timestamp,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    git_meta = get_git_metadata()
    factorial_levels = None
    if mode in ("factorial", "priority_combo"):
        factorial_levels = parse_factorial_levels_string(getattr(args, "FactorialLevels", "auto"))
    config = {
        "timestamp": timestamp,
        "instance": test_identifier.InstanceName,
        "model": test_identifier.Model,
        "solver": test_identifier.Solver,
        "nr_scenario": test_identifier.NrScenario,
        "scenario_generation": test_identifier.ScenarioSampling,
        "scenario_seed": test_identifier.ScenarioSeed,
        "clustering": test_identifier.Clustering,
        "policy": getattr(args, "policy", "_"),
        "nrevaluation": evaluator_identifier.NrEvaluation,
        "run_evaluation": run_evaluation,
        "evaluation_scenario_seed": Constants.EvaluationScenarioSeed,
        "sensitivity_mode": args.SensitivityMode,
        "append_results": append,
        "weight_scheme": getattr(args, "WeightScheme", None),
        "objective_weights_cli": getattr(args, "ObjectiveWeights", None),
        "oat_multipliers": parse_oat_multipliers_string(getattr(args, "OATMultipliers", "auto")),
        "factorial_levels": factorial_levels,
        "gurobi": {
            "MIPTimeLimit": Constants.MIPTimeLimit,
            "MIPGap": instance.My_EpGap,
            "Threads": 1,
            "OutputFlag": Constants.ModelOutputFlag,
            "RiskModel": Constants.Risk,
        },
        "weight_schemes": {
            name: {"mode": m, "weights": weights.as_dict(), "weight_shares": weights.weight_shares()}
            for name, (m, weights) in schemes.items()
        },
        "objective_component_mapping": {
            "travel_time": "CasualtyTransfer_Var * travel times",
            "evacuation_risk": "Land+Aerial evacuated patients * risk coeffs",
            "unmet_demand": "UnsatisfiedCasualties_Var * Casualty_Shortage_Cost",
            "threat_risk": "UnevacuatedPatients_Var(T) * cumulative threat risk",
        },
        "notes": [
            "Policy weights multiply existing instance/model coefficients.",
            "No objective-component normalization is applied.",
            "In-sample scenarios are generated once and reused across all weight settings.",
            "Out-of-sample evaluation uses the same EvaluationScenarioSeed and NrEvaluation for every scheme.",
            "Comparable operational KPIs (on-time / not evacuated / ACF counts) should be taken from oos_* fields.",
            "This is weighted-scalarization sensitivity analysis, not multi-objective reformulation.",
            "Factorial mode uses scale-aware L/M/H levels and appends to existing result CSVs.",
        ],
        **git_meta,
    }

    paths = write_outputs(
        rows=rows,
        output_dir=output_dir,
        config=config,
        generate_plots=bool(args.GeneratePlots) and plot_scope != "none",
        append=append,
        plot_scope=plot_scope if plot_scope != "none" else "none",
        instance_name=test_identifier.InstanceName,
    )
    paths["test_dir"] = test_dir

    elapsed = time.time() - started
    summary = {
        "instance": test_identifier.InstanceName,
        "model": test_identifier.Model,
        "solver": test_identifier.Solver,
        "nr_scenario": test_identifier.NrScenario,
        "scenario_generation": test_identifier.ScenarioSampling,
        "clustering": test_identifier.Clustering,
        "nrevaluation": evaluator_identifier.NrEvaluation,
        "run_evaluation": run_evaluation,
        "runs_completed": len(rows),
        "successful": n_success,
        "failed": n_failed,
        "skipped_existing": len(already_done),
        "runtime_seconds": elapsed,
        "output_dir": str(output_dir),
        "paths": {k: str(v) for k, v in paths.items()},
        "baseline_objective": None if baseline_row is None else baseline_row.get("total_objective"),
        "baseline_oos_mean": None if baseline_row is None else baseline_row.get("oos_mean"),
        "baseline_valid": None if baseline_row is None else baseline_row.get("solution_valid"),
    }
    return summary


def print_execution_summary(summary: Dict[str, Any]) -> None:
    print("\nObjective Function Sensitivity Analysis")
    print("---------------------------------------")
    print(f"Instance: {summary['instance']}")
    print(f"Model: {summary['model']}")
    print(f"Solver: {summary['solver']}")
    print(f"Scenarios (in-sample): {summary['nr_scenario']}")
    print(f"Scenario generation: {summary['scenario_generation']}")
    print(f"Clustering: {summary['clustering']}")
    print(f"Out-of-sample evaluation: {summary.get('run_evaluation')}")
    print(f"NrEvaluation (OOS): {summary.get('nrevaluation')}")
    print("")
    print(f"Runs completed: {summary['runs_completed']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Wall-clock runtime (s): {summary['runtime_seconds']:.2f}")
    if summary.get("baseline_objective") is not None:
        print(f"Baseline in-sample objective: {summary['baseline_objective']}")
        print(f"Baseline OOS mean: {summary.get('baseline_oos_mean')}")
        print(f"Baseline validation OK: {summary['baseline_valid']}")
    print("")
    print("Results:")
    print(f"  {summary['paths'].get('results_csv')}")
    print(f"  {summary['paths'].get('summary_csv')}")
    print(f"  {summary['paths'].get('acf_spatial_csv')}")
    if summary["paths"].get("factorial_csv"):
        print(f"  {summary['paths'].get('factorial_csv')}")
    if summary.get("skipped_existing"):
        print(f"Skipped existing schemes: {summary['skipped_existing']}")
    print("")
    print("Test workbooks (InSample / OutOfSample):")
    print(f"  {summary['paths'].get('test_dir')}")
    print("  (also under ./Test/ with ObjW_<scheme> in the filename)")
    print("")
    print("Plots:")
    print(f"  {summary['paths'].get('plots_dir')}")
