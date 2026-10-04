"""
Command-line entry point for objective-function sensitivity analysis.

Default:
  python run_objective_sensitivity.py

Change INSTANCE below (or pass --Instance) to switch instances. All result
and plot filenames are automatically prefixed with that instance name.
"""

from __future__ import annotations

import argparse
import logging
import platform
import sys
from pathlib import Path

from objective_function_sensitivity_analysis.runner import (
    default_output_dir,
    generate_plots_only,
    print_execution_summary,
    run_sensitivity_analysis,
)

# =============================================================================
# CHANGE THE INSTANCE HERE (or override with --Instance on the command line)
# =============================================================================
INSTANCE = "4_20_5_15_3_3_CRP"
# Examples:
# INSTANCE = "4_20_5_20_3_1_CRP"
# INSTANCE = "4_15_5_15_3_1_CRP"
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sensitivity analysis for second-stage objective-function policy weights "
            "in the existing 2Stage MIP casualty-response model."
        )
    )

    # Mirror main.py core settings (Windows defaults match main.py).
    parser.add_argument(
        "--Action",
        help="Action to perform",
        type=str,
        choices=["GenerateInstances", "Solve"],
        default="Solve",
    )
    parser.add_argument(
        "--Instance",
        help="Instance name (default: INSTANCE variable at top of this script)",
        type=str,
        default=INSTANCE,
    )
    parser.add_argument(
        "--Model",
        help="Stochastic model type",
        type=str,
        choices=["Average", "2Stage"],
        default="2Stage",
    )
    parser.add_argument(
        "--Solver",
        help="Solver type",
        type=str,
        choices=["MIP", "ALNS", "PHA", "BBC"],
        default="MIP",
    )
    parser.add_argument(
        "--NrScenario",
        help="The number of scenarios used for optimization",
        type=str,
        default="50",
    )
    parser.add_argument(
        "--PHAObj",
        help="Obj. function of PHA either Quadratic or Linear",
        type=str,
        choices=["Q", "L"],
        default="Q",
    )
    parser.add_argument(
        "--PHAPenalty",
        help="Penalty Parameter (rho) in PHA",
        type=str,
        choices=["S", "D", "DL"],
        default="S",
    )
    parser.add_argument(
        "--ALNSRL",
        help="Whether we use RL in ALNS or not",
        type=int,
        choices=[0, 1],
        default=0,
    )
    parser.add_argument(
        "--ALNSRL_DeepQ",
        help="Deep Q-Learning(1) or Q-Learning(0)",
        type=int,
        choices=[0, 1],
        default=0,
    )
    parser.add_argument(
        "-c",
        "--bbcsetting",
        help="Enhancements?",
        choices=[
            "NE",
            "JM",
            "NM",
            "JS",
            "NS",
            "JW",
            "NW",
            "JL",
            "NL",
            "AE",
            "NE: NoEnhancement",
            "JM: JustMultiCut",
            "NM: NoMultiCut",
            "JS: JustStrongCut",
            "NS: NoStrongCut",
            "JW: JustWarmUp",
            "NW: NoWarmUp",
            "JL: JustLBF",
            "NL: NoLBF",
            "AE: AllEnhancement",
        ],
        default="NS",
    )
    parser.add_argument(
        "--ScenarioGeneration",
        help="Which Type of Sampling?",
        type=str,
        choices=["MC", "RQMC", "QMC"],
        default="RQMC",
    )
    parser.add_argument(
        "-Cluster",
        "--ClusteringMethod",
        help="The method used for Clustering scenarios",
        type=str,
        choices=["NoC", "KM", "KMPP", "SOM", "DB"],
        default="NoC",
    )
    parser.add_argument("-p", "--policy", help="NearestNeighbor", type=str, default="_")
    parser.add_argument(
        "-n",
        "--nrevaluation",
        help="nr scenario used for evaluation",
        type=int,
        default=50,
    )
    parser.add_argument(
        "-s",
        "--ScenarioSeed",
        help="Index into Constants.SeedArray (-1 => first seed)",
        type=int,
        default=-1,
    )

    # Sensitivity-specific arguments
    parser.add_argument(
        "--SensitivityMode",
        help=(
            "Which sensitivity experiment to run. "
            "'factorial' = scale-aware travel×risk×unmet combinations (appends to existing CSVs)."
        ),
        type=str,
        choices=["baseline", "schemes", "oat", "all", "custom", "factorial", "priority_combo"],
        default="all",
    )
    parser.add_argument(
        "--WeightScheme",
        help="Optional primary scheme filter (reserved for future use)",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--ObjectiveWeights",
        help=(
            "Custom weights: 'travel_time=1,evacuation_risk=1,unmet_demand=1,threat_risk=1' "
            "or '1,1,1,1'"
        ),
        type=str,
        default=None,
    )
    parser.add_argument(
        "--RunAllSchemes",
        help="Include the primary policy schemes even when SensitivityMode=oat/custom",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--OATMultipliers",
        help=(
            "OAT multipliers. Default 'auto' = component-aware grids "
            "(travel needs large multipliers; unmet/threat/evac use 0.1–10 scale). "
            "Shared list: '0.5,1,2'. Per-component: "
            "'travel_time=1,10,100;unmet_demand=0.05,1;threat_risk=0.1,1,10'"
        ),
        type=str,
        default="auto",
    )
    parser.add_argument(
        "--FactorialLevels",
        help=(
            "Factorial L/M/H levels. Default auto = "
            "travel 10/500/10000, risk 0.1/2/10, unmet 0.001/0.25/5. "
            "Override: 'travel_time=L:10,M:500,H:10000;risk=L:0.1,M:2,H:10;unmet_demand=L:0.001,M:0.25,H:5'"
        ),
        type=str,
        default="auto",
    )
    parser.add_argument(
        "--AppendResults",
        help="Append/merge into existing result CSVs instead of overwriting (default on for factorial)",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--SkipExisting",
        help="When appending, skip schemes already successful in the results CSV (default: on)",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--NoSkipExisting",
        dest="SkipExisting",
        action="store_false",
        help="Re-solve schemes even if already present in results CSV",
    )
    parser.add_argument(
        "--PlotsOnly",
        help=(
            "Do not solve; regenerate plots from existing CSVs. "
            "'factorial' -> plots/factorial/; "
            "'factorial+baseline' -> plots/factorial+Baseline/ (factorial overlays with Baseline)"
        ),
        type=str,
        choices=["all", "factorial", "priority_combo", "factorial+baseline", "factorial_baseline"],
        default=None,
    )
    parser.add_argument(
        "--SaveDetailedResults",
        help="Save per-run JSON solution snapshots",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--NoSaveDetailedResults",
        dest="SaveDetailedResults",
        action="store_false",
        help="Disable per-run JSON snapshots",
    )
    parser.add_argument(
        "--GeneratePlots",
        help="Generate publication-quality plots after all runs",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--NoGeneratePlots",
        dest="GeneratePlots",
        action="store_false",
        help="Skip plot generation",
    )
    parser.add_argument(
        "--RunEvaluation",
        help="After each solve, run the same out-of-sample evaluation as main.py",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--NoRunEvaluation",
        dest="RunEvaluation",
        action="store_false",
        help="Skip out-of-sample evaluation (optimize only)",
    )
    parser.add_argument(
        "--OutputDir",
        help="Output directory (default: ./objective_function_sensitivity_analysis)",
        type=str,
        default=str(default_output_dir()),
    )
    parser.add_argument(
        "--LogLevel",
        help="Logging level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.LogLevel),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.Action != "Solve":
        print("This sensitivity module only supports --Action Solve.")
        return 2

    if args.Model != "2Stage" or args.Solver != "MIP":
        print(
            "ERROR: Sensitivity analysis must use --Model 2Stage --Solver MIP "
            "to study the exact Gurobi mathematical objective."
        )
        return 2

    try:
        if getattr(args, "PlotsOnly", None):
            summary = generate_plots_only(args)
        else:
            summary = run_sensitivity_analysis(args)
    except Exception as exc:
        logging.exception("Sensitivity analysis failed")
        print(f"\nFATAL: {type(exc).__name__}: {exc}")
        return 1

    print_execution_summary(summary)
    return 0 if summary.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    # This entry point enables Constants.SensitivityAnalysis and
    # SensitivityAnalysis_ObjectiveFunction for the duration of the run.
    # Leaving both False in Constants.py keeps main.py naming/behavior unchanged.
    sys.exit(main())
