# Objective-Function Sensitivity Analysis

## 1. Purpose

This module studies how **decision-maker preference weights** on the existing
second-stage composite objective affect solutions of the stochastic
casualty-response planning (CRP) model.

It is a response to Reviewer #2 concerning compatibility of the four
patient-centred performance measures combined in the second-stage objective.

## 2. Relationship to the main model

- The **main paper model is unchanged**: same constraints, same stochastic
  structure, same Gurobi `2Stage` + `MIP` pathway (`Solver.LocationAllocation`
  → `MIPSolver.BuildModel` → `optimize`).
- This analysis does **not** reformulate the paper as multi-objective
  optimization and does **not** normalize objective components.
- Only **policy preference weights** that multiply the existing second-stage
  coefficients are varied.
- Scenarios are generated **once** and reused for every weight setting.

## 3. Four second-stage objective components

| Component | Existing variables | Existing coefficients |
|-----------|--------------------|------------------------|
| Travel time | `CasualtyTransfer_Var` (\(q\)) | Land travel times `Time_D_H_Land` / `Time_D_A_Land` |
| Evacuation risk | `LandEvacuatedPatients_Var` (\(u^L\)), `AerialEvacuatedPatients_Var` (\(u^A\)) | `EvacuationRiskCost[j]` × land/aerial risk factors |
| Unmet demand | `UnsatisfiedCasualties_Var` (\(\mu\)) | `Casualty_Shortage_Cost[j]` |
| Threat risk | `UnevacuatedPatients_Var` (\(\Phi\)) at final period \(T\) | `EvacuationRiskCost[j]` × cumulative threat risk |

First-stage terms (ACF establishment, vehicle assignment, backup coordination)
remain at instance coefficients and are **not** varied as policy weights.

**Note:** The instance uses **two** injury levels (`j=0` high priority,
`j=1` lower priority), not three priority classes.

## 4. How weights are varied

Policy weights default to baseline \((1,1,1,1)\), which reproduces the current
model exactly.

Primary schemes (relative multipliers on baseline coefficients):

| Scheme | Travel | Evac. risk | Unmet | Threat |
|--------|--------|------------|-------|--------|
| Baseline | 1 | 1 | 1 | 1 |
| TravelTime_Priority | 2 | 1 | 1 | 1 |
| EvacuationRisk_Priority | 1 | 2 | 1 | 1 |
| UnmetDemand_Priority | 1 | 1 | 2 | 1 |
| ThreatRisk_Priority | 1 | 1 | 1 | 2 |

One-at-a-time (OAT) analysis uses **component-aware** multiplier grids by default
(`--OATMultipliers auto`), because instance coefficients differ by orders of
magnitude (shortage ≈ 150k, travel time ≈ 5–70):

| Component | Default OAT multipliers (others fixed at 1) |
|-----------|-----------------------------------------------|
| `travel_time` | 1, 10, 20, 50, 100, 500, 1000, 2000, 5000, 10000 |
| `evacuation_risk` | 0.1, 0.25, 0.5, 1, 2, 5, 10 |
| `unmet_demand` | 0.001, 0.05, 0.1, 0.25, 0.5, 1, 2, 5 |
| `threat_risk` | 0.1, 0.25, 0.5, 1, 2, 5, 10 |

Override examples:

```bash
# Shared legacy list for every component
python run_objective_sensitivity.py --OATMultipliers 0.5,1,2

# Per-component override (unlisted components keep intelligent defaults)
python run_objective_sensitivity.py --OATMultipliers "travel_time=1,100,1000;unmet_demand=0.05,1"
```

Weight **shares** (normalized preference mix) are recorded for reporting only;
they are **not** used inside Gurobi.

## 5. How to run

From the project root. Default settings:

- `NrScenario=50`
- `Model=2Stage`, `Solver=MIP`
- `ScenarioGeneration=RQMC`, `Clustering=NoC`
- Out-of-sample evaluation **on** (`--nrevaluation 50` by default)

```bash
# Full analysis: baseline + primary schemes + OAT  (each scheme also runs 50 OOS evals by default)
python run_objective_sensitivity.py
```

`run_objective_sensitivity.py` turns on
`Constants.SensitivityAnalysis` and `Constants.SensitivityAnalysis_ObjectiveFunction`
for that process only. With both flags left `False` in `Constants.py`, a normal
`python main.py` run keeps pre-branch naming and objective coefficients.

```bash
# Recommended for the paper subsection first: baseline + 4 priority schemes
python run_objective_sensitivity.py --SensitivityMode schemes

# Scale-aware FACTORIAL combinations only (27 runs). Appends to existing CSVs;
# does NOT overwrite prior OAT/scheme rows. Factorial plots go under plots/factorial/.
python run_objective_sensitivity.py --SensitivityMode factorial

# Resume factorial if interrupted (skips schemes already successful in results CSV)
python run_objective_sensitivity.py --SensitivityMode factorial

# Regenerate factorial-only plots from existing CSVs (no re-solve)
python run_objective_sensitivity.py --PlotsOnly factorial
```

### Why factorial (not only OAT)?

OAT rarely moves ACF locations because shortage (~150k) and risk (~10k) dwarf travel (~35).
Factorial jointly varies priorities so corners can flip dominance:

| Factor | L | M | H |
|--------|---|---|---|
| Travel (`TT`) | 10 | 500 | 10000 |
| Risk (`RK`, applied to **both** evacuation + threat) | 0.1 | 2 | 10 |
| Unmet / shortage (`UD`) | 0.001 | 0.25 | 5 |

→ **27** combinations named `FAC_TT-H_RK-M_UD-L`, etc.
Evacuation and threat share one risk factor to avoid an 81-run 4-way grid.

```bash
# Baseline only (reproduce main 2Stage+MIP, then evaluate OOS)
python run_objective_sensitivity.py --SensitivityMode baseline

# Baseline only (reproduce main 2Stage+MIP, then evaluate OOS)
python run_objective_sensitivity.py --SensitivityMode baseline

# Faster debugging (no OOS evaluation)
python run_objective_sensitivity.py --SensitivityMode schemes --NoRunEvaluation

# Smaller OOS sample for a quick check
python run_objective_sensitivity.py --SensitivityMode schemes -n 50
```

Useful variants:

```bash
# Custom weights
python run_objective_sensitivity.py --SensitivityMode custom --ObjectiveWeights 1,2,1,1

# Another instance (must exist as Instances/<name>.pkl)
python run_objective_sensitivity.py --Instance 4_20_5_20_3_2_CRP
```

Default configuration matches:

`Action=Solve, Model=2Stage, Solver=MIP, NrScenario=50, ScenarioGeneration=RQMC, Clustering=NoC`,
plus the same out-of-sample evaluation pathway as `main.py`.

### Evaluation workflow (important)

For each weight scheme the module now:

1. Optimizes the **in-sample** 2Stage MIP under that scheme’s policy weights  
   (shared frozen in-sample scenarios across schemes).
2. Saves first-stage decisions to `Solutions/` (unique name includes `ObjW_<scheme>`).
3. Runs **out-of-sample** evaluation exactly like `main.py`  
   (`Evaluator` → `EvaluationSimulator`, seed = `Constants.EvaluationScenarioSeed`).
4. Writes `TestResult_*.xlsx` with sheets `Generic Information`, `InSample`, `OutOfSample`  
   under both `./Test/` and `objective_function_sensitivity_analysis/Test/`.

Comparable policy KPIs for the paper should come from **out-of-sample** fields:

- `oos_mean`
- `oos_pct_on_time_transfer`
- `oos_pct_on_time_evacuation`
- `oos_pct_not_evacuated`
- `oos_nr_acf_established`
- `oos_nr_land_res_vehicle_assigned`
- `oos_nr_backup_hospitals`

First-stage decisions are fixed from the in-sample solve; OOS scenarios are identical
across schemes, so these KPIs are comparable.

## 6. Where results are saved

```
objective_function_sensitivity_analysis/
├── README.md
├── config/
│   ├── sensitivity_config.json
│   └── weight_schemes.json
├── results/
│   ├── objective_function_sensitivity_results.csv
│   ├── objective_function_sensitivity_detailed.csv
│   ├── objective_function_sensitivity_summary.csv
│   ├── objective_function_sensitivity_long.csv
│   ├── acf_spatial_geometry_by_scheme.csv
│   └── factorial_priority_combinations.csv
├── Test/
│   └── TestResult_*_ObjW_<scheme>_*.xlsx   # same structure as ./Test/
├── solutions/
├── plots/
│   └── factorial/   # factorial-only figures (does not overwrite OAT/scheme plots)
└── scripts/
```

Also written (unique per scheme, no overwrite):

- `./Test/TestResult_<...>_ObjW_<scheme>_...xlsx`
- `./Solutions/<...>_ObjW_<scheme>_...` pickle/excel solution files
- `./Evaluations/...` evaluation intermediate pickles

ACF geometry uses instance plane coordinates `ACF_Position` / `DisasterArea_Position`
and Euclidean `Distance_D_A` / `Distance_A_A` (synthetic CRP instances are not
geographic lat/lon). Key columns in `acf_spatial_geometry_by_scheme.csv`:

- `acf_locations` — open ACF indices
- `mean_nearest_open_acf_distance_to_demand`
- `mean_pairwise_open_acf_distance`
- `mean_open_acf_distance_to_demand_centroid`
- `open_acf_centroid_x`, `open_acf_centroid_y`

## 7. CSV meanings

- **results / detailed**: one row per optimization run with weights, raw/weighted
  components, operational KPIs, decision-change metrics, Gurobi stats, validation.
- **summary**: compact paper-ready table for primary schemes.
- **long**: component-level long format for plotting.

## 8. Plot meanings

| File | Question answered |
|------|-------------------|
| `objective_by_weight_scheme.png` | How does the scalarized objective change by scheme? (not comparable welfare) |
| `objective_components_by_scheme.png` | How do the four raw components trade off? |
| `unmet_demand_vs_threat_risk.png` | Unmet demand vs non-evacuated patients |
| `transport_time_vs_evacuation_risk.png` | Transport time vs evacuation risk |
| `on_time_vs_weight.png` | On-time transfer/evacuation by scheme |
| `non_evacuated_vs_weight.png` | Threat-weight OAT vs non-evacuated / threat component |
| `unmet_demand_sensitivity.png` | Unmet-weight OAT effects |
| `priority_class_effects.png` | High vs low priority unmet demand |
| `decision_changes_by_scheme.png` | How much first-stage decisions change vs baseline |
| `empirical_tradeoff_unmet_vs_threat.png` | Empirical scalarization trade-offs (not a formal Pareto set) |
| `acf_distance_vs_travel_weight.png` | Open-ACF proximity / spacing vs travel-time weight |
| `acf_centroid_vs_travel_weight.png` | Centroid of established ACFs as travel weight changes |
| `transport_time_vs_travel_weight.png` | Flow-weighted transport time vs travel-time weight |

## 9. Reproducibility

- Same instance pickle, scenario count, RQMC seed, clustering, and Gurobi
  parameters across all runs.
- Frozen scenario tree reused for every weight setting.
- `config/sensitivity_config.json` stores the full configuration, including
  Git commit/branch when available.

## 10. Methodological limitations

- This is **weighted-scalarization sensitivity**, not a multi-objective
  reformulation and not a guaranteed Pareto frontier.
- Total objective values under different weights are **not** directly
  comparable as welfare measures.
- Evacuation-risk and threat-risk terms both use `EvacuationRiskCost[j]` in the
  existing model; only their **policy multipliers** are varied independently here.
- Injury levels are binary (high/low), matching the instance data.
