"""Extract objective components, operational metrics, and Gurobi stats from a solved model."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gurobipy import GRB

from Constants import Constants
from ObjectivePolicyWeights import ObjectivePolicyWeights
from objective_function_sensitivity_analysis.exceptions import (
    ObjectiveValidationError,
    ResultExtractionError,
)


def _safe_pct_change(new: float, base: float) -> Optional[float]:
    if base is None or abs(base) < 1e-12:
        return None
    return 100.0 * (new - base) / base


def _sum_or_zero(values) -> float:
    try:
        return float(np.sum(values))
    except Exception:
        return 0.0


def extract_gurobi_performance(mipsolver) -> Dict[str, Any]:
    model = mipsolver.LocAloc
    status = int(model.status)
    record: Dict[str, Any] = {
        "gurobi_status": status,
        "gurobi_status_name": _status_name(status),
        "runtime": float(getattr(model, "Runtime", float("nan"))),
        "num_vars": int(model.NumVars),
        "num_constrs": int(model.NumConstrs),
        "num_binary_vars": int(getattr(model, "NumBinVars", 0)),
        "num_integer_vars": int(getattr(model, "NumIntVars", 0)),
        "num_continuous_vars": int(model.NumVars) - int(getattr(model, "NumIntVars", 0)),
    }
    for attr, key in (
        ("ObjBound", "objective_bound"),
        ("MIPGap", "mip_gap"),
        ("NodeCount", "node_count"),
        ("IterCount", "simplex_iterations"),
        ("BarIterCount", "barrier_iterations"),
    ):
        try:
            record[key] = float(getattr(model, attr))
        except Exception:
            record[key] = None
    return record


def _status_name(status: int) -> str:
    mapping = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INTERRUPTED: "INTERRUPTED",
    }
    return mapping.get(status, f"STATUS_{status}")


def compute_raw_objective_components(mipsolver) -> Dict[str, float]:
    """
    Recompute unweighted (baseline-coefficient) second-stage component values
    plus unchanged first-stage contributions from the incumbent solution.
    """
    inst = mipsolver.Instance
    p = mipsolver.ScenarioProbability

    travel_time_raw = 0.0
    for w in mipsolver.ScenarioSet:
        for t in inst.TimeBucketSet:
            for j in inst.InjuryLevelSet:
                for l in inst.DisasterAreaSet:
                    for u in inst.MedFacilitySet:
                        for m in inst.RescueVehicleSet:
                            index = mipsolver.GetIndexCasualtyTransferVariables(w, t, j, l, u, m)
                            if index not in mipsolver.CasualtyTransfer_Var:
                                continue
                            x = mipsolver.CasualtyTransfer_Var[index].X
                            if u < inst.NrHospitals:
                                travel_time_raw += inst.Time_D_H_Land[l][u] * p * x
                            else:
                                travel_time_raw += inst.Time_D_A_Land[l][u - inst.NrHospitals] * p * x

    unmet_demand_raw = 0.0
    for w in mipsolver.ScenarioSet:
        for t in inst.TimeBucketSet:
            for j in inst.InjuryLevelSet:
                for l in inst.DisasterAreaSet:
                    index = mipsolver.GetIndexUnsatisfiedCasualtiesVariables(w, t, j, l)
                    if index not in mipsolver.UnsatisfiedCasualties_Var:
                        continue
                    unmet_demand_raw += (
                        inst.Casualty_Shortage_Cost[j]
                        * p
                        * mipsolver.UnsatisfiedCasualties_Var[index].X
                    )

    land_evac_raw = 0.0
    for w in mipsolver.ScenarioSet:
        for t in inst.TimeBucketSet:
            for j in inst.InjuryLevelSet:
                for h in inst.HospitalSet:
                    for u in inst.MedFacilitySet:
                        for m in inst.RescueVehicleSet:
                            index = mipsolver.GetIndexLandEvacuatedPatientsVariables(w, t, j, h, u, m)
                            if index not in mipsolver.LandEvacuatedPatients_Var:
                                continue
                            land_evac_raw += (
                                mipsolver.GetLandEvacuatedPatientsCoeff(w, t, j, h, u, m)
                                * mipsolver.LandEvacuatedPatients_Var[index].X
                            )

    aerial_evac_raw = 0.0
    for w in mipsolver.ScenarioSet:
        for t in inst.TimeBucketSet:
            for j in inst.InjuryLevelSet:
                for h in inst.HospitalSet:
                    for i in inst.ACFSet:
                        for hprime in inst.HospitalSet:
                            for m in inst.RescueVehicleSet:
                                index = mipsolver.GetIndexAerialEvacuatedPatientsVariables(
                                    w, t, j, h, i, hprime, m
                                )
                                if index not in mipsolver.AerialEvacuatedPatients_Var:
                                    continue
                                aerial_evac_raw += (
                                    mipsolver.GetAerialEvacuatedPatientsCoeff(
                                        w, t, j, h, i, hprime, m
                                    )
                                    * mipsolver.AerialEvacuatedPatients_Var[index].X
                                )

    evacuation_risk_raw = land_evac_raw + aerial_evac_raw

    threat_risk_raw = 0.0
    t_last = inst.TimeBucketSet[-1]
    for w in mipsolver.ScenarioSet:
        for j in inst.InjuryLevelSet:
            for h in inst.HospitalSet:
                index = mipsolver.GetIndexUnevacuatedPatientsVariables(w, t_last, j, h)
                if index not in mipsolver.UnevacuatedPatients_Var:
                    continue
                threat_risk_raw += (
                    mipsolver.GetUnevacuatedPatientsCoeff(w, t_last, j, h)
                    * mipsolver.UnevacuatedPatients_Var[index].X
                )

    first_stage_raw = 0.0
    for w in mipsolver.ScenarioSet:
        for i in inst.ACFSet:
            index = mipsolver.GetIndexACFEstablishmentVariable(w, i)
            if index in mipsolver.ACFEstablishment_Var:
                first_stage_raw += (
                    inst.Fixed_Cost_ACF_Objective[i] * p * mipsolver.ACFEstablishment_Var[index].X
                )
        for i in inst.ACFSet:
            for m in inst.RescueVehicleSet:
                index = mipsolver.GetIndexLandRescueVehicleVariable(w, i, m)
                if index in mipsolver.LandRescueVehicle_Var:
                    first_stage_raw += (
                        inst.VehicleAssignment_Cost[m] * p * mipsolver.LandRescueVehicle_Var[index].X
                    )
        for h in inst.HospitalSet:
            for hprime in inst.HospitalSet:
                index = mipsolver.GetIndexBackupHospitalVariable(w, h, hprime)
                if index in mipsolver.BackupHospital_Var:
                    first_stage_raw += (
                        inst.CoordinationCost[h, hprime] * p * mipsolver.BackupHospital_Var[index].X
                    )

    return {
        "travel_time_raw": float(travel_time_raw),
        "evacuation_risk_raw": float(evacuation_risk_raw),
        "land_evacuation_risk_raw": float(land_evac_raw),
        "air_evacuation_risk_raw": float(aerial_evac_raw),
        "unmet_demand_raw": float(unmet_demand_raw),
        "threat_risk_raw": float(threat_risk_raw),
        "first_stage_raw": float(first_stage_raw),
    }


def validate_weighted_objective(
    components: Dict[str, float],
    weights: ObjectivePolicyWeights,
    gurobi_objective: float,
    tol: float = 1e-4,
    rtol: float = 1e-6,
) -> Dict[str, Any]:
    travel_w = weights.travel_time * components["travel_time_raw"]
    evac_w = weights.evacuation_risk * components["evacuation_risk_raw"]
    unmet_w = weights.unmet_demand * components["unmet_demand_raw"]
    threat_w = weights.threat_risk * components["threat_risk_raw"]
    reconstructed = (
        components["first_stage_raw"] + travel_w + evac_w + unmet_w + threat_w
    )
    abs_err = abs(reconstructed - gurobi_objective)
    denom = max(1.0, abs(gurobi_objective))
    rel_err = abs_err / denom
    ok = abs_err <= tol or rel_err <= rtol
    message = "OK" if ok else (
        f"Objective reconstruction mismatch: reconstructed={reconstructed:.8g}, "
        f"gurobi={gurobi_objective:.8g}, abs_err={abs_err:.3g}, rel_err={rel_err:.3g}"
    )
    return {
        "travel_time_weighted": float(travel_w),
        "evacuation_risk_weighted": float(evac_w),
        "unmet_demand_weighted": float(unmet_w),
        "threat_risk_weighted": float(threat_w),
        "total_objective_reconstructed": float(reconstructed),
        "total_objective": float(gurobi_objective),
        "objective_recalculation_error": float(abs_err),
        "objective_recalculation_rel_error": float(rel_err),
        "solution_valid": bool(ok),
        "validation_message": message,
    }


def compute_acf_spatial_metrics(instance, acf_binary) -> Dict[str, Any]:
    """
    Spatial geometry of established ACFs vs demand points.

    Instance positions are plane (x, y) coordinates in Square_Dimension (not
    geographic lat/lon for the synthetic CRP instances). Distances reuse the
    instance Euclidean matrices Distance_D_A / Distance_A_A that also drive
    travel times.
    """
    acf = np.asarray(acf_binary, dtype=float).ravel()
    open_idx = [int(i) for i, v in enumerate(acf) if v > 0.5]
    n_open = len(open_idx)

    empty = {
        "acf_locations": ",".join(str(i) for i in open_idx),
        "number_of_active_acfs": float(n_open),
        "mean_nearest_open_acf_distance_to_demand": None,
        "max_nearest_open_acf_distance_to_demand": None,
        "min_nearest_open_acf_distance_to_demand": None,
        "mean_allpair_demand_to_open_acf_distance": None,
        "mean_pairwise_open_acf_distance": None,
        "open_acf_centroid_x": None,
        "open_acf_centroid_y": None,
        "mean_open_acf_distance_to_demand_centroid": None,
    }
    if n_open == 0:
        return empty

    dist_da = np.asarray(instance.Distance_D_A, dtype=float)
    # Nearest open ACF for each demand point
    nearest = []
    all_pairs = []
    for l in range(dist_da.shape[0]):
        d_row = [float(dist_da[l, i]) for i in open_idx]
        nearest.append(min(d_row))
        all_pairs.extend(d_row)

    # Pairwise distances among open ACFs (exclude self-entries which are 1000)
    pair_vals = []
    if n_open >= 2:
        dist_aa = np.asarray(instance.Distance_A_A, dtype=float)
        for a, i in enumerate(open_idx):
            for j in open_idx[a + 1 :]:
                pair_vals.append(float(dist_aa[i, j]))

    positions = np.asarray(instance.ACF_Position, dtype=float)
    demand_pos = np.asarray(instance.DisasterArea_Position, dtype=float)
    open_pos = positions[open_idx]
    centroid = open_pos.mean(axis=0)
    demand_centroid = demand_pos.mean(axis=0)
    dist_to_demand_centroid = float(
        np.sqrt(np.sum((open_pos - demand_centroid) ** 2, axis=1)).mean()
    )

    return {
        "acf_locations": ",".join(str(i) for i in open_idx),
        "number_of_active_acfs": float(n_open),
        "mean_nearest_open_acf_distance_to_demand": float(np.mean(nearest)),
        "max_nearest_open_acf_distance_to_demand": float(np.max(nearest)),
        "min_nearest_open_acf_distance_to_demand": float(np.min(nearest)),
        "mean_allpair_demand_to_open_acf_distance": float(np.mean(all_pairs)),
        "mean_pairwise_open_acf_distance": (
            float(np.mean(pair_vals)) if pair_vals else 0.0
        ),
        "open_acf_centroid_x": float(centroid[0]),
        "open_acf_centroid_y": float(centroid[1]),
        "mean_open_acf_distance_to_demand_centroid": dist_to_demand_centroid,
    }


def extract_operational_metrics(solution, mipsolver) -> Dict[str, Any]:
    """Operational KPIs derived from existing solution arrays / scenario data."""
    if solution is None:
        raise ResultExtractionError("Cannot extract metrics from a null solution.")

    inst = mipsolver.Instance
    scenarios = solution.Scenarioset
    n_scen = len(scenarios)
    metrics: Dict[str, Any] = {}

    # First-stage decisions (non-anticipative: use scenario 0)
    acf = np.array(solution.ACFEstablishment_x_wi[0], dtype=float)
    vehicles = np.array(solution.LandRescueVehicle_thetaVar_wim[0], dtype=float)
    backup = np.array(solution.BackupHospital_W_whhPrime[0], dtype=float)

    spatial = compute_acf_spatial_metrics(inst, acf)
    metrics.update(spatial)
    metrics["total_land_vehicles_assigned"] = float(np.sum(vehicles))
    metrics["number_of_backup_hospital_links"] = float(np.sum(backup > 0.5))


    # Aggregate patient/casualty quantities over scenarios
    total_casualty_demand = 0.0
    total_unmet = 0.0
    total_transfer = 0.0
    total_transport_time_flow = 0.0
    total_land_evac = 0.0
    total_air_evac = 0.0
    total_non_evacuated = 0.0
    total_patient_demand = 0.0
    on_time_transfer = 0.0
    on_time_evacuation = 0.0

    unmet_by_j = {j: 0.0 for j in inst.InjuryLevelSet}
    nonevac_by_j = {j: 0.0 for j in inst.InjuryLevelSet}
    demand_by_j = {j: 0.0 for j in inst.InjuryLevelSet}
    patient_demand_by_j = {j: 0.0 for j in inst.InjuryLevelSet}
    threat_by_j = {j: 0.0 for j in inst.InjuryLevelSet}

    t_last = inst.TimeBucketSet[-1]

    for w, scenario in enumerate(scenarios):
        for t in inst.TimeBucketSet:
            for j in inst.InjuryLevelSet:
                for l in inst.DisasterAreaSet:
                    dem = float(scenario.CasualtyDemand[t][j][l])
                    total_casualty_demand += dem
                    demand_by_j[j] += dem
                    mu = float(solution.UnsatisfiedCasualties_mu_wtjl[w][t][j][l])
                    total_unmet += mu
                    unmet_by_j[j] += mu
                    on_time_transfer += max(dem - mu, 0.0)
                    for u in inst.MedFacilitySet:
                        for m in inst.RescueVehicleSet:
                            q = float(solution.CasualtyTransfer_q_wtjlum[w][t][j][l][u][m])
                            total_transfer += q
                            if u < inst.NrHospitals:
                                total_transport_time_flow += q * float(inst.Time_D_H_Land[l][u])
                            else:
                                total_transport_time_flow += q * float(
                                    inst.Time_D_A_Land[l][u - inst.NrHospitals]
                                )

        for j in inst.InjuryLevelSet:
            for h in inst.HospitalSet:
                if float(scenario.HospitalDisruption[h]) != 1:
                    continue
                pd = float(scenario.PatientDemand[j][h])
                total_patient_demand += pd
                patient_demand_by_j[j] += pd
                phi0 = float(solution.UnevacuatedPatients_Phi_wtjh[w][0][j][h])
                phiT = float(solution.UnevacuatedPatients_Phi_wtjh[w][t_last][j][h])
                on_time_evacuation += max(pd - phi0, 0.0)
                total_non_evacuated += phiT
                nonevac_by_j[j] += phiT
                threat_by_j[j] += (
                    mipsolver.GetUnevacuatedPatientsCoeff(w, t_last, j, h) * phiT
                )

        for t in inst.TimeBucketSet:
            for j in inst.InjuryLevelSet:
                for h in inst.HospitalSet:
                    for u in inst.MedFacilitySet:
                        for m in inst.RescueVehicleSet:
                            total_land_evac += float(
                                solution.LandEvacuatedPatients_u_L_wtjhum[w][t][j][h][u][m]
                            )
                    for i in inst.ACFSet:
                        for hprime in inst.HospitalSet:
                            for m in inst.RescueVehicleSet:
                                total_air_evac += float(
                                    solution.AerialEvacuatedPatients_u_A_wtjhihPrimem[w][t][j][h][i][hprime][m]
                                )

    metrics["total_casualties"] = float(total_casualty_demand)
    metrics["served_casualties"] = float(total_transfer)
    metrics["unserved_casualties"] = float(total_unmet)
    metrics["unmet_demand"] = float(total_unmet)
    metrics["average_unmet_demand"] = float(total_unmet) / n_scen
    metrics["non_evacuated_patients"] = float(total_non_evacuated)
    metrics["average_non_evacuated_patients"] = float(total_non_evacuated) / n_scen
    metrics["total_transport_time"] = float(total_transport_time_flow)
    metrics["average_transport_time"] = (
        float(total_transport_time_flow) / total_transfer if total_transfer > 1e-12 else None
    )
    metrics["land_evacuation_count"] = float(total_land_evac)
    metrics["air_evacuation_count"] = float(total_air_evac)
    metrics["total_evacuation_count"] = float(total_land_evac + total_air_evac)
    metrics["patients_delivered_on_time"] = float(on_time_transfer)
    metrics["on_time_transfer_percentage"] = (
        100.0 * on_time_transfer / total_casualty_demand if total_casualty_demand > 1e-12 else None
    )
    metrics["patients_evacuated_on_time"] = float(on_time_evacuation)
    metrics["on_time_evacuation_percentage"] = (
        100.0 * on_time_evacuation / total_patient_demand if total_patient_demand > 1e-12 else None
    )
    metrics["total_patient_demand"] = float(total_patient_demand)

    # Injury levels: j=0 high priority, j=1 lower priority in this project
    for j in inst.InjuryLevelSet:
        label = "high_priority" if j == 0 else "low_priority"
        metrics[f"{label}_demand"] = float(demand_by_j[j])
        metrics[f"{label}_unmet"] = float(unmet_by_j[j])
        metrics[f"{label}_patient_demand"] = float(patient_demand_by_j[j])
        metrics[f"{label}_non_evacuated"] = float(nonevac_by_j[j])
        metrics[f"{label}_threat_risk_contrib"] = float(threat_by_j[j])

    # Reuse solution cost fields when available (baseline-coefficient contributions)
    metrics["final_acf_cost"] = float(getattr(solution, "FinalACFEstablishmentCost", 0.0) or 0.0)
    metrics["final_vehicle_cost"] = float(getattr(solution, "FinalLandRescueVehicleCost", 0.0) or 0.0)
    metrics["final_backup_cost"] = float(getattr(solution, "FinalBackupHospitalCost", 0.0) or 0.0)
    metrics["grb_cost"] = float(getattr(solution, "GRBCost", float("nan")))
    metrics["grb_gap"] = float(getattr(solution, "GRBGap", float("nan")))
    metrics["grb_time"] = float(getattr(solution, "GRBTime", float("nan")))
    metrics["grb_nr_variables"] = float(getattr(solution, "GRBNrVariables", float("nan")))
    metrics["grb_nr_constraints"] = float(getattr(solution, "GRBNrConstraints", float("nan")))

    return metrics


def extract_decision_vectors(solution) -> Dict[str, np.ndarray]:
    return {
        "acf": np.array(solution.ACFEstablishment_x_wi[0], dtype=float),
        "vehicles": np.array(solution.LandRescueVehicle_thetaVar_wim[0], dtype=float),
        "backup": np.array(solution.BackupHospital_W_whhPrime[0], dtype=float),
    }


def compare_decisions_to_baseline(
    current: Dict[str, np.ndarray],
    baseline: Optional[Dict[str, np.ndarray]],
) -> Dict[str, Any]:
    if baseline is None:
        return {
            "number_of_changed_binary_decisions": 0,
            "percentage_of_changed_binary_decisions": 0.0,
            "number_of_changed_acf_decisions": 0,
            "number_of_changed_backup_decisions": 0,
            "number_of_changed_vehicle_assignments": 0,
            "percentage_of_changed_vehicle_assignments": 0.0,
        }

    acf_diff = np.abs(current["acf"] - baseline["acf"]) > 0.5
    backup_diff = np.abs(current["backup"] - baseline["backup"]) > 0.5
    veh_diff = np.abs(current["vehicles"] - baseline["vehicles"]) > 1e-6

    n_bin = int(acf_diff.size + backup_diff.size)
    n_bin_changed = int(np.sum(acf_diff) + np.sum(backup_diff))
    n_veh = int(veh_diff.size)
    n_veh_changed = int(np.sum(veh_diff))

    return {
        "number_of_changed_binary_decisions": n_bin_changed,
        "percentage_of_changed_binary_decisions": (
            100.0 * n_bin_changed / n_bin if n_bin else 0.0
        ),
        "number_of_changed_acf_decisions": int(np.sum(acf_diff)),
        "number_of_changed_backup_decisions": int(np.sum(backup_diff)),
        "number_of_changed_vehicle_assignments": n_veh_changed,
        "percentage_of_changed_vehicle_assignments": (
            100.0 * n_veh_changed / n_veh if n_veh else 0.0
        ),
    }


def add_baseline_deltas(row: Dict[str, Any], baseline_row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if baseline_row is None:
        return row
    keys = [
        "total_objective",
        "travel_time_raw",
        "evacuation_risk_raw",
        "unmet_demand_raw",
        "threat_risk_raw",
        "unmet_demand",
        "non_evacuated_patients",
        "total_transport_time",
        "on_time_transfer_percentage",
        "on_time_evacuation_percentage",
        "number_of_active_acfs",
        "total_land_vehicles_assigned",
        "mean_nearest_open_acf_distance_to_demand",
        "mean_pairwise_open_acf_distance",
        "mean_open_acf_distance_to_demand_centroid",
    ]
    for key in keys:
        if key not in row or key not in baseline_row:
            continue
        if row[key] is None or baseline_row[key] is None:
            continue
        try:
            new_v = float(row[key])
            base_v = float(baseline_row[key])
        except (TypeError, ValueError):
            continue
        row[f"delta_{key}"] = new_v - base_v
        row[f"pct_change_{key}"] = _safe_pct_change(new_v, base_v)
    return row


def build_result_record(
    *,
    run_id: str,
    weight_scheme: str,
    sensitivity_mode: str,
    test_identifier,
    weights: ObjectivePolicyWeights,
    solution,
    mipsolver,
    baseline_row: Optional[Dict[str, Any]] = None,
    baseline_decisions: Optional[Dict[str, np.ndarray]] = None,
    timestamp: str = "",
    error: Optional[str] = None,
) -> Dict[str, Any]:
    shares = weights.weight_shares()
    row: Dict[str, Any] = {
        "run_id": run_id,
        "instance": test_identifier.InstanceName,
        "weight_scheme": weight_scheme,
        "sensitivity_mode": sensitivity_mode,
        "timestamp": timestamp,
        "model": test_identifier.Model,
        "solver": test_identifier.Solver,
        "nr_scenario": test_identifier.NrScenario,
        "scenario_generation": test_identifier.ScenarioSampling,
        "scenario_seed": test_identifier.ScenarioSeed,
        "clustering_method": test_identifier.Clustering,
        "weight_travel_time": weights.travel_time,
        "weight_evacuation_risk": weights.evacuation_risk,
        "weight_unmet_demand": weights.unmet_demand,
        "weight_threat_risk": weights.threat_risk,
        "travel_time_weight_share": shares["travel_time"],
        "evacuation_risk_weight_share": shares["evacuation_risk"],
        "unmet_demand_weight_share": shares["unmet_demand"],
        "threat_risk_weight_share": shares["threat_risk"],
        "success": False,
        "error_message": error or "",
    }

    if error or solution is None or mipsolver is None or getattr(mipsolver, "LocAloc", None) is None:
        row["solution_valid"] = False
        row["validation_message"] = error or "No solution"
        return row

    try:
        row.update(extract_gurobi_performance(mipsolver))
        components = compute_raw_objective_components(mipsolver)
        row.update(components)
        gurobi_obj = float(mipsolver.LocAloc.ObjVal)
        validation = validate_weighted_objective(components, weights, gurobi_obj)
        row.update(validation)
        if not validation["solution_valid"]:
            # Soft-fail: keep the row, mark invalid, continue analysis.
            row["error_message"] = validation["validation_message"]
        ops = extract_operational_metrics(solution, mipsolver)
        row.update(ops)
        decisions = extract_decision_vectors(solution)
        row.update(compare_decisions_to_baseline(decisions, baseline_decisions))
        row = add_baseline_deltas(row, baseline_row)
        row["success"] = True
        row["_decisions"] = decisions  # stripped before CSV write
    except Exception as exc:
        row["success"] = False
        row["solution_valid"] = False
        row["error_message"] = f"{type(exc).__name__}: {exc}"
        row["validation_message"] = row["error_message"]
    return row
