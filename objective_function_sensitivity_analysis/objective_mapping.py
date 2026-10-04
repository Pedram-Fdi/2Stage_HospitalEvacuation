"""
Objective-component mapping used by the sensitivity analysis.

Existing second-stage Gurobi objective terms (see MIPSolver):

A. Travel time
   Variable: CasualtyTransfer_Var (q)
   Coefficient: Time_D_H_Land / Time_D_A_Land * ScenarioProbability
   Getter: GetCasualtyTransferCoeff
   Policy weight: travel_time

B. Evacuation risk
   Variables: LandEvacuatedPatients_Var (u_L), AerialEvacuatedPatients_Var (u_A)
   Coefficient: EvacuationRiskCost[j] * Land/AerialEvacuationRisk_* * ScenarioProbability
   Getters: GetLandEvacuatedPatientsCoeff, GetAerialEvacuatedPatientsCoeff
   Policy weight: evacuation_risk

C. Unmet demand
   Variable: UnsatisfiedCasualties_Var (mu)
   Coefficient: Casualty_Shortage_Cost[j] * ScenarioProbability
   Getter: GetUnsatisfiedCasualtiesCoeff
   Policy weight: unmet_demand

D. Unevacuated threat risk
   Variable: UnevacuatedPatients_Var (Phi) at final period T only
   Coefficient: EvacuationRiskCost[j] * CumulativeThreatRisk_* * ScenarioProbability
   Getter: GetUnevacuatedPatientsCoeff
   Policy weight: threat_risk

First-stage terms (ACF establishment, vehicle assignment, backup coordination)
remain at their instance coefficients and are NOT varied as policy weights.
"""

OBJECTIVE_COMPONENT_MAPPING = {
    "travel_time": {
        "variables": ["CasualtyTransfer_Var"],
        "instance_parameters": ["Time_D_H_Land", "Time_D_A_Land"],
        "getter": "GetCasualtyTransferCoeff",
    },
    "evacuation_risk": {
        "variables": ["LandEvacuatedPatients_Var", "AerialEvacuatedPatients_Var"],
        "instance_parameters": [
            "EvacuationRiskCost",
            "LandEvacuationRisk_Constant/Linear/Exponential",
            "AerialEvacuationRisk_Constant/Linear/Exponential",
        ],
        "getters": ["GetLandEvacuatedPatientsCoeff", "GetAerialEvacuatedPatientsCoeff"],
    },
    "unmet_demand": {
        "variables": ["UnsatisfiedCasualties_Var"],
        "instance_parameters": ["Casualty_Shortage_Cost"],
        "getter": "GetUnsatisfiedCasualtiesCoeff",
    },
    "threat_risk": {
        "variables": ["UnevacuatedPatients_Var (final period only)"],
        "instance_parameters": [
            "EvacuationRiskCost",
            "CumulativeThreatRiskConstant/Linear/Exponential",
        ],
        "getter": "GetUnevacuatedPatientsCoeff",
    },
}
