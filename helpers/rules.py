
def determine_missing_impl(extracted: dict) -> list:
    """Determine which parameters still need to be collected based on business rules."""
    missing = []

    # Check Region FIRST - this is the most important parameter
    region_value = extracted.get("Region")
    if not region_value or region_value == "UNKNOWN":
        missing.append("Region")
    
    # Check Model
    model_value = extracted.get("Model")
    if not model_value or model_value == "UNKNOWN":
        missing.append("Model")
    
    # Check Solver
    solver_value = extracted.get("Solver")
    if not solver_value or solver_value == "UNKNOWN":
        missing.append("Solver")
    
    # If Model is 2Stage, check NrScenario
    if extracted.get("Model") == "2Stage":
        scenario_value = extracted.get("NrScenario")
        if not scenario_value or scenario_value == "UNKNOWN":
            missing.append("NrScenario")
    
    # If Solver is ALNS, check RL preference
    if extracted.get("Solver") == "ALNS":
        alnsrl_value = extracted.get("ALNSRL")
        if not alnsrl_value or alnsrl_value == "UNKNOWN":
            missing.append("ALNSRL")
        
        # If ALNSRL is 1, check ALNSRL_DeepQ
        alnsrl_str = str(extracted.get("ALNSRL", "")).strip()
        if alnsrl_str == "1":
            deepq_value = extracted.get("ALNSRL_DeepQ")
            if not deepq_value or deepq_value == "UNKNOWN":
                missing.append("ALNSRL_DeepQ")
    
    # Only check ClusteringMethod if Model is 2Stage
    if extracted.get("Model") == "2Stage":
        clustering_value = extracted.get("ClusteringMethod")
        if not clustering_value or clustering_value == "UNKNOWN":
            missing.append("ClusteringMethod")
    
    return missing

def question_for_param_impl(param: str) -> str:
    """Return the question prompt for a given missing parameter."""
    questions = {
        "Region": "Which region or city is affected by the disaster? (e.g., Montreal, Toronto, New York, etc.)",
        "Model": "Which model do you prefer: `Average` (Only an average scenario for the number of injured people) or `2Stage` (stochastic number of casualties)?",
        "Solver": "Which solver should be used: `MIP` (exact), `ALNS` (heuristic)?",
        "NrScenario": "How many scenarios should be considered for the stochastic model? (e.g., 50, 100, etc.)",
        "ALNSRL": "Do you want to use Reinforcement Learning in the heuristic method? (0 = No, 1 = Yes)",
        "ALNSRL_DeepQ": "Which type of RL? Deep Q-Learning (1) or regular Q-Learning (0)?",
        "ClusteringMethod": "Choose the clustering method: `NoC` (No Clustering), `KM` (K-Means), `KMPP` (K-Means++), `SOM` (Self-Organizing Map), or `DB` (Decision-Based)?"
    }
    return questions.get(param, f'Please provide value for {param}')
