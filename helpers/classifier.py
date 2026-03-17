import re
from langchain.schema import HumanMessage


def classify_single_input_impl(llm, user_message: str, key: str) -> str:
    """Classify a single parameter input using the provided LLM or rule-based logic."""
    # Handle Region parameter specially
    if key == "Region":
        region_name = user_message.strip()
        region_name = region_name.replace("region", "").replace("city", "").replace("area", "").strip()
        region_name = ' '.join(word.capitalize() for word in region_name.split())
        return region_name if region_name else "UNKNOWN"

    # Define valid choices for parameters
    valid_choices = {
        "Model": ["Average", "2Stage"],
        "Solver": ["MIP", "ALNS"],
        "ALNSRL": ["0", "1"],
        "ALNSRL_DeepQ": ["0", "1"],
        "ClusteringMethod": ["NoC", "KM", "KMPP", "SOM", "DB"]
    }

    user_lower = user_message.lower().strip()

    # Key-specific rule-based shortcuts
    if key == "ALNSRL_DeepQ":
        if any(term in user_lower for term in ["deep q", "dql", "deep-q", "deepq", "deep q learning", "deep q-learning"]):
            return "1"
        elif any(term in user_lower for term in ["q learning", "q-learning", "regular q", "standard q", "basic q"]):
            return "0"
        elif any(term in user_lower for term in ["yes", "y", "1", "true"]):
            return "1"
        elif any(term in user_lower for term in ["no", "n", "0", "false"]):
            return "0"

    if key == "ALNSRL":
        if any(term in user_lower for term in ["yes", "y", "1", "true", "with rl", "use rl", "reinforcement learning", "rl"]):
            return "1"
        elif any(term in user_lower for term in ["no", "n", "0", "false", "without rl", "no rl", "basic alns", "standard alns"]):
            return "0"

    if key == "Model":
        if any(term in user_lower for term in ["average", "deterministic", "exact", "single scenario"]):
            return "Average"
        elif any(term in user_lower for term in ["2stage", "two stage", "stochastic", "scenarios", "uncertainty"]):
            return "2Stage"

    if key == "Solver":
        if any(term in user_lower for term in ["mip", "exact", "optimal", "mixed integer"]):
            return "MIP"
        elif any(term in user_lower for term in ["alns", "heuristic", "greedy", "adaptive large neighborhood"]):
            return "ALNS"

    if key == "ClusteringMethod":
        if any(term in user_lower for term in ["noc", "no clustering", "no cluster", "none", "no clustering method", "no cluster method"]):
            return "NoC"
        elif any(term in user_lower for term in ["kmpp", "k-means++", "k means++", "kmeans++", "k-means plus", "km++"]):
            return "KMPP"
        elif any(term in user_lower for term in ["km", "k-means", "k means", "kmeans"]) and not any(term in user_lower for term in ["++", "plus", "pp"]):
            return "KM"
        elif any(term in user_lower for term in ["som", "self organizing", "self-organizing", "self organizing map"]):
            return "SOM"
        elif any(term in user_lower for term in ["db", "decision based", "decision-based"]):
            return "DB"

    # Fallback to LLM classification
    try:
        if key in valid_choices:
            choices_str = ", ".join(valid_choices[key])
            
            # Enhanced prompt with abbreviation awareness
            if key == "ALNSRL_DeepQ":
                prompt = f"""The user answered: "{user_message}".
What is the value for parameter: {key}? 
Valid options are: {choices_str}
Note: "1" means Deep Q-Learning (DQL), "0" means regular Q-Learning
Common abbreviations: DQL/Deep Q = 1, Q-Learning/QL = 0
Only return one of these exact valid values: {choices_str}. If unclear, return "UNCLEAR"."""
            else:
                prompt = f"""The user answered: "{user_message}".
What is the value for parameter: {key}? 
Valid options are: {choices_str}
Only return one of these exact valid values. If unclear, return "UNCLEAR"."""
        else:
            prompt = f"""The user answered: "{user_message}".
What is the numeric value for parameter: {key}? 
Only return a single number. If unclear, return "UNCLEAR"."""
            
        response = llm([HumanMessage(content=prompt)])
        val = response.content.strip()
        if key in valid_choices and val in valid_choices[key]:
            return val
        if key not in valid_choices and re.fullmatch(r"\d+", val):
            return val
        return "UNKNOWN"
    except Exception:
        return "UNKNOWN"
