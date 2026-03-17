import json
import re
from langchain.schema import HumanMessage


def extract_parameters_impl(llm, user_message: str) -> dict:
    """Extract disaster planning parameters from natural language using Langchain"""
    
    system_prompt = """You are a disaster planning assistant. Analyze the user's message and extract relevant parameters.

Extract the following parameters if mentioned:
- Region: Name of the region, city, or area affected by the disaster (e.g., "Montreal", "Toronto", "New York", etc.)
- Model: "Average" (for exact/deterministic model) or "2Stage" (for stochastic model)
- Solver: "MIP", "ALNS" (for stochastic model / Heuristic model)
- NrScenario: Number of scenarios (only relevant for 2Stage model)
- ALNSRL: "0" (No RL) or "1" (Yes RL) - ONLY if explicitly mentioned
- ALNSRL_DeepQ: "0" (Q-Learning) or "1" (Deep Q-Learning) - only relevant if ALNSRL is 1
- ClusteringMethod: "NoC" (No scenario clustering), "KM" (K-Means Scenario Clustering), "KMPP" (K-Means++ Scenario Clustering), "SOM" (Self-Organizing Maps scenario clustering), or "DB" (Decision-Based scenario clustering)

IMPORTANT RULES:
1. If user mentions "exact", "exactly", "deterministic", or "Average" → Model = "Average"
2. If user mentions "stochastic", "scenarios", "uncertainty", "2Stage" → Model = "2Stage"
3. If user mentions "heuristic", "heuristically", "greedy", "ALNS" → Solver = "ALNS"
4. If user mentions "exact solver", "MIP", "optimal" → Solver = "MIP"
5. If Model is "Average", then Solver MUST be "MIP" (no ALNS or heuristic options)
6. If Model is "2Stage", Solver can be any of: "MIP", "ALNS"
7. For Region: Extract any city, region, area, or location name mentioned (e.g., "Montreal", "Toronto", "New York", "Quebec", "California", etc.)

REINFORCEMENT LEARNING RULES (ABSOLUTELY CRITICAL):
8. ALNSRL should ONLY be set to "1" if user uses these EXACT phrases:
    - "reinforcement learning", "RL", "with RL", "use RL", "machine learning", "AI learning", "learning algorithm"
9. ALNSRL should be set to "0" if user uses these EXACT phrases:
    - "no RL", "no reinforcement learning", "without RL", "basic ALNS", "standard ALNS"
10. CRITICAL: If user mentions ONLY "ALNS" WITHOUT any RL-related words → ALNSRL = "UNKNOWN"
11. If user mentions "Deep Q-Learning" or "DQN" → ALNSRL_DeepQ = "1"
12. If user mentions "Q-Learning" (but not Deep Q) → ALNSRL_DeepQ = "0"
13. If user mentions numbers like "50 scenarios", "100 scenarios" → extract the number for NrScenario
14. DO NOT make assumptions about RL preferences!

Return ONLY a JSON object with the extracted parameters. Use "UNKNOWN" for parameters not mentioned or unclear.
Example: {"Region": "Montreal", "Model": "2Stage", "Solver": "ALNS", "NrScenario": "UNKNOWN", "ALNSRL": "UNKNOWN", "ALNSRL_DeepQ": "UNKNOWN", "ClusteringMethod": "UNKNOWN"}"""

    prompt = f"{system_prompt}\n\nUser message: '{user_message}'"
    try:
        response = llm([HumanMessage(content=prompt)])
        text = response.content.strip()
        # Try direct JSON
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Fallback to regex
            m = re.search(r"\{.*\}", text, re.DOTALL)
            data = json.loads(m.group()) if m else {}

        # Post-process ALNS without RL
        if data.get("Solver") == "ALNS":
            user_lower = user_message.lower()
            rl_keywords = ["reinforcement learning", "rl", "with rl", "use rl", "machine learning", "ai learning", "learning algorithm"]
            no_rl_keywords = ["no rl", "no reinforcement learning", "without rl", "basic alns", "standard alns"]
            
            has_rl_mention = any(keyword in user_lower for keyword in rl_keywords)
            has_no_rl_mention = any(keyword in user_lower for keyword in no_rl_keywords)
            
            if not has_rl_mention and not has_no_rl_mention:
                data["ALNSRL"] = "UNKNOWN"
                data["ALNSRL_DeepQ"] = "UNKNOWN"
        return data
    except Exception:
        return {}  # In debug mode, could log error
