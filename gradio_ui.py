# Enhanced version of your existing code with better result display

import gradio as gr
from main import run_model_with_parameters
from dotenv import load_dotenv
import os
from openai import AzureOpenAI
import json
import time
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import shutil

# Load .env values
load_dotenv()

client = AzureOpenAI(api_key=os.getenv("AZURE_OPENAI_KEY"),
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"))

deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# Default values for parameters
default_values = {
    "Action": "Solve",
    "Instance": "3_10_3_15_3_1_CRP",
    "Model": "Average",
    "Solver": "MIP",
    "NrScenario": "1",
    "PHAObj": "Q",
    "PHAPenalty": "S",
    "ALNSRL": "0",  # Default to NO RL
    "ALNSRL_DeepQ": "0",
    "ScenarioGeneration": "RQMC",
    "ClusteringMethod": "NoC",
    "ScenarioSeed": "-1",
    "nrevaluation": "5"
}

# User inputs will be collected here
user_inputs = {
    "Action": "Solve",
    "Instance": "3_10_3_15_3_1_CRP",
    "Model": None,
    "Solver": None,
    "NrScenario": None,
    "PHAObj": "Q",
    "PHAPenalty": "S",
    "ALNSRL": None,
    "ALNSRL_DeepQ": None,
    "ScenarioGeneration": "RQMC",
    "ClusteringMethod": None,
    "ScenarioSeed": "-1",
    "nrevaluation": None
}

# Conversation state
conversation_state = {
    "initial_query": None,
    "extracted_params": {},
    "missing_params": [],
    "current_missing_index": 0,
    "ready_to_run": False
}

def typing_effect_generator(text, delay=0.03):
    """Generator that yields text with typing effect"""
    current_text = ""
    for char in text:
        current_text += char
        #time.sleep(delay)
        yield current_text

def copy_images_to_gradio_static(instance_name):
    """Copy generated images to Gradio's static directory for web access"""
    
    output_dir = r"C:\PhD\Thesis\Papers\3rd\Code\RL\UI\Solution_UI"
    
    # Create a static directory in the current working directory
    static_dir = os.path.join(os.getcwd(), "static")
    os.makedirs(static_dir, exist_ok=True)
    
    copied_images = []
    
    # Check and copy images
    all_facilities_img = os.path.join(output_dir, f"{instance_name}_all_facilities.png")
    open_acfs_img = os.path.join(output_dir, f"{instance_name}_open_acfs_only.png")
    
    if os.path.exists(all_facilities_img):
        dest_path = os.path.join(static_dir, f"{instance_name}_all_facilities.png")
        shutil.copy2(all_facilities_img, dest_path)
        copied_images.append(dest_path)
        print(f"Copied {all_facilities_img} to {dest_path}")
    
    if os.path.exists(open_acfs_img):
        dest_path = os.path.join(static_dir, f"{instance_name}_open_acfs_only.png")
        shutil.copy2(open_acfs_img, dest_path)
        copied_images.append(dest_path)
        print(f"Copied {open_acfs_img} to {dest_path}")
    
    return copied_images

def determine_missing_parameters(extracted_params):
    """Determine which parameters still need to be collected"""
    
    missing = []
    
    print(f"DEBUG: Determining missing params for: {extracted_params}")
    
    # Check Model
    model_value = extracted_params.get("Model")
    if not model_value or model_value == "UNKNOWN":
        missing.append("Model")
        print("DEBUG: Model is missing")
    
    # Check Solver
    solver_value = extracted_params.get("Solver")
    if not solver_value or solver_value == "UNKNOWN":
        missing.append("Solver")
        print("DEBUG: Solver is missing")
    
    # If Model is 2Stage, check NrScenario
    if extracted_params.get("Model") == "2Stage":
        scenario_value = extracted_params.get("NrScenario")
        if not scenario_value or scenario_value == "UNKNOWN":
            missing.append("NrScenario")
            print("DEBUG: NrScenario is missing")
    
    # If Solver is ALNS, check RL preference
    if extracted_params.get("Solver") == "ALNS":
        alnsrl_value = extracted_params.get("ALNSRL")
        print(f"DEBUG: ALNS detected, ALNSRL value: '{alnsrl_value}'")
        if not alnsrl_value or alnsrl_value == "UNKNOWN":
            missing.append("ALNSRL")
            print("DEBUG: ALNSRL is missing - will ask about RL preference")
        
        # If ALNSRL is 1 (Yes to RL), check ALNSRL_DeepQ
        alnsrl_str = str(extracted_params.get("ALNSRL", "")).strip()
        print(f"DEBUG: ALNSRL value for DeepQ check: '{alnsrl_str}'")
        if alnsrl_str == "1":
            deepq_value = extracted_params.get("ALNSRL_DeepQ")
            print(f"DEBUG: ALNSRL is 1, checking ALNSRL_DeepQ: '{deepq_value}'")
            if not deepq_value or deepq_value == "UNKNOWN":
                missing.append("ALNSRL_DeepQ")
                print("DEBUG: ALNSRL_DeepQ is missing")
    
    # Only check ClusteringMethod if Model is 2Stage
    if extracted_params.get("Model") == "2Stage":
        clustering_value = extracted_params.get("ClusteringMethod")
        if not clustering_value or clustering_value == "UNKNOWN":
            missing.append("ClusteringMethod")
            print("DEBUG: ClusteringMethod is missing")
    
    print(f"DEBUG: Missing parameters: {missing}")
    return missing

def get_question_for_parameter(param):
    """Get the appropriate question for a missing parameter"""
    
    questions = {
        "Model": "Which model do you prefer: `Average` (Only an average scenario for the number of injured people) or `2Stage` (stochastic number of casualties)?",
        "Solver": "Which solver should be used: `MIP` (exact), `ALNS` (heuristic)?",
        "NrScenario": "How many scenarios should be considered for the stochastic model? (e.g., 50, 100, etc.)",
        "ALNSRL": "Do you want to use Reinforcement Learning in the heuristic method? (0 = No, 1 = Yes)",
        "ALNSRL_DeepQ": "Which type of RL? Deep Q-Learning (1) or regular Q-Learning (0)?",
        "ClusteringMethod": "Choose the clustering method: `NoC` (No Clustering), `KM` (K-Means), `KMPP` (K-Means++), `SOM` (Self-Organizing Map), or `DB` (Decision-Based)?"
    }
    
    return questions.get(param, f"Please provide value for {param}")

def classify_single_input(user_message, key):
    """Classify a single parameter input"""
    
    valid_choices = {
        "Model": ["Average", "2Stage"],
        "Solver": ["MIP", "ALNS"],
        "ALNSRL": ["0", "1"],
        "ALNSRL_DeepQ": ["0", "1"],
        "ClusteringMethod": ["NoC", "KM", "KMPP", "SOM", "DB"]
    }
    
    try:
        if key in valid_choices:
            choices_str = ", ".join(valid_choices[key])
            prompt = f"""The user answered: "{user_message}".
What is the value for parameter: {key}? 
Valid options are: {choices_str}
Only return one of these exact valid values. If unclear, return "UNCLEAR"."""
        else:
            prompt = f"""The user answered: "{user_message}".
What is the numeric value for parameter: {key}? 
Only return a single number. If unclear, return "UNCLEAR"."""
        
        response = client.chat.completions.create(model=deployment,
                                                    messages=[{"role": "user", "content": prompt}],
                                                    temperature=0.9)
        
        classified_value = response.choices[0].message.content.strip()
        
        if classified_value == "UNCLEAR" or (key in valid_choices and classified_value not in valid_choices[key]):
            return "UNKNOWN"
        else:
            return classified_value
            
    except Exception as e:
        print(f"Error classifying input for {key}, using default: {default_values[key]}. Error: {str(e)}")
        if key == "ALNSRL":
            return "0"
        return default_values[key]

def extract_parameters_from_text(user_message):
    """Extract disaster planning parameters from natural language using LLM"""
    
    prompt = f"""You are a disaster planning assistant. Analyze the user's message and extract relevant parameters.

User message: "{user_message}"

Extract the following parameters if mentioned:
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

REINFORCEMENT LEARNING RULES:
7. ALNSRL should ONLY be set to "1" if user uses these EXACT phrases:
    - "reinforcement learning"
    - "RL" 
    - "with RL"
    - "use RL"
    - "machine learning"
    - "AI learning"
    - "learning algorithm"
8. ALNSRL should be set to "0" if user mentions no RL or standard ALNS
9. If user mentions ONLY "ALNS" without RL keywords → ALNSRL = "UNKNOWN"
10. If user mentions "Deep Q-Learning" or "DQN" → ALNSRL_DeepQ = "1"
11. If user mentions "Q-Learning" (but not Deep Q) → ALNSRL_DeepQ = "0"
12. If user mentions numbers like "50 scenarios", "100 scenarios" → extract the number for NrScenario
13. If ALNSRL_DeepQ is not explicitly mentioned → set ALNSRL_DeepQ to "UNKNOWN"

Return ONLY a JSON object with the extracted parameters. Use "UNKNOWN" for parameters not mentioned or unclear.
Example: {{"Model": "2Stage", "Solver": "ALNS", "NrScenario": "UNKNOWN", "ALNSRL": "UNKNOWN", "ALNSRL_DeepQ": "UNKNOWN", "ClusteringMethod": "UNKNOWN"}}"""

    try:
        response = client.chat.completions.create(model=deployment,
                                                    messages=[{"role": "user", "content": prompt}],
                                                    temperature=0.9) 
        
        response_text = response.choices[0].message.content.strip()
        print(f"DEBUG: User message: '{user_message}'")
        print(f"DEBUG: LLM response: {response_text}")
        
        # Try to extract JSON from response
        try:
            extracted = json.loads(response_text)
            print(f"DEBUG: Extracted params: {extracted}")
            return extracted
        except json.JSONDecodeError as e:
            print(f"DEBUG: JSON decode error: {e}")
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                extracted = json.loads(json_match.group())
                print(f"DEBUG: Extracted params from regex: {extracted}")
                return extracted
            else:
                print("DEBUG: No JSON found in response")
                return {}
        
    except Exception as e:
        print(f"Error extracting parameters: {str(e)}")
        return {}

def chat_fn(user_message, history):
    """Main chat function with improved parameter handling"""
    global conversation_state
    
    # Initial greeting
    if not conversation_state["initial_query"]:
        if len(history) == 0:
            greeting = """Hi there,

I'm truly sorry that you're facing such a difficult situation. Please know that you are not alone—I'm here to support you through this.

My goal is to help you evacuate patients safely, minimize risk, and optimize your resources as quickly and efficiently as possible. Together, we can make informed decisions that save lives.

Depending on the size and complexity of the disaster, I can assist you with:
- A **deterministic** model using exact MIP methods for smaller events
- A **stochastic** model using advanced heuristics (ALNS with or without reinforcement learning) for large-scale crises

To get started, I'll need a few details:
- Which region is affected?
- Do you prefer an exact or stochastic model?
- If stochastic: how many scenarios should we consider?
- Do you have a preferred solver (e.g., Exact MIP or ALNS)?
- If ALNS: would you like to include reinforcement learning?
- Which scenario clustering method would you prefer (if any)?

Take a deep breath and let me know your situation. I'll help you create a solid plan from here."""
            
            for partial_text in typing_effect_generator(greeting):
                yield partial_text
            return
        
        # Process initial query
        conversation_state["initial_query"] = user_message
        extracted = extract_parameters_from_text(user_message)
        conversation_state["extracted_params"] = extracted
        
        # Apply business rules
        if extracted.get("Model") == "Average":
            extracted["Solver"] = "MIP"
            extracted["ALNSRL"] = "0"
            extracted["ALNSRL_DeepQ"] = "0"
            print("DEBUG: Applied Average model business rules")
        
        # Determine missing parameters
        missing = determine_missing_parameters(extracted)
        conversation_state["missing_params"] = missing
        conversation_state["current_missing_index"] = 0
        
        if not missing:
            conversation_state["ready_to_run"] = True
            response = prepare_final_summary(extracted)
        else:
            # Ask questions one by one
            response = f"Thanks — I've reviewed your request and extracted most of the needed info.\n\nJust one more question: {get_question_for_parameter(missing[0])}"
        
        for partial_text in typing_effect_generator(response):
            yield partial_text
        return

    if conversation_state["ready_to_run"]:
        if "yes" in user_message.lower() or "run" in user_message.lower() or "proceed" in user_message.lower() or "solve" in user_message.lower():
            response = enhanced_run_model_with_collected_params()  # Use enhanced version
        else:
            response = "Please type 'run' to proceed with the model execution, or describe any changes you'd like to make."
        
        for partial_text in typing_effect_generator(response):
            yield partial_text
        return

    # Check for updates in the message before continuing
    updated_keys = []
    for key in default_values.keys():
        if key in ["NrScenario"]:
            # Special case for numeric values
            import re
            if key in conversation_state["extracted_params"] and re.search(rf"{key}|scenario[s]?", user_message, re.IGNORECASE):
                nums = re.findall(r'\d+', user_message)
                if nums:
                    conversation_state["extracted_params"][key] = nums[0]
                    updated_keys.append(key)
        elif key in ["Model", "Solver", "ALNSRL", "ClusteringMethod"]:
            updated_value = classify_single_input(user_message, key)
            if updated_value != "UNKNOWN":
                conversation_state["extracted_params"][key] = updated_value
                updated_keys.append(key)

        # Only classify ALNSRL_DeepQ if RL is mentioned AND DeepQ/Q is explicitly in message
        elif key == "ALNSRL_DeepQ":
            if any(keyword in user_message.lower() for keyword in ["deep q", "dqn", "q-learning"]):
                updated_value = classify_single_input(user_message, key)
                if updated_value != "UNKNOWN":
                    conversation_state["extracted_params"][key] = updated_value
                    updated_keys.append(key)

    if updated_keys:
        # Update missing params again
        conversation_state["missing_params"] = determine_missing_parameters(conversation_state["extracted_params"])
        conversation_state["current_missing_index"] = 0  # Start asking again from first missing param

        response = f"✅ Got it! I updated the following: {', '.join(updated_keys)}.\n\n"
        if not conversation_state["missing_params"]:
            conversation_state["ready_to_run"] = True
            response += prepare_final_summary(conversation_state["extracted_params"])
        else:
            next_param = conversation_state["missing_params"][0]
            response += get_question_for_parameter(next_param)

        for partial_text in typing_effect_generator(response):
            yield partial_text
        return
    
    # Handle missing parameter collection
    if conversation_state["missing_params"] and conversation_state["current_missing_index"] < len(conversation_state["missing_params"]):
        current_param = conversation_state["missing_params"][conversation_state["current_missing_index"]]
        
        # Handle multi-parameter responses better
        if current_param == "NrScenario":
            # Extract number from response
            import re
            numbers = re.findall(r'\d+', user_message)
            if numbers:
                classified_value = numbers[0]  # Take first number found
            else:
                classified_value = classify_single_input(user_message, current_param)
        else:
            classified_value = classify_single_input(user_message, current_param)
        
        conversation_state["extracted_params"][current_param] = classified_value
        print(f"DEBUG: Set {current_param} = {classified_value}")
        
        # Apply business rules after each input
        if current_param == "Model" and classified_value == "Average":
            conversation_state["extracted_params"]["Solver"] = "MIP"
            conversation_state["extracted_params"]["ALNSRL"] = "0"
            conversation_state["extracted_params"]["ALNSRL_DeepQ"] = "0"
            conversation_state["missing_params"] = [p for p in conversation_state["missing_params"] if p not in ["ALNSRL", "ALNSRL_DeepQ"]]
        
        # If we just set ALNSRL to "1", we need to add ALNSRL_DeepQ to missing params
        if current_param == "ALNSRL" and classified_value == "1":
            if "ALNSRL_DeepQ" not in conversation_state["missing_params"]:
                conversation_state["missing_params"].insert(conversation_state["current_missing_index"] + 1, "ALNSRL_DeepQ")
                print("DEBUG: Added ALNSRL_DeepQ to missing params")
        
        # If we just set ALNSRL to "0", remove ALNSRL_DeepQ from missing params
        if current_param == "ALNSRL" and classified_value == "0":
            if "ALNSRL_DeepQ" in conversation_state["missing_params"]:
                conversation_state["missing_params"].remove("ALNSRL_DeepQ")
                print("DEBUG: Removed ALNSRL_DeepQ from missing params (RL disabled)")
            conversation_state["extracted_params"]["ALNSRL_DeepQ"] = "0"
        
        conversation_state["current_missing_index"] += 1
        
        # Check if we have more missing parameters
        if conversation_state["current_missing_index"] < len(conversation_state["missing_params"]):
            next_param = conversation_state["missing_params"][conversation_state["current_missing_index"]]
            response = get_question_for_parameter(next_param)
        else:
            # All parameters collected
            conversation_state["ready_to_run"] = True
            response = prepare_final_summary(conversation_state["extracted_params"])
        
        for partial_text in typing_effect_generator(response):
            yield partial_text
        return
    
    response = "I'm not sure how to help with that. Could you please clarify your request?"
    for partial_text in typing_effect_generator(response):
        yield partial_text

def prepare_final_summary(extracted_params):
    """Prepare final summary before running the model"""
    
    final_params = {}
    for key in default_values.keys():
        if key in extracted_params and extracted_params[key] not in ["UNKNOWN", None]:
            final_params[key] = extracted_params[key]
        else:
            final_params[key] = default_values[key]
    
    conversation_state["final_params"] = final_params
    
    params_summary = "\n".join([f"**{k}:** {v}" for k, v in final_params.items()])
    
    return f"""Perfect! I have all the information needed. Here's what I'll run for you:
{params_summary}
Type 'yes' or 'run' to proceed with the model execution!
If you need to change anything, just let me know."""


def enhanced_run_model_with_collected_params():
    """Enhanced version that returns detailed results and handles images properly"""
    
    try:
        final_params = conversation_state["final_params"]
        
        args = {
            "Action": final_params["Action"],
            "Instance": final_params["Instance"],
            "Model": final_params["Model"],
            "Solver": final_params["Solver"],
            "NrScenario": final_params["NrScenario"],
            "PHAObj": final_params["PHAObj"],
            "PHAPenalty": final_params["PHAPenalty"],
            "ALNSRL": int(final_params["ALNSRL"]),
            "ALNSRL_DeepQ": int(final_params["ALNSRL_DeepQ"]),
            "ScenarioGeneration": final_params["ScenarioGeneration"],
            "ClusteringMethod": final_params["ClusteringMethod"],
            "ScenarioSeed": int(final_params["ScenarioSeed"]),
            "nrevaluation": int(final_params["nrevaluation"])
        }
        
        # Run the model
        result = run_model_with_parameters(args)
        
        # Copy images to accessible location
        instance_name = args["Instance"]
        copied_images = copy_images_to_gradio_static(instance_name)
        conversation_state["generated_images"] = copied_images
        
        # Get solution details
        solution_details = get_solution_details(args, copied_images)
        
        # Reset conversation state for next query
        conversation_state["initial_query"] = None
        conversation_state["extracted_params"] = {}
        conversation_state["missing_params"] = []
        conversation_state["current_missing_index"] = 0
        conversation_state["ready_to_run"] = False
        
        return solution_details
        
    except Exception as e:
        return f"❌ Error occurred while running the model:\n{str(e)}\n\nPlease try again or contact support."

def get_solution_details(args, copied_images):
    """Extract and format solution details with proper image references"""
    
    result_text = f"""✅ **Model run completed successfully!**

**Configuration Used:**
- **Instance:** {args["Instance"]}
- **Model:** {args["Model"]}
- **Solver:** {args["Solver"]}
- **Number of Scenarios:** {args["NrScenario"]}
- **Reinforcement Learning:** {'Yes' if args["ALNSRL"] else 'No'}
- **Clustering Method:** {args["ClusteringMethod"]}

**Results:**"""
    
    # Try to get solution metrics
    try:
        solution_metrics = get_solution_metrics(args)
        result_text += f"\n{solution_metrics}"
    except Exception as e:
        result_text += f"\n- Solution computed and saved to output directory"
    
    # Add information about generated images
    if copied_images:
        result_text += f"\n\n📊 **Visualizations Generated:**"
        for img_path in copied_images:
            filename = os.path.basename(img_path)
            result_text += f"\n- {filename}"
        
        result_text += f"\n\n💡 **To view the images:** The visualization files have been generated. You can find them in the static folder or check the original Solution_UI directory."
    
    result_text += f"\n\n📁 **Output Location:** `C:\\PhD\\Thesis\\Papers\\3rd\\Code\\RL\\UI\\Solution_UI`"
    result_text += f"\n\n---\n\nFeel free to ask me about another disaster planning scenario!"
    
    return result_text

def get_solution_metrics(args):
    """Extract solution metrics from the solved instance"""
    
    try:
        from main import LastFoundSolution
        
        if LastFoundSolution:
            # Calculate metrics safely
            total_acfs_opened = sum(LastFoundSolution.ACFEstablishment_x_wi[0][i]
                                   for i in range(len(LastFoundSolution.ACFEstablishment_x_wi[0])))
            
            total_land_rescue_assignments = sum(LastFoundSolution.LandRescueVehicle_thetaVar_wim[0][i][m]
                                               for i in range(len(LastFoundSolution.LandRescueVehicle_thetaVar_wim[0]))
                                               for m in range(len(LastFoundSolution.LandRescueVehicle_thetaVar_wim[0][i])))
            
            metrics = f"""**Solution Metrics:**
- **Total Resources Assigned to ACFs:** {total_land_rescue_assignments}
- **Total Open ACFs:** {total_acfs_opened}
- **Total Cost:** ${LastFoundSolution.TotalCost:,.0f}
- More detailed metrics can be found in the output directory."""
            return metrics
        else:
            return "- Solution computed successfully"
            
    except Exception as e:
        return f"- Solution computed (metrics unavailable: {str(e)})"

# Create the interface
iface = gr.ChatInterface(
    fn=chat_fn,
    chatbot=gr.Chatbot(label="Disaster Planning Assistant"),
    textbox=gr.Textbox(placeholder="Describe your disaster situation and planning needs...", label="Your Message"),
    title="🏥 Hospital Evacuation Planning Assistant - CIRRELT",
    description="I help you plan optimal locations for Alternative Care Facilities (temporary hospitals) during disasters using stochastic optimization, scenario-based planning, and reinforcement learning.<br><span style='font-size: 0.9em; color: #400;'>Developed by Farghadani-Chaharsooghi, P., Hashemi Doulabi, H., Rei, W., & Gendreau, M. (2025)</span>",
    theme="soft"
)

if __name__ == "__main__":
    iface.launch(inbrowser=True, share=True)