import streamlit as st
import time
import os
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Import your helper functions (UPDATED)
from helpers.processor import process_message_impl
from helpers.extractor import extract_parameters_impl
from helpers.classifier import classify_single_input_impl
from helpers.rules import determine_missing_impl, question_for_param_impl
from main import run_model_with_parameters

# Load environment variables
load_dotenv()

# Initialize Langchain Azure OpenAI
@st.cache_resource
def init_langchain_llm():
    return AzureChatOpenAI(
        openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        temperature=0.9
    )

# Create wrapper classes that your helper functions expect
class LLMService:
    def __init__(self):
        self.llm = init_langchain_llm()

class ParamRules:
    # <-- add this block
    default_values = {
        "Action": "Solve",
        "Instance": "3_10_3_15_3_1_CRP",
        "Model": "Average",
        "Solver": "MIP",
        "NrScenario": "10",
        "PHAObj": "Q",
        "PHAPenalty": "S",
        "ALNSRL": "0",
        "ALNSRL_DeepQ": "0",
        "ScenarioGeneration": "RQMC",
        "ClusteringMethod": "NoC",
        "ScenarioSeed": "-1",
        "nrevaluation": "5"
    }

    def confirm_region(self, region_name):
        return f"✅ **Great!** The data for **{region_name}** (…)"
    
class ModelRunner:
    def __init__(self):
        self.default_values = {
            "Action": "Solve",
            "Instance": "3_10_3_15_3_1_CRP",
            "Model": "Average",
            "Solver": "MIP",
            "NrScenario": "10",
            "PHAObj": "Q",
            "PHAPenalty": "S",
            "ALNSRL": "0",
            "ALNSRL_DeepQ": "0",
            "ScenarioGeneration": "RQMC",
            "ClusteringMethod": "NoC",
            "ScenarioSeed": "-1",
            "nrevaluation": "5"
        }
    
    def run(self, extracted_params):
        """Run the model with collected parameters"""
        try:
            # Prepare final parameters
            final_params = {}
            for key in self.default_values.keys():
                if key in extracted_params and extracted_params[key] not in ["UNKNOWN", None]:
                    final_params[key] = extracted_params[key]
                else:
                    final_params[key] = self.default_values[key]
            
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
            
            with st.spinner("Running model... This may take a few minutes."):
                result = run_model_with_parameters(args)
            
            # Try to import LastFoundSolution after model execution
            try:
                from main import LastFoundSolution
                has_solution = True
            except ImportError:
                has_solution = False
            
            # Prepare the response
            response = "✅ **Model run completed successfully!**\n---\n"
            
            # Add metrics if available
            if has_solution and LastFoundSolution:
                try:
                    total_acfs = sum(LastFoundSolution.ACFEstablishment_x_wi[0])
                    total_assign = sum(
                        LastFoundSolution.LandRescueVehicle_thetaVar_wim[0][i][m]
                        for i in range(len(LastFoundSolution.LandRescueVehicle_thetaVar_wim[0]))
                        for m in range(len(LastFoundSolution.LandRescueVehicle_thetaVar_wim[0][i]))
                    )
                    cost = LastFoundSolution.TotalCost
                    
                    response += f"""**📊 Solution Metrics:**
- **Total Resources Assigned to ACFs:** {total_assign}
- **Total Established Temporary Hospitals (ACFs):** {total_acfs}
- **Total Cost:** ${cost:,.0f}

---
"""
                except Exception:
                    response += "**📊 Solution Metrics:** Available in the output directory.\n---\n"
            
            # Handle images
            instance_name = args["Instance"]
            model_name = args["Model"]
            solver_name = args["Solver"]
            
            image_path_all = os.path.join("UI", "Solution_UI", f"{instance_name}_{model_name}_{solver_name}_all_facilities.png")
            image_path_open = os.path.join("UI", "Solution_UI", f"{instance_name}_{model_name}_{solver_name}_open_acfs_only.png")
            
            # Display images if they exist
            if os.path.exists(image_path_all) or os.path.exists(image_path_open):
                col1, col2 = st.columns(2)
                
                with col1:
                    if os.path.exists(image_path_all):
                        st.image(image_path_all, caption="All Facilities", width=400)
                
                with col2:
                    if os.path.exists(image_path_open):
                        st.image(image_path_open, caption="Established ACFs Only", width=400)
            
            # Generate professional explanation
            explanation = self.generate_solution_explanation(args["Model"])
            response += explanation
            
            response += "\nFeel free to ask me about another disaster planning scenario!"
            
            return response
            
        except Exception as e:
            return f"❌ Error occurred while running the model:\n{str(e)}\n\nPlease try again or contact support."
    
    def generate_solution_explanation(self, model_type):
        """Generate professional explanation about ACF advantages"""
        base_explanation = """
**🏥 Strategic Benefits of Alternative Care Facilities (ACFs):**

The established ACFs provide several critical advantages:
- **Geographic Coverage**: Strategically positioned to cover underserved areas
- **Hospital Capacity Augmentation**: Significantly increase overall treatment capacity
- **Load Distribution**: Help distribute patient load across multiple facilities
- **Emergency Response Resilience**: Create redundancy in the healthcare system

---
"""
        
        if model_type == "Average":
            model_recommendation = """
**💡 Professional Recommendation:**

While your current **deterministic model** provides an excellent baseline solution, I strongly recommend also running a **stochastic analysis** for comprehensive disaster preparedness. Here's why:

- **Uncertainty Management**: The number of casualties and actual treatment capacities of hospitals are inherently uncertain during real disasters.

- **Robust Planning**: A stochastic model accounts for multiple scenarios, providing solutions that perform well across various possible outcomes.

- **Risk Mitigation**: Better safe than sorry—stochastic optimization helps identify potential vulnerabilities in your current plan.

- **Decision Confidence**: Having both deterministic and stochastic results gives you greater confidence in your emergency response strategy.

Would you like me to run a stochastic analysis with multiple scenarios to complement your current solution?

"""
        else:
            model_recommendation = """
**✅ Robust Stochastic Analysis:**

Your **stochastic model** provides a comprehensive solution that:

- **Accounts for Uncertainty**: Considers multiple scenarios of casualty numbers and hospital capacities.

- **Optimizes for Robustness**: The solution performs well across various possible disaster outcomes.

- **Provides Confidence**: You can trust this plan to handle the inherent uncertainties in emergency situations.

- **Strategic Flexibility**: The ACF network is designed to adapt to changing conditions during the disaster response.

"""
        
        return base_explanation + model_recommendation

# Initialize session state
def init_session_state():
    if "conversation_state" not in st.session_state:
        st.session_state.conversation_state = {
            "initial_query": None,
            "extracted_params": {},
            "missing_params": [],
            "current_missing_index": 0,
            "ready_to_run": False,
            "final_params": {}
        }
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "llm_service" not in st.session_state:
        st.session_state.llm_service = LLMService()
    
    if "param_rules" not in st.session_state:
        st.session_state.param_rules = ParamRules()
    
    if "model_runner" not in st.session_state:
        st.session_state.model_runner = ModelRunner()

def typing_effect(text, container):
    """Create typing effect in Streamlit"""
    placeholder = container.empty()
    current_text = ""
    
    for char in text:
        current_text += char
        placeholder.markdown(current_text)
        time.sleep(0.02)
    
    return current_text

def main():
    st.set_page_config(
        page_title="Hospital Evacuation Planning Assistant",
        page_icon="🏥",
        layout="wide"
    )

    st.markdown(
        """
        <style>
        .stApp {
            background-color: #d3d3d3;
        }
        html, body, [class*="st-"] {
            color: black;
        }
        .stChatMessage {
            background-color: rgba(0, 0, 0, 0.05);
            padding: 10px;
            border-radius: 10px;
        }
        .stSidebar {
            background-color: #c0c0c0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Insert logo top-right
    col1, col2 = st.columns([7, 1])
    with col2:
        if os.path.exists("Logo-CIRRELT.png"):
            st.image("Logo-CIRRELT.png", width=80)

    # Initialize session state
    init_session_state()

    # App title and description
    st.title("🏥 Hospital Evacuation Planning Assistant")
    st.markdown("I help you plan optimal locations for Alternative Care Facilities (temporary hospitals) during disasters using stochastic optimization, scenario-based planning, and reinforcement learning.")
    st.markdown("*Developed by Farghadani-Chaharsooghi, P., Hashemi Doulabi, H., Rei, W., & Gendreau, M. (2025)*")
    
    # Sidebar for parameters display
    with st.sidebar:
        st.header("Current Parameters")
        if st.session_state.conversation_state["extracted_params"]:
            for key, value in st.session_state.conversation_state["extracted_params"].items():
                if value not in ["UNKNOWN", None]:
                    st.write(f"**{key}:** {value}")
        else:
            st.write("No parameters extracted yet.")
        
        # Debug info
        debug_mode = st.checkbox("Show Debug Info")
        if debug_mode:
            st.write("**Conversation State:**")
            st.json(st.session_state.conversation_state)
    
    # Chat interface
    st.header("Chat")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Describe your disaster situation and planning needs..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process user message using the helper function
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Use the helper function with proper parameters
            response = process_message_impl(
                prompt, 
                st.session_state.conversation_state,
                st.session_state.llm_service,
                st.session_state.param_rules,
                st.session_state.model_runner
            )
            
            # Create typing effect
            typing_effect(response, message_placeholder)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.conversation_state = {
            "initial_query": None,
            "extracted_params": {},
            "missing_params": [],
            "current_missing_index": 0,
            "ready_to_run": False,
            "final_params": {}
        }
        st.rerun()

if __name__ == "__main__":
    main()