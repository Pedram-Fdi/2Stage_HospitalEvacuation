import time
import re
from helpers.classifier import classify_single_input_impl
from helpers.extractor import extract_parameters_impl
from helpers.rules import determine_missing_impl, question_for_param_impl
from helpers.runner import format_solution_impl
from helpers.rag_system import RAGSystem 
from main import run_model_with_parameters
import streamlit as st


def process_message_impl(user_message, state, llm_service, param_rules, model_runner) -> str:
    """Core conversational logic for handling user messages and state transitions."""
    
    # Initialize RAG system if not already done
    if not hasattr(llm_service, 'rag_system'):
        llm_service.rag_system = RAGSystem(llm_service.llm)
    
    # Check if this is an informational query FIRST (before other logic)
    if llm_service.rag_system.is_informational_query(user_message):
        return llm_service.rag_system.answer_informational_query(user_message)    
    
    greeting_keywords = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"]
    user_lower = user_message.lower().strip()
    
    # Greeting
    if not state.get('initial_query') and any(kw in user_lower for kw in greeting_keywords) and len(user_message.split()) <= 3:
        return (
        """Hi there,
        
I'm sorry you're facing this challenging situation. I'm here to help you evacuate patients, minimize risk, and allocate resources effectively.

To begin, I need a few details:
- Which region or city is affected?
- Do you prefer a deterministic or stochastic model?
- If stochastic: how many scenarios?
- Preferred solver (e.g., Exact MIP or ALNS)?
- Should I include reinforcement learning?
- Any scenario clustering method you'd like to use?

Let me know your situation, and I’ll guide you from here."""
        )

    # Initial query
    if not state.get('initial_query'):
        state['initial_query'] = user_message
        extracted = extract_parameters_impl(llm_service.llm, user_message)
        state['extracted_params'] = extracted

        if extracted.get('Model') == 'Average':
            extracted['Solver'] = 'MIP'
            extracted['ALNSRL'] = '0'
            extracted['ALNSRL_DeepQ'] = '0'

        missing = determine_missing_impl(extracted)
        state['missing_params'] = missing
        state['current_missing_index'] = 0

        if not missing:
            state['ready_to_run'] = True
            return prepare_final_summary(extracted, param_rules)
        return f"Just one more question: {question_for_param_impl(missing[0])}"

    # Ready to run confirmation
    if state.get('ready_to_run'):
        if any(kw in user_lower for kw in ['yes','run','proceed','solve']):
            # 1) run the model
            response = model_runner.run(state['extracted_params'])
            
            # 2) reset all of the conversation_state fields
            state.update({
                'initial_query': None,
                'extracted_params': {},
                'missing_params': [],
                'current_missing_index': 0,
                'ready_to_run': False,
                'final_params': {}
            })
            
            # 3) return the response you got from the runner
            return response
        
        return "Please type 'run' to proceed or describe changes."

    # Detect parameter changes
    def detect_parameter_changes(message: str):
        changes = {}
        is_change = any(kw in message.lower() for kw in ['change','modify','update','instead'])
        # Region
        if 'region' in message.lower() or is_change:
            for pat in [r'region\s+(?:is\s+)?([\w ]+)', r'in\s+([\w ]+)']:
                m = re.search(pat, message.lower())
                if m:
                    val = m.group(1).title()
                    changes['Region'] = val
                    break
        # Model
        if 'average' in message.lower(): changes['Model']='Average'
        if any(term in message.lower() for term in ['stochastic','2stage']): changes['Model']='2Stage'
        # Solver
        if 'mip' in message.lower(): changes['Solver']='MIP'
        if 'alns' in message.lower(): changes['Solver']='ALNS'
        # Scenarios
        nums = re.findall(r'\d+', message)
        if nums: changes['NrScenario']=nums[0]
        # RL
        if 'rl' in message.lower(): changes['ALNSRL']='1'
        if 'no rl' in message.lower(): changes['ALNSRL']='0'
        # DeepQ
        if 'deep q' in message.lower(): changes['ALNSRL_DeepQ']='1'
        if 'q-learning' in message.lower(): changes['ALNSRL_DeepQ']='0'
        # Clustering
        if 'noc' in message.lower(): changes['ClusteringMethod']='NoC'
        if 'kmpp' in message.lower(): changes['ClusteringMethod']='KMPP'
        if 'km ' in message.lower(): changes['ClusteringMethod']='KM'
        return changes, is_change

    changes, is_change = detect_parameter_changes(user_message)
    if changes:
        for k,v in changes.items(): state['extracted_params'][k]=v
        # Business rules
        if state['extracted_params'].get('Model')=='Average':
            state['extracted_params'].update({'Solver':'MIP','ALNSRL':'0','ALNSRL_DeepQ':'0'})
        missing = determine_missing_impl(state['extracted_params'])
        state['missing_params']=missing
        state['current_missing_index']=0
        if not missing:
            state['ready_to_run']=True
            return prepare_final_summary(state['extracted_params'], param_rules)
        return f"Changes applied. Next: {question_for_param_impl(missing[0])}"

    # Collect missing parameters
    if state.get('missing_params') and state['current_missing_index']<len(state['missing_params']):
        param = state['missing_params'][state['current_missing_index']]
        if param=='NrScenario':
            nums = re.findall(r'\d+', user_message)
            val = nums[0] if nums else classify_single_input_impl(llm_service.llm, user_message, param)
        else:
            val = classify_single_input_impl(llm_service.llm, user_message, param)
        state['extracted_params'][param]=val
        resp = ''
        if param=='Region': resp+=param_rules.confirm_region(val)+'\n\n'
        # Apply rules
        if param=='Model' and val=='Average':
            state['extracted_params'].update({'Solver':'MIP','ALNSRL':'0','ALNSRL_DeepQ':'0'})
        if param=='ALNSRL' and val=='1': state['missing_params'].insert(state['current_missing_index']+1,'ALNSRL_DeepQ')
        if param=='ALNSRL' and val=='0': state['extracted_params']['ALNSRL_DeepQ']='0'
        state['current_missing_index']+=1
        if state['current_missing_index']<len(state['missing_params']):
            return resp+question_for_param_impl(state['missing_params'][state['current_missing_index']])
        state['ready_to_run']=True
        return prepare_final_summary(state['extracted_params'], param_rules)

    return "I'm not sure how to help with that. Could you clarify?"


def prepare_final_summary(extracted_params, param_rules):
    """Prepare final summary before running the model"""
    final_params = {}
    for key, default in param_rules.default_values.items():
        if key in extracted_params and extracted_params[key] not in ["UNKNOWN", None]:
            final_params[key] = extracted_params[key]
        else:
            final_params[key] = default

    st.session_state.conversation_state["final_params"] = final_params

    params_summary = "\n".join([f"**{k}:** {v}  " for k, v in final_params.items()])

    return (
        "Perfect! I have all the information needed. Here's what I'll run for you:\n\n"
        f"{params_summary}\n\n"
        "Type 'yes' or 'run' to proceed with the model execution!\n"
        "If you need to change anything, just let me know."
    )
