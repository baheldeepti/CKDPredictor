import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import shap
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional
import json
import openai
from datetime import datetime

# Load and preprocess data
@st.cache_data
def load_data():
    """Load and preprocess the kidney disease dataset."""
    try:
        data = pd.read_csv("kidney_disease.csv")
        # Perform necessary data cleaning and preprocessing
        # Handle missing values, encode categorical variables, etc.
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

data = load_data()

# Train model
@st.cache_resource
def train_model(data):
    """Train the Random Forest classifier model."""
    try:
        X = data.drop("ckd", axis=1)
        y = data["ckd"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model, X_train, X_test, y_train, y_test
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None, None, None, None

if data is not None:
    model, X_train, X_test, y_train, y_test = train_model(data)
else:
    model, X_train, X_test, y_train, y_test = None, None, None, None, None

# Streamlit app
st.title("CKD Risk Assessment Tool")

# Initialize session state
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "explanation" not in st.session_state:
    st.session_state.explanation = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "ctx_digital_twin" not in st.session_state:
    st.session_state.ctx_digital_twin = None
if "ctx_counterfactual" not in st.session_state:
    st.session_state.ctx_counterfactual = None
if "ctx_similar" not in st.session_state:
    st.session_state.ctx_similar = None
if "ctx_explain" not in st.session_state:
    st.session_state.ctx_explain = None

# Sidebar for feature input
st.sidebar.header("Patient Information")
age = st.sidebar.slider("Age", 18, 100, 50)
blood_pressure = st.sidebar.slider("Blood Pressure", 50, 180, 120)
specific_gravity = st.sidebar.slider("Specific Gravity", 1.005, 1.025, 1.015, step=0.001)
albumin = st.sidebar.selectbox("Albumin", [0, 1, 2, 3, 4, 5])
sugar = st.sidebar.selectbox("Sugar", [0, 1, 2, 3, 4, 5])
red_blood_cells = st.sidebar.selectbox("Red Blood Cells", ["normal", "abnormal"])
pus_cell = st.sidebar.selectbox("Pus Cell", ["normal", "abnormal"])
pus_cell_clumps = st.sidebar.selectbox("Pus Cell Clumps", ["present", "notpresent"])
bacteria = st.sidebar.selectbox("Bacteria", ["present", "notpresent"])
blood_glucose_random = st.sidebar.slider("Blood Glucose Random", 20, 400, 100)
blood_urea = st.sidebar.slider("Blood Urea", 10, 200, 50)
serum_creatinine = st.sidebar.slider("Serum Creatinine", 0.4, 15.0, 1.0, step=0.1)
sodium = st.sidebar.slider("Sodium", 100, 170, 135)
potassium = st.sidebar.slider("Potassium", 2.5, 7.5, 4.0, step=0.1)
hemoglobin = st.sidebar.slider("Hemoglobin", 3.0, 18.0, 12.0, step=0.1)
packed_cell_volume = st.sidebar.slider("Packed Cell Volume", 9, 55, 40)
white_blood_cell_count = st.sidebar.slider("White Blood Cell Count", 2000, 20000, 8000)
red_blood_cell_count = st.sidebar.slider("Red Blood Cell Count", 2.0, 8.0, 4.5, step=0.1)
hypertension = st.sidebar.selectbox("Hypertension", ["yes", "no"])
diabetes_mellitus = st.sidebar.selectbox("Diabetes Mellitus", ["yes", "no"])
coronary_artery_disease = st.sidebar.selectbox("Coronary Artery Disease", ["yes", "no"])
appetite = st.sidebar.selectbox("Appetite", ["good", "poor"])
pedal_edema = st.sidebar.selectbox("Pedal Edema", ["yes", "no"])
anemia = st.sidebar.selectbox("Anemia", ["yes", "no"])

# Create input dataframe
input_data = pd.DataFrame({
    "age": [age],
    "blood_pressure": [blood_pressure],
    "specific_gravity": [specific_gravity],
    "albumin": [albumin],
    "sugar": [sugar],
    "red_blood_cells": [red_blood_cells],
    "pus_cell": [pus_cell],
    "pus_cell_clumps": [pus_cell_clumps],
    "bacteria": [bacteria],
    "blood_glucose_random": [blood_glucose_random],
    "blood_urea": [blood_urea],
    "serum_creatinine": [serum_creatinine],
    "sodium": [sodium],
    "potassium": [potassium],
    "hemoglobin": [hemoglobin],
    "packed_cell_volume": [packed_cell_volume],
    "white_blood_cell_count": [white_blood_cell_count],
    "red_blood_cell_count": [red_blood_cell_count],
    "hypertension": [hypertension],
    "diabetes_mellitus": [diabetes_mellitus],
    "coronary_artery_disease": [coronary_artery_disease],
    "appetite": [appetite],
    "pedal_edema": [pedal_edema],
    "anemia": [anemia]
})


def _build_chat_prompts(user_msg: str) -> tuple[str, str]:
    """
    Build system and user prompts for the chat assistant using available context.
    
    Args:
        user_msg: The user's question
        
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    # Base system prompt with safety guardrails
    system_prompt = """You are a CKD Risk Assessment Assistant. Your role is strictly scoped to answering questions about:
- Digital Twin simulations and what-if scenarios
- Counterfactual explanations (how to change outcomes)
- Similar patient comparisons
- Model explainability (SHAP values, feature importance)

STRICT RULES:
1. ONLY answer questions using the provided context from these four areas
2. If a question is outside this scope, respond EXACTLY with: "Out of scope for this chat. Load Digital Twin, Counterfactuals, or Similar Patients first."
3. NEVER provide medication dosing or prescribing advice
4. ALWAYS include appropriate units: mmHg (blood pressure), mg/dL (glucose, creatinine, urea), mEq/L (electrolytes), g/dL (hemoglobin), % (packed cell volume)
5. Round numeric values appropriately (0-3 decimal places)
6. ALWAYS end your response with: "Educational tool; not medical advice."
7. Provide concise, structured answers with headings when useful

Format your responses professionally with clear sections when appropriate."""

    # Check available contexts
    ctx_digital_twin = st.session_state.ctx_digital_twin
    ctx_counterfactual = st.session_state.ctx_counterfactual
    ctx_similar = st.session_state.ctx_similar
    ctx_explain = st.session_state.ctx_explain
    
    available_contexts = []
    context_data = []
    
    # Build context sections
    if ctx_explain:
        available_contexts.append("Explainability")
        context_data.append(f"""
### EXPLAINABILITY CONTEXT:
Schema Version: {ctx_explain.get('schema_version', 'N/A')}
Model: {ctx_explain.get('model', 'N/A')}
Top Features: {ctx_explain.get('top', 'N/A')}
Mode: {ctx_explain.get('mode', 'N/A')}
SHAP Values: {json.dumps(ctx_explain.get('shap_values', {}), indent=2)}
Feature Contributions: {json.dumps(ctx_explain.get('feature_contributions', {}), indent=2)}
""")
    
    if ctx_digital_twin:
        available_contexts.append("Digital Twin")
        context_data.append(f"""
### DIGITAL TWIN CONTEXT:
Schema Version: {ctx_digital_twin.get('schema_version', 'N/A')}
Model: {ctx_digital_twin.get('model', 'N/A')}
Threshold: {ctx_digital_twin.get('threshold', 'N/A')}
Swept Features: {json.dumps(ctx_digital_twin.get('swept_features', []), indent=2)}
Summary: {ctx_digital_twin.get('summary', 'N/A')}
Sample Results: {json.dumps(ctx_digital_twin.get('rows_sample', []), indent=2)}
""")
    
    if ctx_counterfactual:
        available_contexts.append("Counterfactuals")
        context_data.append(f"""
### COUNTERFACTUAL CONTEXT:
Schema Version: {ctx_counterfactual.get('schema_version', 'N/A')}
Model: {ctx_counterfactual.get('model', 'N/A')}
Threshold: {ctx_counterfactual.get('threshold', 'N/A')}
Target Probability: {ctx_counterfactual.get('target_prob', 'N/A')}
Initial Probability: {ctx_counterfactual.get('initial_prob', 'N/A')}
Final Probability: {ctx_counterfactual.get('final_prob', 'N/A')}
Converged: {ctx_counterfactual.get('converged', 'N/A')}
Steps Taken: {ctx_counterfactual.get('steps', 'N/A')}
Final Changes: {json.dumps(ctx_counterfactual.get('final_candidate', {}), indent=2)}
""")
    
    if ctx_similar:
        available_contexts.append("Similar Patients")
        context_data.append(f"""
### SIMILAR PATIENTS CONTEXT:
Schema Version: {ctx_similar.get('schema_version', 'N/A')}
K Neighbors: {ctx_similar.get('k', 'N/A')}
Distance Metric: {ctx_similar.get('metric', 'N/A')}
Summary: {ctx_similar.get('summary', 'N/A')}
Sample Neighbors: {json.dumps(ctx_similar.get('neighbors_sample', []), indent=2)}
""")
    
    # Build user prompt
    if not available_contexts:
        user_prompt = f"""User Question: {user_msg}

AVAILABLE CONTEXT: None

Since no context is available, respond with the out-of-scope message."""
    else:
        user_prompt = f"""User Question: {user_msg}

AVAILABLE CONTEXTS: {', '.join(available_contexts)}

{''.join(context_data)}

Answer the user's question using ONLY the above context. If the question cannot be answered with this context, use the out-of-scope response."""
    
    return system_prompt, user_prompt


def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Call the LLM API to generate a response.
    
    Args:
        system_prompt: System-level instructions
        user_prompt: User question with context
        
    Returns:
        Generated response text
    """
    try:
        # Configure OpenAI API (ensure API key is set in environment or Streamlit secrets)
        if "openai_api_key" in st.secrets:
            openai.api_key = st.secrets["openai_api_key"]
        elif "OPENAI_API_KEY" in st.session_state:
            openai.api_key = st.session_state.OPENAI_API_KEY
        else:
            return "⚠️ OpenAI API key not configured. Please set it in Streamlit secrets or session state."
        
        response = openai.ChatCompletion.create(
            model="gpt-4",  # or "gpt-3.5-turbo" for faster/cheaper responses
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,  # Lower temperature for more consistent, factual responses
            max_tokens=1000
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Ensure disclaimer is present
        disclaimer = "Educational tool; not medical advice."
        if disclaimer not in answer:
            answer += f"\n\n{disclaimer}"
        
        return answer
        
    except Exception as e:
        return f"⚠️ Error calling LLM: {str(e)}\n\nPlease check your API configuration and try again."


def render_context_status():
    """Render status chips showing which contexts are available."""
    st.markdown("##### Available Context:")
    
    cols = st.columns(4)
    contexts = [
        ("Explainability", st.session_state.ctx_explain, cols[0]),
        ("Digital Twin", st.session_state.ctx_digital_twin, cols[1]),
        ("Counterfactuals", st.session_state.ctx_counterfactual, cols[2]),
        ("Similar Patients", st.session_state.ctx_similar, cols[3])
    ]
    
    for name, ctx, col in contexts:
        with col:
            if ctx:
                st.success(f"✓ {name}", icon="✅")
            else:
                st.warning(f"✗ {name}", icon="⚠️")
    
    st.markdown("---")


# Main app layout with Chat as first tab
tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Chat (Assistant)", 
    "Single Prediction", 
    "Digital Twin", 
    "Counterfactuals", 
    "Similar Patients",
    "Explainability"
])

# Chat (Assistant) Tab
with tab0:
    st.header("🤖 CKD Risk Assessment Assistant")
    
    st.markdown("""
    Ask questions about:
    - **Digital Twin**: What-if scenarios and feature sweeps
    - **Counterfactuals**: How to change risk predictions
    - **Similar Patients**: Comparison with similar cases
    - **Explainability**: Feature importance and SHAP values
    """)
    
    # Show context status
    render_context_status()
    
    # Display chat history
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the patient's CKD risk assessment..."):
        # Add user message to chat
        st.session_state.chat_messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                system_prompt, user_prompt = _build_chat_prompts(prompt)
                response = call_llm(system_prompt, user_prompt)
                st.markdown(response)
        
        # Add assistant response to chat
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat()
        })
    
    # Clear chat button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()
    
    with col2:
        if st.button("🔄 Clear Context", use_container_width=True):
            st.session_state.ctx_digital_twin = None
            st.session_state.ctx_counterfactual = None
            st.session_state.ctx_similar = None
            st.session_state.ctx_explain = None
            st.rerun()

# Single Prediction Tab
with tab1:
    st.header("Single Prediction")
    
    if model is None:
        st.error("Model not available. Please check data loading.")
    else:
        if st.button("Predict"):
            try:
                prediction = model.predict_proba(input_data)[0][1]
                st.session_state.prediction = prediction
                st.write(f"**Probability of CKD:** {prediction:.2%}")
                
                # SHAP explanation
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_data)
                st.session_state.explanation = shap_values
                
                # Generate SHAP waterfall plot
                st.subheader("Feature Importance (SHAP)")
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.plots.waterfall(
                    shap.Explanation(
                        values=shap_values[1][0],
                        base_values=explainer.expected_value[1],
                        data=input_data.iloc[0],
                        feature_names=input_data.columns.tolist()
                    ),
                    show=False
                )
                st.pyplot(fig)
                
                # Populate explainability context
                feature_contributions = {}
                feature_names = input_data.columns.tolist()
                shap_vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
                
                for i, feature in enumerate(feature_names):
                    feature_contributions[feature] = {
                        "value": float(input_data.iloc[0][feature]),
                        "shap_value": float(shap_vals[i]),
                        "contribution": f"{float(shap_vals[i]):.4f}"
                    }
                
                # Sort by absolute SHAP value
                sorted_features = sorted(
                    feature_contributions.items(),
                    key=lambda x: abs(x[1]["shap_value"]),
                    reverse=True
                )
                
                st.session_state.ctx_explain = {
                    "schema_version": "1.0",
                    "model": "RandomForestClassifier",
                    "top": 10,
                    "mode": "single_prediction",
                    "prediction_probability": float(prediction),
                    "base_value": float(explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value),
                    "shap_values": {k: v["shap_value"] for k, v in sorted_features[:10]},
                    "feature_contributions": dict(sorted_features[:10]),
                    "timestamp": datetime.now().isoformat()
                }
                
                st.success("✅ Explainability context updated for chat assistant")
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

# Digital Twin Tab
with tab2:
    st.header("Digital Twin")
    st.markdown("Simulate what-if scenarios by modifying patient features.")
    
    if model is None:
        st.error("Model not available. Please check data loading.")
    else:
        # Feature selection for sweep
        st.subheader("Feature Sweep Configuration")
        
        numeric_features = [
            "age", "blood_pressure", "specific_gravity", "blood_glucose_random",
            "blood_urea", "serum_creatinine", "sodium", "potassium", "hemoglobin",
            "packed_cell_volume", "white_blood_cell_count", "red_blood_cell_count"
        ]
        
        selected_feature = st.selectbox("Select feature to sweep:", numeric_features)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            min_val = st.number_input("Min value:", value=float(input_data[selected_feature].iloc[0]) * 0.5)
        with col2:
            max_val = st.number_input("Max value:", value=float(input_data[selected_feature].iloc[0]) * 1.5)
        with col3:
            num_steps = st.number_input("Number of steps:", min_value=5, max_value=50, value=20)
        
        if st.button("Run Digital Twin Simulation"):
            try:
                # Generate sweep values
                sweep_values = np.linspace(min_val, max_val, num_steps)
                results = []
                
                progress_bar = st.progress(0)
                for i, val in enumerate(sweep_values):
                    # Create modified input
                    modified_input = input_data.copy()
                    modified_input[selected_feature] = val
                    
                    # Predict
                    pred_proba = model.predict_proba(modified_input)[0][1]
                    results.append({
                        "feature": selected_feature,
                        "value": float(val),
                        "probability": float(pred_proba)
                    })
                    
                    progress_bar.progress((i + 1) / num_steps)
                
                progress_bar.empty()
                
                # Create visualization
                results_df = pd.DataFrame(results)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=results_df["value"],
                    y=results_df["probability"],
                    mode='lines+markers',
                    name='CKD Probability',
                    line=dict(color='#FF6B6B', width=3),
                    marker=dict(size=6)
                ))
                
                # Add current value marker
                current_val = float(input_data[selected_feature].iloc[0])
                current_pred = model.predict_proba(input_data)[0][1]
                fig.add_trace(go.Scatter(
                    x=[current_val],
                    y=[current_pred],
                    mode='markers',
                    name='Current Value',
                    marker=dict(size=15, color='#4ECDC4', symbol='star')
                ))
                
                fig.update_layout(
                    title=f"Digital Twin: Impact of {selected_feature} on CKD Risk",
                    xaxis_title=selected_feature,
                    yaxis_title="CKD Probability",
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary statistics
                st.subheader("Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Min Probability", f"{results_df['probability'].min():.2%}")
                with col2:
                    st.metric("Max Probability", f"{results_df['probability'].max():.2%}")
                with col3:
                    st.metric("Range", f"{(results_df['probability'].max() - results_df['probability'].min()):.2%}")
                
                # Populate digital twin context
                summary = f"Swept {selected_feature} from {min_val:.2f} to {max_val:.2f} in {num_steps} steps. " \
                          f"CKD probability ranged from {results_df['probability'].min():.2%} to {results_df['probability'].max():.2%}."
                
                st.session_state.ctx_digital_twin = {
                    "schema_version": "1.0",
                    "model": "RandomForestClassifier",
                    "threshold": 0.5,
                    "swept_features": [selected_feature],
                    "summary": summary,
                    "rows_sample": results[:10],  # First 10 results
                    "current_value": current_val,
                    "current_probability": float(current_pred),
                    "min_probability": float(results_df['probability'].min()),
                    "max_probability": float(results_df['probability'].max()),
                    "timestamp": datetime.now().isoformat()
                }
                
                st.success("✅ Digital Twin context updated for chat assistant")
                
            except Exception as e:
                st.error(f"Error during digital twin simulation: {str(e)}")

# Counterfactuals Tab
with tab3:
    st.header("Counterfactuals")
    st.markdown("Find the minimal changes needed to achieve a target CKD risk.")
    
    if model is None:
        st.error("Model not available. Please check data loading.")
    else:
        # Get current prediction
        if st.session_state.prediction is None:
            st.info("Please run a prediction first in the 'Single Prediction' tab.")
        else:
            current_prob = st.session_state.prediction
            st.write(f"**Current CKD Probability:** {current_prob:.2%}")
            
            target_prob = st.slider(
                "Target CKD Probability:",
                min_value=0.0,
                max_value=1.0,
                value=max(0.0, current_prob - 0.2),
                step=0.01,
                format="%.2f"
            )
            
            # Feature constraints
            st.subheader("Feature Constraints")
            st.markdown("Select which features can be modified:")
            
            modifiable_features = {}
            for feature in numeric_features:
                modifiable_features[feature] = st.checkbox(
                    f"Allow modifying {feature}",
                    value=True,
                    key=f"cf_{feature}"
                )
            
            max_iterations = st.slider("Max iterations:", 10, 200, 50)
            
            if st.button("Find Counterfactual"):
                try:
                    with st.spinner("Searching for counterfactual..."):
                        # Simple gradient-based counterfactual search
                        candidate = input_data.copy()
                        learning_rate = 0.1
                        converged = False
                        
                        for step in range(max_iterations):
                            current_pred = model.predict_proba(candidate)[0][1]
                            
                            if abs(current_pred - target_prob) < 0.01:
                                converged = True
                                break
                            
                            # Calculate gradients (approximation using finite differences)
                            for feature in numeric_features:
                                if not modifiable_features.get(feature, False):
                                    continue
                                
                                # Small perturbation
                                epsilon = 0.01
                                candidate_plus = candidate.copy()
                                candidate_plus[feature] = candidate[feature].iloc[0] + epsilon
                                
                                pred_plus = model.predict_proba(candidate_plus)[0][1]
                                gradient = (pred_plus - current_pred) / epsilon
                                
                                # Update feature value
                                adjustment = -learning_rate * gradient * (current_pred - target_prob)
                                new_val = candidate[feature].iloc[0] + adjustment
                                
                                # Clip to reasonable bounds (using input feature as reference)
                                min_bound = input_data[feature].iloc[0] * 0.5
                                max_bound = input_data[feature].iloc[0] * 2.0
                                new_val = np.clip(new_val, min_bound, max_bound)
                                
                                candidate[feature] = new_val
                        
                        final_prob = model.predict_proba(candidate)[0][1]
                        
                        # Display results
                        if converged:
                            st.success(f"✅ Counterfactual found! Converged in {step + 1} steps.")
                        else:
                            st.warning(f"⚠️ Did not fully converge. Got to {final_prob:.2%} (target: {target_prob:.2%})")
                        
                        st.write(f"**Final Probability:** {final_prob:.2%}")
                        
                        # Show changes
                        st.subheader("Required Changes")
                        changes = []
                        for feature in numeric_features:
                            original_val = input_data[feature].iloc[0]
                            new_val = candidate[feature].iloc[0]
                            if abs(new_val - original_val) > 0.01:
                                change_pct = ((new_val - original_val) / original_val) * 100
                                changes.append({
                                    "Feature": feature,
                                    "Original": f"{original_val:.2f}",
                                    "New": f"{new_val:.2f}",
                                    "Change": f"{change_pct:+.1f}%"
                                })
                        
                        if changes:
                            changes_df = pd.DataFrame(changes)
                            st.dataframe(changes_df, use_container_width=True)
                            
                            # Populate counterfactual context
                            final_candidate = {row["Feature"]: {
                                "original": row["Original"],
                                "new": row["New"],
                                "change": row["Change"]
                            } for _, row in changes_df.iterrows()}
                            
                            st.session_state.ctx_counterfactual = {
                                "schema_version": "1.0",
                                "model": "RandomForestClassifier",
                                "threshold": 0.5,
                                "target_prob": float(target_prob),
                                "initial_prob": float(current_prob),
                                "final_prob": float(final_prob),
                                "converged": converged,
                                "steps": step + 1,
                                "final_candidate": final_candidate,
                                "num_features_changed": len(changes),
                                "timestamp": datetime.now().isoformat()
                            }
                            
                            st.success("✅ Counterfactual context updated for chat assistant")
                        else:
                            st.info("No significant changes needed.")
                            
                except Exception as e:
                    st.error(f"Error finding counterfactual: {str(e)}")

# Similar Patients Tab
with tab4:
    st.header("Similar Patients")
    st.markdown("Find similar patients from the training dataset.")
    
    if model is None or X_train is None:
        st.error("Model or training data not available.")
    else:
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler
        
        k = st.slider("Number of similar patients:", 3, 20, 5)
        distance_metric = st.selectbox("Distance metric:", ["euclidean", "manhattan", "cosine"])
        
        if st.button("Find Similar Patients"):
            try:
                with st.spinner("Finding similar patients..."):
                    # Standardize features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    input_scaled = scaler.transform(input_data)
                    
                    # Find nearest neighbors
                    nn = NearestNeighbors(n_neighbors=k, metric=distance_metric)
                    nn.fit(X_train_scaled)
                    distances, indices = nn.kneighbors(input_scaled)
                    
                    # Get similar patients
                    similar_patients = X_train.iloc[indices[0]]
                    similar_outcomes = y_train.iloc[indices[0]]
                    
                    # Display results
                    st.subheader("Similar Patients Overview")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        ckd_count = similar_outcomes.sum()
                        st.metric("Patients with CKD", f"{ckd_count}/{k}")
                    with col2:
                        avg_distance = distances[0].mean()
                        st.metric("Average Distance", f"{avg_distance:.3f}")
                    with col3:
                        ckd_rate = (ckd_count / k) * 100
                        st.metric("CKD Rate", f"{ckd_rate:.1f}%")
                    
                    # Detailed comparison
                    st.subheader("Patient Comparisons")
                    
                    comparison_data = []
                    for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
                        patient = X_train.iloc[idx]
                        outcome = "CKD" if y_train.iloc[idx] == 1 else "No CKD"
                        
                        comparison_data.append({
                            "Patient": f"#{i+1}",
                            "Distance": f"{dist:.3f}",
                            "Outcome": outcome,
                            "Age": patient["age"],
                            "BP": patient["blood_pressure"],
                            "Creatinine": patient["serum_creatinine"],
                            "Hemoglobin": patient["hemoglobin"]
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Feature comparison visualization
                    st.subheader("Feature Distribution Comparison")
                    
                    features_to_compare = st.multiselect(
                        "Select features to compare:",
                        numeric_features,
                        default=["age", "blood_pressure", "serum_creatinine", "hemoglobin"]
                    )
                    
                    if features_to_compare:
                        fig = go.Figure()
                        
                        for feature in features_to_compare:
                            similar_vals = similar_patients[feature].values
                            current_val = input_data[feature].iloc[0]
                            
                            fig.add_trace(go.Box(
                                y=similar_vals,
                                name=feature,
                                boxmean='sd'
                            ))
                        
                        fig.update_layout(
                            title="Feature Distribution in Similar Patients",
                            yaxis_title="Value",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Populate similar patients context
                    neighbors_sample = []
                    for i, (idx, dist) in enumerate(zip(indices[0][:5], distances[0][:5])):  # First 5
                        patient = X_train.iloc[idx]
                        neighbors_sample.append({
                            "patient_id": int(idx),
                            "distance": float(dist),
                            "outcome": "CKD" if y_train.iloc[idx] == 1 else "No CKD",
                            "features": {k: float(v) for k, v in patient.to_dict().items()}
                        })
                    
                    summary = f"Found {k} similar patients using {distance_metric} distance. " \
                              f"{ckd_count} out of {k} ({ckd_rate:.1f}%) have CKD. " \
                              f"Average distance: {avg_distance:.3f}"
                    
                    st.session_state.ctx_similar = {
                        "schema_version": "1.0",
                        "k": k,
                        "metric": distance_metric,
                        "neighbors_sample": neighbors_sample,
                        "summary": summary,
                        "ckd_count": int(ckd_count),
                        "ckd_rate": float(ckd_rate),
                        "average_distance": float(avg_distance),
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    st.success("✅ Similar Patients context updated for chat assistant")
                    
            except Exception as e:
                st.error(f"Error finding similar patients: {str(e)}")

# Explainability Tab (separate from Single Prediction)
with tab5:
    st.header("Model Explainability")
    st.markdown("Deep dive into model behavior and feature importance.")
    
    if model is None:
        st.error("Model not available. Please check data loading.")
    else:
        if st.session_state.explanation is None:
            st.info("Please run a prediction first in the 'Single Prediction' tab.")
        else:
            st.subheader("Global Feature Importance")
            
            # Feature importance from Random Forest
            feature_importance = pd.DataFrame({
                'feature': input_data.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig = go.Figure(go.Bar(
                x=feature_importance['importance'][:15],
                y=feature_importance['feature'][:15],
                orientation='h',
                marker=dict(color='#4ECDC4')
            ))
            
            fig.update_layout(
                title="Top 15 Most Important Features (Random Forest)",
                xaxis_title="Importance",
                yaxis_title="Feature",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("SHAP Analysis")
            st.markdown("Local explanation for the current prediction:")
            
            if st.button("Generate SHAP Summary"):
                try:
                    explainer = shap.TreeExplainer(model)
                    
                    # SHAP summary for test set sample
                    sample_size = min(100, len(X_test))
                    X_sample = X_test.sample(n=sample_size, random_state=42)
                    shap_values_sample = explainer.shap_values(X_sample)
                    
                    st.write("**SHAP Summary Plot (Sample of Test Set)**")
                    fig, ax = plt.subplots(figsize=(10, 8))
                    shap.summary_plot(
                        shap_values_sample[1] if len(shap_values_sample) > 1 else shap_values_sample,
                        X_sample,
                        show=False
                    )
                    st.pyplot(fig)
                    
                    st.success("SHAP analysis complete")
                    
                except Exception as e:
                    st.error(f"Error generating SHAP summary: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>⚠️ Educational Tool - Not Medical Advice</strong></p>
    <p>This application is for educational and research purposes only. 
    Always consult healthcare professionals for medical decisions.</p>
</div>
""", unsafe_allow_html=True)
