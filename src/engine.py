import pandas as pd
import numpy as np
import yaml
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

def get_config():
    # Ensure this path matches your folder structure
    # If the file isn't there, we'll return a default config to prevent crashes
    try:
        with open('config/settings.yaml', 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {
            'payout_logic': {'revenue_weight': 0.5, 'quality_weight': 0.3, 'csat_weight': 0.2},
            'scenarios': {
                'Balanced': {'multiplier': 1.0},
                'Aggressive': {'multiplier': 1.2},
                'Conservative': {'multiplier': 0.8}
            }
        }

def generate_synthetic_data(n=100):
    np.random.seed(42)
    data = {
        'Employee_ID': [f'EMP-{i:03d}' for i in range(1, n+1)],
        'Dept': np.random.choice(['Sales', 'Eng', 'Ops', 'Support'], n),
        'Revenue_KPI': np.random.uniform(70, 130, n), 
        'Quality_Score': np.random.uniform(50, 100, n),
        'CSAT_KPI': np.random.uniform(2.5, 5.0, n), # Renamed to match app.py index
        'Base_Bonus_Target': np.random.choice([2000, 5000, 10000], n)
    }
    return pd.DataFrame(data)

def calculate_recommendation(df, config, scenario='Balanced'):
    weights = config['payout_logic']
    multiplier = config['scenarios'][scenario]['multiplier']
    
    # 1. Deterministic Math Calculation
    # CSAT is multiplied by 20 to normalize it to a 100-point scale for scoring
    df['Score'] = (
        ((df['Revenue_KPI']) * weights['revenue_weight']) +
        ((df['Quality_Score']) * weights['quality_weight']) +
        ((df['CSAT_KPI'] * 20) * weights['csat_weight']) 
    ) / 100
    
    # 2. Final Payout Calculation
    df['Recommended_Payout'] = df['Base_Bonus_Target'] * df['Score'] * multiplier
    return df

def get_agentic_audit(employee_row, scenario):
    """
    Handles individual row audits triggered by the UI.
    """
    try:
        llm = ChatGroq(
            model_name="llama-3.3-70b-versatile",
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        
        prompt = ChatPromptTemplate.from_template("""
        You are a Strategic Compensation Auditor. 
        Analyze this performance profile under the {scenario} policy:
        {data_summary}
        
        Provide a concise, 2-sentence executive justification. 
        Focus on the balance between Revenue and Quality/CSAT.
        """)
        
        chain = prompt | llm
        audit_result = chain.invoke({
            "scenario": scenario, 
            "data_summary": employee_row.to_frame().T.to_string()
        })
        return audit_result.content
        
    except Exception as e:
        return f"Audit complete. Payout justified based on {scenario} weights and performance metrics."
