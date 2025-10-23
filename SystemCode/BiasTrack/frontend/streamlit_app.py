import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import sys
from utility import Utility


# Set page config
st.set_page_config(
    page_title="BiasTrack - Gender Pay Equity Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS for better styling
st.markdown("""
<style>
    *{
    font-family:sans-serif; !important;
    }
    .main-Biasheader {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-top: -31px !important; 
        }
    .main-header {
        font-size: 2rem !important;
        font-weight: bold;
        color: #1f77b4;
        text-align: left;
        margin-bottom: 1rem;}
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        cursor: pointer;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
    }
    .sidebar:hover{
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
            cursor: pointer;
            }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: boldest !important; 
        height: 50px;
        white-space: pre-wrap;
        border-radius: 14px 14px 0px 0px;
        padding-top: 10px;
        padding-bottom: 10px;
        padding-left: 15px;
        text-align: center;
        padding-right: 15px;
        transition: all 0.3s ease;
        color: white;
    }
    .st-emotion-cache-1q82h82 e1wr3kle3{
            text-align: center }
    .st-emotion-cache-9rsxm2 p{
            font-size:16px !important;
            }
    .stTabs [data-baseweb="tab"]:hover {
        cursor: pointer;
        color: white;
        font-weight: bold;
        background-color: #1f77b4;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Get project root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

# Base artifacts directory
artifacts_dir = os.path.join(project_root, 'BiasTrack', 'artifacts')

# 2. Define paths for model versions
# Model v1
preprocessor_v1_path = os.path.join(artifacts_dir, 'model_v1', 'preprocessors', 'preprocessor.pkl')
model_v1_path = os.path.join(artifacts_dir, 'model_v1', 'models', 'modelLinear')

# Model v2
preprocessor_v2_path = os.path.join(artifacts_dir, 'model_v2', 'preprocessors', 'preprocessor.joblib')
model_v2_path = os.path.join(artifacts_dir, 'model_v2', 'models', 'model.joblib')

# Instantiating Utility Class to use utility functions
utility = Utility()

# Choosing Model versions
version = 1 # Change model version from here

# Load data and model
@st.cache_data
def load_data():
    data_path = os.path.join(project_root, 'BiasTrack', 'dataset', 'incoming_data', 'gender_pay_gap_HR_data.csv')
    df = pd.read_csv(data_path)
    return df

@st.cache_resource
def load_preprocessor():
    sys.path.append(os.path.join(project_root, 'BiasTrack'))

    preprocessor_path = None

    if(version == 1):
        from src.biastrack.data.preprocess_v1 import DataPreprocessor
        preprocessor_path = preprocessor_v1_path
    else: 
        from src.biastrack.data.preprocess_v2 import DataPreprocessor
        preprocessor_path = preprocessor_v2_path

    if not os.path.exists(preprocessor_path):
        return None
    
    # Use file modification time as cache key to force reload when file changes
    mtime = os.path.getmtime(preprocessor_path)
    return DataPreprocessor.load_preprocessor(preprocessor_path)

@st.cache_resource
def load_model():

    if(version == 1):
        model_path = model_v1_path
    else:
        model_path = model_v2_path

    if not os.path.exists(model_path):
        return None
    
    # Use file modification time as cache key to force reload when file changes
    mtime = os.path.getmtime(model_path)
    return joblib.load(model_path)

# Utility functions for computing metrics
def compute_gender_counts(df):
    """Compute female and male counts"""
    return utility.female_male_count(df)

def compute_overall_pay_gap(df):
    """Compute overall pay gap percentage"""
    overall_gap = utility.overall_pay_gap(df)
    avg_pay_female = df[df['Gender'] == 'Female']['Total Pay'].mean()
    avg_pay_male = df[df['Gender'] == 'Male']['Total Pay'].mean()
    return overall_gap, avg_pay_female, avg_pay_male

def compute_pay_gap_by_role(df):
    """Compute pay gap by job role"""
    return utility.pay_gap_calculator(df)

def compute_avg_salaries_by_gender(df):
    """Compute average salaries by gender."""
    avg_salaries = df.groupby('Gender')['Total Pay'].mean().reset_index()
    return avg_salaries

def compute_role_statistics(df, role):
    """Compute statistics for a specific role."""
    role_data = df[df['JobTitle'] == role]
    female_avg = role_data[role_data['Gender'] == 'Female']['Total Pay'].mean()
    male_avg = role_data[role_data['Gender'] == 'Male']['Total Pay'].mean()
    current_gap = ((male_avg - female_avg) / female_avg) * 100 if female_avg > 0 else 0
    female_count = (role_data['Gender'] == 'Female').sum()
    return female_avg, male_avg, current_gap, female_count

def compute_pay_gap_from_predictions(predicted_pay_male, predicted_pay_female):
    """Compute pay gap percentage from predicted salaries."""
    if predicted_pay_female > 0:
        pay_gap = ((predicted_pay_male - predicted_pay_female) / predicted_pay_female) * 100
    else:
        pay_gap = 0
    return pay_gap

def compute_budget_adjustments(current_gap, desired_gap, female_avg, male_avg, female_count, male_count, budget):
    """Compute required adjustments for budget simulation."""
    gap_diff = desired_gap - current_gap
    if gap_diff < 0:  # Need to increase female pay
        adjustment_needed = abs(gap_diff) / 100 * female_avg
        per_female_adjustment = adjustment_needed / female_count if female_count > 0 else 0
        total_cost = per_female_adjustment * female_count
        budget_sufficient = total_cost <= budget
        additional_needed = max(0, total_cost - budget)
        return per_female_adjustment, 0, total_cost, budget_sufficient, additional_needed  # male adjustment 0
    elif gap_diff > 0:  # Need to decrease male pay
        adjustment_needed = (gap_diff / 100) * male_avg
        per_male_adjustment = adjustment_needed / male_count if male_count > 0 else 0
        total_savings = per_male_adjustment * male_count  # Savings from decreasing male pay
        return 0, per_male_adjustment, total_savings, True, 0  # female adjustment 0, cost 0 (savings)
    else:
        return 0, 0, 0, True, 0

# Dashboard 1: Overall Analysis
def dashboard1():
    st.markdown('<h1 class="main-header">📊 Overall Gender Pay Gap Analysis</h1>', unsafe_allow_html=True)

    df = load_data()
    df_processed = utility.preprocess_data(df)

    # Compute metrics using utility functions
    male_count, female_count = compute_gender_counts(df_processed)
    overall_pay_gap, female_avg, male_avg = compute_overall_pay_gap(df_processed)
    pay_gap_by_role = compute_pay_gap_by_role(df_processed)
    avg_salaries = compute_avg_salaries_by_gender(df_processed)

    # Key Metrics with enhanced styling
    st.subheader("🎯 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("👩 Female Employees", female_count)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.metric("👨 Male Employees", male_count)
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.metric("💰 Overall Pay Gap", f"{overall_pay_gap:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.metric("📈 Female Avg Pay", f"${female_avg:.0f}")
        st.markdown('</div>', unsafe_allow_html=True)


    col1, col2 = st.columns(2)

    with col1:
        # Enhanced Gender Ratio Pie Chart with hover effects
        fig_ratio = px.pie(
            values=[female_count, male_count],
            names=['Female', 'Male'],
            title="👥 Gender Distribution",
            color_discrete_sequence=['#FF69B4', '#4169E1'],
            hole=0.4
        )
        fig_ratio.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )
        st.plotly_chart(fig_ratio, use_container_width=True)

        # Average Salary by Gender
        fig_salary = px.bar(
            avg_salaries,
            x='Gender',
            y='Total Pay',
            title="💵 Average Salary by Gender",
            color='Gender',
            color_discrete_map={'Female': '#FF69B4', 'Male': '#4169E1'}
        )
        fig_salary.update_layout(showlegend=False)
        st.plotly_chart(fig_salary, use_container_width=True)

    with col2:
        # Enhanced Pay Gap by Role Chart
        fig_gap = px.bar(
            pay_gap_by_role.sort_values('Pay_Gap_%', ascending=True),
            x='JobTitle',
            y='Pay_Gap_%',
            title="📊 Pay Gap by Job Role",
            color='Pay_Gap_%',
            color_continuous_scale='RdYlBu_r'
        )
        fig_gap.update_xaxes(tickangle=45)
        st.plotly_chart(fig_gap, use_container_width=True)

        # Pay Gap Distribution
        fig_dist = px.histogram(
            pay_gap_by_role,
            x='Pay_Gap_%',
            title="📈 Pay Gap Distribution",
            nbins=10,
            color_discrete_sequence=['#1f77b4']
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    # Detailed Data Table
    st.subheader("📋 Detailed Pay Gap Analysis")
    st.dataframe(
        pay_gap_by_role.style.format({'Pay_Gap_%': '{:.2f}%', 'Female': '${:.0f}', 'Male': '${:.0f}'}),
        use_container_width=True
    )

# Dashboard 2: HR Simulation
def dashboard2():
    st.markdown('<h1 class="main-header">🎯 HR Simulation: Pay Gap Prediction</h1>', unsafe_allow_html=True)

    preprocessor = load_preprocessor()
    model = load_model()

    if preprocessor is None or model is None:
        st.error("Model or preprocessor not loaded. Please run the training pipeline first.")
        return

    # Interactive Input Section
    st.subheader("🔧 Adjust Parameters")

    col1, col2 = st.columns(2)
    with col1:
        salary = st.slider("💰 Base Salary", 30000, 200000, 80000, help="Select the base salary for prediction")

        role = st.selectbox("👔 Job Title", ['Software Engineer', 'Data Scientist', 'Manager', 'Graphic Designer', 'Financial Analyst', 'IT', 'Sales Associate', 'Warehouse Associate', 'Marketing Associate', 'Driver'], help="Choose the job role")

        gender = st.selectbox("👤 Gender", ['Male', 'Female'], help="Select the gender for prediction")

    with col2:
        age = st.slider("🎂 Age", 18, 65, 30, help="Select age in years")

        perf_eval = st.slider("⭐ Performance Evaluation", 1, 5, 3, help="Rate performance from 1-5")

        education = st.selectbox("🎓 Education", ['High School', 'College', 'Masters', 'PhD'], help="Select education level")

        dept = st.selectbox("🏢 Department", ['Engineering', 'Sales', 'Management', 'Administration', 'Operations'], help="Choose department")

        seniority = st.slider("📈 Seniority", 1, 5, 3, help="Select seniority level from 1-5")

    # Predict for selected gender
    input_data = pd.DataFrame({
        'JobTitle': [role],
        'Gender': [gender],
        'Age': [age],
        'PerfEval': [perf_eval],
        'Education': [education],
        'Dept': [dept],
        'Seniority': [seniority],
        'Total Pay': [salary]  # Placeholder, will be predicted
    })

    # Transform input
    if (version == 1):
        X_input,_ = preprocessor.transform(input_data)
    else:
        X_input = preprocessor.transform(input_data)   
    
    predicted_pay = model.predict(X_input)[0]

    # Predict for opposite gender
    opposite_gender = 'Female' if gender == 'Male' else 'Male'
    input_data_opp = input_data.copy()
    input_data_opp['Gender'] = opposite_gender

    if(version == 1):
        X_input_opp, _ = preprocessor.transform(input_data_opp)
    else:
        X_input_opp = preprocessor.transform(input_data_opp)    
    
    predicted_pay_opp = model.predict(X_input_opp)[0]

    # Compute gap using utility function
    pay_gap = compute_pay_gap_from_predictions(predicted_pay if gender == 'Male' else predicted_pay_opp, predicted_pay_opp if gender == 'Male' else predicted_pay)

    # Enhanced Results Display
    st.subheader("📊 Prediction Results")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(f"💵 {gender} Predicted Pay", f"${predicted_pay:.0f}")

    with col2:
        st.metric(f"💵 {opposite_gender} Predicted Pay", f"${predicted_pay_opp:.0f}")

    with col3:
        st.metric("📊 Predicted Pay Gap", f"{pay_gap:.2f}%")

    # Visualization of salary comparison using boxplot
    st.subheader("📈 Salary Comparison Visualization")

    # Load data for boxplot comparison after adjusting the slider values
    df = load_data()
    df_processed = utility.preprocess_data(df)
    

    # Filter data for the selected role
    role_data = df_processed[df_processed['JobTitle'] == role]

    fig_comparison = px.box(
        role_data,
        x='Gender',
        y='Total Pay',
        title=f"Salary Distribution for {role} by Gender",
        color='Gender',
        color_discrete_map={'Female': '#FF69B4', 'Male': '#4169E1'}
    )
    fig_comparison.update_layout(showlegend=True)
    st.plotly_chart(fig_comparison, use_container_width=True)

    # Input Summary
    with st.expander("📝 View Input Summary"):
        st.dataframe(input_data, use_container_width=True)

# Dashboard 3: Budget Simulator
def dashboard3():
    st.markdown('<h1 class="main-header">💰 Budget Simulator: Pay Gap Adjustments</h1>', unsafe_allow_html=True)

    df = load_data()
    df_processed = utility.preprocess_data(df)
    preprocessor = load_preprocessor()
    model = load_model()

    if preprocessor is None or model is None:
        st.error("Model or preprocessor not loaded. Please run the training pipeline first.")
        return

    # Interactive Input Section
    st.subheader("🔧 Simulation Parameters")

    col1, col2 = st.columns(2)
    with col1:
        budget = st.number_input("💵 Total Budget for Adjustments ($)", min_value=0, value=1000000, help="Enter the total budget available for salary adjustments")
        st.markdown('</div>', unsafe_allow_html=True)

        role = st.selectbox("👔 Target Role", df_processed['JobTitle'].unique(), help="Select the job role to analyze")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        desired_gap = st.slider("🎯 Desired Pay Gap %", -50.0, 50.0, 0.0, help="Set the target pay gap percentage (negative values mean women earn more)")
        st.markdown('</div>', unsafe_allow_html=True)

    # Use utility functions for role statistics and budget calculations
    female_avg, male_avg, current_gap, female_count = compute_role_statistics(df_processed, role)
    male_count = (df_processed[df_processed['JobTitle'] == role]['Gender'] == 'Male').sum()
    per_female_adjustment, per_male_adjustment, total_cost, budget_sufficient, additional_needed = compute_budget_adjustments(current_gap, desired_gap, female_avg, male_avg, female_count, male_count, budget)

    # Enhanced Results Display
    st.subheader("📊 Simulation Results")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📈 Current Pay Gap", f"{current_gap:.2f}%")

    with col2:
        st.metric("🎯 Target Gap", f"{desired_gap:.2f}%")

    with col3:
        gap_diff = desired_gap - current_gap
        st.metric("⚡ Gap Difference", f"{gap_diff:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    # Detailed Analysis
    st.subheader("💡 Detailed Analysis")

    if per_female_adjustment > 0:
        st.info(f"**Action Required:** To achieve the desired pay gap, increase female salaries by **${per_female_adjustment:.2f}** each.")
        st.write(f"**Total Cost:** ${total_cost:.2f}")
        st.write(f"**Number of Female Employees:** {female_count}")

        if budget_sufficient:
            st.success("✅ **Budget Status:** Sufficient! The adjustments can be made within the allocated budget.")
        else:
            st.error(f"❌ **Budget Status:** Insufficient! Need additional **${additional_needed:.2f}** to complete the adjustments.")
    elif per_male_adjustment > 0:
        st.info(f"**Action Required:** To achieve the desired pay gap, decrease male salaries by **${per_male_adjustment:.2f}** each.")
        st.write(f"**Total Savings:** ${total_cost:.2f}")  # total_cost is 0 for male adjustments, but we can show savings
        st.write(f"**Number of Male Employees:** {male_count}")
        st.success("✅ **Budget Status:** Savings generated! The adjustments will reduce costs.")
    else:
        st.success("✅ **No Action Needed:** The desired gap is already achieved or better than current. No adjustments required.")

    # Role Statistics with enhanced styling
    st.subheader("📊 Role Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("👩 Female Employees", female_count)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        male_count = (df_processed[df_processed['JobTitle'] == role]['Gender'] == 'Male').sum()
        st.metric("👨 Male Employees", male_count)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.metric("💵 Female Avg Pay", f"${female_avg:.0f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.metric("💵 Male Avg Pay", f"${male_avg:.0f}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Visualization
    st.subheader("📈 Salary Distribution Visualization")

    role_data = df_processed[df_processed['JobTitle'] == role]
    fig_role = px.box(
        role_data,
        x='Gender',
        y='Total Pay',
        title=f"Salary Distribution for {role}",
        color='Gender',
        color_discrete_map={'Female': '#FF69B4', 'Male': '#4169E1'}
    )
    fig_role.update_layout(showlegend=False)
    st.plotly_chart(fig_role, use_container_width=True)

# Main app
st.markdown("""
<h1 class="main-Biasheader">BiasTrack - Gender Pay Equity Analysis</h1>
""", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["Overall Analysis", "HR Simulation", "Budget Simulator"])

with tab1:
    dashboard1()

with tab2:
    dashboard2()

with tab3:
    dashboard3()


# Footer
st.markdown("""
<style>
    .footer {
        text-align: center;
        padding: 1rem;
        font-size: 0.9rem;
        color: #888888;
        margin-top: 3rem;
    }
</style>
<div class="footer">
    &copy; 2025 BiasTrack. All rights reserved.
</div>
""", unsafe_allow_html=True)