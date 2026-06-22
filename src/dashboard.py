import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import io

# Import config paths
from src.config import RAW_DATA_PATH, MODEL_PATH, FIGURES_DIR
from src.explain import get_explainer, get_individual_explanation

# Page config
st.set_page_config(
    page_title="HR Attrition Insights Dashboard",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
        color: #212529;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    .kpi-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 5px solid #4f46e5;
        margin-bottom: 20px;
    }
    .kpi-title {
        font-size: 0.9rem;
        color: #6c757d;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f2937;
        margin-top: 8px;
    }
    .risk-high {
        color: #dc3545;
        font-weight: bold;
    }
    .risk-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .risk-low {
        color: #28a745;
        font-weight: bold;
    }
    h1, h2, h3 {
        color: #1e293b;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to load data
@st.cache_data
def get_dataset():
    if not Path(RAW_DATA_PATH).exists():
        st.error(f"Raw dataset not found at: {RAW_DATA_PATH}")
        return pd.DataFrame()
    return pd.read_csv(RAW_DATA_PATH)

# Helper function to load model package
@st.cache_resource
def get_model_package():
    if not Path(MODEL_PATH).exists():
        return None
    try:
        pkg = joblib.load(MODEL_PATH)
        # Pre-initialize explainer
        pipeline = pkg["pipeline"]
        X_train_proc = pkg["X_train_proc"]
        explainer, _ = get_explainer(pipeline, X_train_proc)
        pkg["explainer"] = explainer
        return pkg
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load dataset and model
df = get_dataset()
model_pkg = get_model_package()

# Main Header
st.markdown("<h1 style='text-align: center; margin-bottom: 20px;'>👥 Employee Attrition Intelligence Portal</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748b; font-size: 1.1rem; margin-top: -15px;'>Transforming HR analytics into predictive insights with machine learning and explainable AI.</p>", unsafe_allow_html=True)
st.divider()

if df.empty:
    st.info("Please run the training pipeline to prepare the dataset and models.")
else:
    # Sidebar
    st.sidebar.markdown("<h3 style='margin-bottom: 20px;'>Navigation & Filters</h3>", unsafe_allow_html=True)
    mode = st.sidebar.radio(
        "View Mode",
        options=["📊 Executive Analytics", "🔍 Employee Risk Scoring"]
    )
    
    # Global sidebar filters (only relevant for EDA)
    if mode == "📊 Executive Analytics":
        st.sidebar.divider()
        st.sidebar.subheader("Analytics Filters")
        
        selected_dept = st.sidebar.multiselect(
            "Departments",
            options=sorted(df["Department"].unique()),
            default=sorted(df["Department"].unique())
        )
        
        selected_travel = st.sidebar.multiselect(
            "Business Travel",
            options=sorted(df["BusinessTravel"].unique()),
            default=sorted(df["BusinessTravel"].unique())
        )
        
        selected_ot = st.sidebar.multiselect(
            "Overtime Status",
            options=sorted(df["OverTime"].unique()),
            default=sorted(df["OverTime"].unique())
        )
        
        # Apply filters
        filtered_df = df[
            (df["Department"].isin(selected_dept)) &
            (df["BusinessTravel"].isin(selected_travel)) &
            (df["OverTime"].isin(selected_ot))
        ]
    else:
        filtered_df = df.copy()

    # View Mode 1: Executive Analytics
    if mode == "📊 Executive Analytics":
        
        # KPI calculations
        total_employees = len(filtered_df)
        if total_employees == 0:
            st.warning("No data matching selection filters.")
        else:
            attrition_count = (filtered_df["Attrition"] == "Yes").sum()
            attrition_rate = attrition_count / total_employees
            avg_tenure = filtered_df["YearsAtCompany"].mean()
            avg_satisfaction = filtered_df["JobSatisfaction"].mean()
            
            # Row 1: KPI Cards
            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
            
            with kpi_col1:
                st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-title">Total Headcount</div>
                        <div class="kpi-value">{total_employees:,}</div>
                    </div>
                """, unsafe_allow_html=True)
                
            with kpi_col2:
                st.markdown(f"""
                    <div class="kpi-card" style="border-left-color: #dc3545;">
                        <div class="kpi-title">Attrition Rate</div>
                        <div class="kpi-value">{attrition_rate:.1%}</div>
                    </div>
                """, unsafe_allow_html=True)
                
            with kpi_col3:
                st.markdown(f"""
                    <div class="kpi-card" style="border-left-color: #10b981;">
                        <div class="kpi-title">Average Tenure</div>
                        <div class="kpi-value">{avg_tenure:.1f} Yrs</div>
                    </div>
                """, unsafe_allow_html=True)
                
            with kpi_col4:
                st.markdown(f"""
                    <div class="kpi-card" style="border-left-color: #ffc107;">
                        <div class="kpi-title">Avg Job Satisfaction</div>
                        <div class="kpi-value">{avg_satisfaction:.2f}/4</div>
                    </div>
                """, unsafe_allow_html=True)
                
            # Tab selection for breakdown visualizations
            tab1, tab2, tab3 = st.tabs(["📊 Department & Role Drivers", "📈 Key Employee Behaviors", "🧠 Global SHAP Drivers"])
            
            with tab1:
                st.subheader("Attrition Trends by Job Hierarchy")
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    # Attrition by Department
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.countplot(
                        data=filtered_df, x="Department", hue="Attrition", 
                        palette={"Yes": "#ef4444", "No": "#3b82f6"}, ax=ax
                    )
                    ax.set_title("Attrition count by Department")
                    ax.grid(axis='y', linestyle='--', alpha=0.5)
                    st.pyplot(fig)
                    plt.close()
                    
                with col_chart2:
                    # Attrition rate by Job Role
                    fig, ax = plt.subplots(figsize=(7, 4.5))
                    role_attr = filtered_df.groupby("JobRole")["Attrition"].apply(
                        lambda x: (x == "Yes").mean()
                    ).reset_index().sort_values("Attrition", ascending=False)
                    
                    sns.barplot(
                        data=role_attr, y="JobRole", x="Attrition", 
                        color="#ef4444", ax=ax
                    )
                    ax.set_title("Attrition Rate by Job Role (Sorted)")
                    ax.set_xlabel("Attrition Rate")
                    ax.set_ylabel("")
                    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
                    ax.grid(axis='x', linestyle='--', alpha=0.5)
                    st.pyplot(fig)
                    plt.close()
                    
            with tab2:
                st.subheader("Key Attrition Correlates")
                col_chart3, col_chart4 = st.columns(2)
                
                with col_chart3:
                    # Overtime vs Attrition
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ot_attr = pd.crosstab(filtered_df["OverTime"], filtered_df["Attrition"], normalize="index") * 100
                    ot_attr.plot(kind="bar", stacked=True, color=["#3b82f6", "#ef4444"], ax=ax)
                    ax.set_title("Attrition Percentage by Overtime Working")
                    ax.set_ylabel("Percentage")
                    ax.set_xlabel("Overtime")
                    plt.xticks(rotation=0)
                    ax.legend(title="Attrition", bbox_to_anchor=(1.05, 1))
                    st.pyplot(fig)
                    plt.close()
                    
                with col_chart4:
                    # Income vs Attrition
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.boxplot(
                        data=filtered_df, x="Attrition", y="MonthlyIncome", 
                        palette={"Yes": "#ef4444", "No": "#3b82f6"}, ax=ax
                    )
                    ax.set_title("Monthly Income Distribution vs Attrition Status")
                    st.pyplot(fig)
                    plt.close()
                    
                col_chart5, col_chart6 = st.columns(2)
                with col_chart5:
                    # Business Travel vs Attrition
                    fig, ax = plt.subplots(figsize=(6, 4))
                    travel_attr = pd.crosstab(filtered_df["BusinessTravel"], filtered_df["Attrition"], normalize="index") * 100
                    travel_attr.plot(kind="bar", stacked=True, color=["#3b82f6", "#ef4444"], ax=ax)
                    ax.set_title("Attrition vs Business Travel Frequency")
                    ax.set_ylabel("Percentage")
                    plt.xticks(rotation=15)
                    st.pyplot(fig)
                    plt.close()
                    
                with col_chart6:
                    # Work Life Balance vs Attrition
                    fig, ax = plt.subplots(figsize=(6, 4))
                    wlb_attr = pd.crosstab(filtered_df["WorkLifeBalance"], filtered_df["Attrition"], normalize="index") * 100
                    wlb_attr.plot(kind="bar", stacked=True, color=["#3b82f6", "#ef4444"], ax=ax)
                    ax.set_title("Attrition Rate vs Work-Life Balance Rating")
                    ax.set_xlabel("Work Life Balance Rating (1=Bad, 4=Best)")
                    ax.set_ylabel("Percentage")
                    plt.xticks(rotation=0)
                    st.pyplot(fig)
                    plt.close()

            with tab3:
                st.subheader("Global Feature Importance (SHAP)")
                shap_summary_path = Path(FIGURES_DIR) / "shap_global_summary.png"
                if shap_summary_path.exists():
                    st.image(str(shap_summary_path), caption="SHAP Global Importance Summary")
                else:
                    st.info("The SHAP global summary plot will appear here once the model is trained.")

    # View Mode 2: Employee Risk Scoring
    else:
        st.subheader("🔍 Predict Attrition Risk for Individual Employees")
        
        if model_pkg is None:
            st.warning("⚠️ No trained model artifact found. Please complete the model training step first to unlock this scoring module.")
        else:
            # Layout: Form on left, Prediction result and SHAP explanation on right
            form_col, result_col = st.columns([1, 1.2])
            
            # Setup input forms
            with form_col:
                st.markdown("### Input Employee Attributes")
                
                with st.form("employee_scoring_form"):
                    # Category columns
                    st.markdown("**Employment Demographics**")
                    f_age = st.slider("Age", 18, 70, 35)
                    f_gender = st.selectbox("Gender", ["Male", "Female"])
                    f_marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
                    
                    st.markdown("**Department & Role**")
                    f_dept = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
                    
                    # Filter job roles depending on department
                    role_options = {
                        "Sales": ["Sales Executive", "Sales Representative", "Manager"],
                        "Research & Development": ["Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Research Director", "Manager"],
                        "Human Resources": ["Human Resources", "Manager"]
                    }
                    f_role = st.selectbox("Job Role", role_options[f_dept])
                    f_level = st.slider("Job Level", 1, 5, 2)
                    
                    st.markdown("**Compensation & Hours**")
                    f_income = st.number_input("Monthly Income ($)", 1000, 30000, 5000)
                    f_ot = st.selectbox("Overtime Working?", ["Yes", "No"])
                    f_daily = st.number_input("Daily Rate ($)", 100, 2000, 800)
                    f_hourly = st.number_input("Hourly Rate ($)", 30, 150, 65)
                    f_rate = st.number_input("Monthly Rate ($)", 2000, 30000, 14000)
                    
                    st.markdown("**Work Environment & Quality of Life**")
                    f_travel = st.selectbox("Business Travel Frequency", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
                    f_distance = st.slider("Distance From Home (miles)", 1, 30, 5)
                    f_env_sat = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
                    f_job_sat = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
                    f_job_inv = st.slider("Job Involvement (1-4)", 1, 4, 3)
                    f_rel_sat = st.slider("Relationship Satisfaction (1-4)", 1, 4, 3)
                    f_wlb = st.slider("Work-Life Balance (1-4)", 1, 4, 3)
                    
                    st.markdown("**History & Career**")
                    f_stock = st.slider("Stock Option Level (0-3)", 0, 3, 1)
                    f_total_years = st.slider("Total Working Years", 0, 45, 10)
                    f_num_comp = st.slider("Number of Companies Worked At", 0, 9, 2)
                    f_training = st.slider("Training Times Last Year", 0, 6, 2)
                    f_years_co = st.slider("Years At Company", 0, 40, 5)
                    f_years_role = st.slider("Years In Current Role", 0, 20, 3)
                    f_years_promo = st.slider("Years Since Last Promotion", 0, 15, 1)
                    f_years_mgr = st.slider("Years With Current Manager", 0, 20, 3)
                    
                    # Education
                    f_edu = st.slider("Education Level (1=Below College, 5=Doctor)", 1, 5, 3)
                    f_field = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"])
                    f_hike = st.slider("Percent Salary Hike", 10, 30, 14)
                    f_perf = st.slider("Performance Rating (1-4)", 1, 4, 3)
                    
                    submit_button = st.form_submit_button("Score Employee")
            
            # Predict and Explain
            with result_col:
                st.markdown("### Attrition Risk Output")
                
                # Check model package
                pipeline = model_pkg["pipeline"]
                explainer = model_pkg["explainer"]
                
                # Setup data structure matching training format
                input_data = pd.DataFrame([{
                    "Age": f_age,
                    "BusinessTravel": f_travel,
                    "DailyRate": f_daily,
                    "Department": f_dept,
                    "DistanceFromHome": f_distance,
                    "Education": f_edu,
                    "EducationField": f_field,
                    "EnvironmentSatisfaction": f_env_sat,
                    "Gender": f_gender,
                    "HourlyRate": f_hourly,
                    "JobInvolvement": f_job_inv,
                    "JobLevel": f_level,
                    "JobRole": f_role,
                    "JobSatisfaction": f_job_sat,
                    "MaritalStatus": f_marital,
                    "MonthlyIncome": f_income,
                    "MonthlyRate": f_rate,
                    "NumCompaniesWorked": f_num_comp,
                    "OverTime": f_ot,
                    "PercentSalaryHike": f_hike,
                    "PerformanceRating": f_perf,
                    "RelationshipSatisfaction": f_rel_sat,
                    "StockOptionLevel": f_stock,
                    "TotalWorkingYears": f_total_years,
                    "TrainingTimesLastYear": f_training,
                    "WorkLifeBalance": f_wlb,
                    "YearsAtCompany": f_years_co,
                    "YearsInCurrentRole": f_years_role,
                    "YearsSinceLastPromotion": f_years_promo,
                    "YearsWithCurrManager": f_years_mgr
                }])
                
                # Predict
                prob = pipeline.predict_proba(input_data)[0, 1]
                
                # Risk level card
                if prob >= 0.6:
                    risk_status = "HIGH RISK"
                    risk_class = "risk-high"
                    alert_func = st.error
                elif prob >= 0.3:
                    risk_status = "MEDIUM RISK"
                    risk_class = "risk-medium"
                    alert_func = st.warning
                else:
                    risk_status = "LOW RISK"
                    risk_class = "risk-low"
                    alert_func = st.success
                
                # Render results nicely
                st.markdown(f"""
                    <div class="kpi-card" style="border-left-color: {'#dc3545' if prob >= 0.6 else '#ffc107' if prob >= 0.3 else '#28a745'}">
                        <div class="kpi-title">Attrition Probability</div>
                        <div class="kpi-value">{prob:.1%}</div>
                        <div style="font-size: 1.2rem; margin-top: 8px;">Status: <span class="{risk_class}">{risk_status}</span></div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Explanations
                st.markdown("### 🧠 Key Attrition Drivers (SHAP Value Breakdown)")
                st.markdown("This chart shows the top factors pushing this employee towards leaving (red) or staying (blue).")
                
                try:
                    exp = get_individual_explanation(pipeline, explainer, input_data)
                    feat_exp = exp["feature_explanations"][:12]  # top 12 drivers
                    
                    df_shap = pd.DataFrame(feat_exp)
                    
                    # Add colors depending on positive/negative impact
                    df_shap["color"] = df_shap["shap_value"].apply(
                        lambda x: "#ef4444" if x > 0 else "#3b82f6"
                    )
                    
                    # Sort so highest pushes are at top
                    df_shap = df_shap.sort_values("shap_value", ascending=True)
                    
                    # Horizontal bar plot
                    fig, ax = plt.subplots(figsize=(7, 5))
                    bars = ax.barh(
                        df_shap["display_name"], df_shap["shap_value"], 
                        color=df_shap["color"], edgecolor="gray", height=0.6
                    )
                    
                    # Labels & grids
                    ax.axvline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.7)
                    ax.set_xlabel("Impact on Attrition Decision (SHAP Value)")
                    ax.grid(axis='x', linestyle='--', alpha=0.5)
                    
                    # Adjust text spacing
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    # Detailed report table
                    st.markdown("#### Feature Details")
                    df_details = pd.DataFrame(exp["feature_explanations"])
                    df_details["Impact Direction"] = df_details["shap_value"].apply(
                        lambda x: "Increases Risk (Leave)" if x > 0 else "Decreases Risk (Stay)"
                    )
                    st.dataframe(
                        df_details[["display_name", "raw_value", "Impact Direction", "shap_value"]]
                        .rename(columns={
                            "display_name": "Attribute", 
                            "raw_value": "Employee Value",
                            "shap_value": "SHAP Score"
                        }), 
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # PDF/CSV Report Generation
                    st.divider()
                    st.markdown("#### Download Scoring Report")
                    
                    # Convert to CSV
                    csv_buffer = io.StringIO()
                    df_details.to_csv(csv_buffer, index=False)
                    csv_bytes = csv_buffer.getvalue().encode("utf-8")
                    
                    st.download_button(
                        label="📥 Download Detailed CSV Report",
                        data=csv_bytes,
                        file_name=f"attrition_report_emp_prob_{prob:.2f}.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"Error computing SHAP values: {e}")
                    logger.error(f"Error calculating individual prediction explanation: {e}", exc_info=True)
