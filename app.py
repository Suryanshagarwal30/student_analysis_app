# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import shap
# import matplotlib.pyplot as plt
# import seaborn as sns
# from utils import load_data, prepare_features, train_model

# st.set_page_config(page_title="🎓 Student Dashboard", layout="wide")

# @st.cache_data
# def get_data():
#     df = load_data()
#     X, y = prepare_features(df)
#     model, X_test = train_model(X, y)
#     return df, X, y, model, X_test

# df, X, y, model, X_test = get_data()

# st.title("📊 Engineering Student Performance Dashboard")

# # --- Sidebar ---
# with st.sidebar:
#     st.header("🔍 Filter & Predict")

#     # Filter
#     branch = st.multiselect("Branch", df['Branch'].unique(), default=df['Branch'].unique())
#     year = st.multiselect("Year", sorted(df['Year'].unique()), default=sorted(df['Year'].unique()))
#     filtered_df = df[df['Branch'].isin(branch) & df['Year'].isin(year)]

#     # Prediction Inputs
#     st.markdown("---")
#     st.subheader("🎯 Predict High CGPA")
#     ia = st.slider("IA Marks", 0, 50, 25)
#     att = st.slider("Attendance (%)", 0.0, 100.0, 75.0)
#     student_year = st.selectbox("Year", [1, 2, 3, 4])
#     interest_code = st.number_input("Interest Code (0–4)", 0, 4)

#     if st.button("Predict"):
#         input_data = pd.DataFrame([[ia, att, student_year, interest_code] + [0]*10])
#         prediction = model.predict(input_data)[0]
#         if prediction == 1:
#             st.success("🎉 Likely to Score High CGPA")
#         else:
#             st.error("⚠️ Risk of Low CGPA")

# # --- Tabs Layout ---
# tab1, tab2, tab3 = st.tabs(["📊 Visual Insights", "📉 Correlation", "🧠 SHAP Explainability"])

# with tab1:
#     st.subheader("Average CGPA by Branch")
#     fig = px.bar(filtered_df.groupby("Branch")["CGPA"].mean().reset_index(), x="Branch", y="CGPA")
#     st.plotly_chart(fig, use_container_width=True)

#     st.subheader("Attendance vs CGPA")
#     st.plotly_chart(px.scatter(filtered_df, x="Attendance (%)", y="CGPA", color="Year"), use_container_width=True)

# with tab2:
#     st.subheader("Correlation Heatmap")
#     corr = filtered_df[['IA Marks', 'CGPA', 'Attendance (%)', 'Year']].corr()
#     fig, ax = plt.subplots()
#     sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
#     st.pyplot(fig)

# with tab3:
#     st.subheader("Feature Impact (SHAP)")
#     shap.initjs()
#     try:
#         explainer = shap.TreeExplainer(model)
#         shap_values = explainer.shap_values(X_test[:100])
#         st.write("Showing SHAP values for 100 students")
#         shap.summary_plot(shap_values[1], X_test[:100], show=False)
#         st.pyplot(bbox_inches='tight')
#     except Exception as e:
#         st.error(f"SHAP error: {e}")


# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import shap
# import matplotlib.pyplot as plt
# import seaborn as sns
# from utils import load_data, prepare_features, train_model

# st.set_page_config(page_title="🎓 Student Dashboard", layout="wide")

# @st.cache_data
# def get_data():
#     df = load_data()
#     X, y = prepare_features(df)
#     model, X_test = train_model(X, y)
#     return df, X, y, model, X_test

# df, X, y, model, X_test = get_data()

# # Sidebar Filters
# st.sidebar.header("🎛️ Filters")
# branches = st.sidebar.multiselect("Select Branch(es)", df['Branch'].unique(), default=df['Branch'].unique())
# years = st.sidebar.multiselect("Select Year(s)", sorted(df['Year'].unique()), default=sorted(df['Year'].unique()))
# filtered_df = df[df['Branch'].isin(branches) & df['Year'].isin(years)]

# # KPIs
# st.title("📊 Engineering Student Performance Dashboard")
# col1, col2, col3 = st.columns(3)
# col1.metric("Average CGPA", f"{filtered_df['CGPA'].mean():.2f}")
# col2.metric("Total Students", len(filtered_df))
# col3.metric("High CGPA %", f"{(filtered_df['High_CGPA'].mean() * 100):.1f}%")

# # Layout with tabs
# tab1, tab2, tab3, tab4 = st.tabs(["📈 Charts", "📉 Correlation", "🧠 SHAP", "🎯 Predict"])

# with tab1:
#     st.subheader("Performance by Branch")
#     col1, col2 = st.columns(2)

#     with col1:
#         st.plotly_chart(
#             px.bar(filtered_df.groupby("Branch")["CGPA"].mean().reset_index(), x="Branch", y="CGPA", color="Branch"),
#             use_container_width=True
#         )

#     with col2:
#         st.plotly_chart(
#             px.scatter(filtered_df, x="Attendance (%)", y="CGPA", color="Year", hover_data=["Branch"]),
#             use_container_width=True
#         )

# with tab2:
#     st.subheader("Correlation Heatmap")
#     corr = filtered_df[['IA Marks', 'CGPA', 'Attendance (%)', 'Year']].corr()
#     fig, ax = plt.subplots()
#     sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
#     st.pyplot(fig)

# with tab3:
#     st.subheader("SHAP Feature Importance")
#     shap.initjs()
#     try:
#         explainer = shap.TreeExplainer(model)
#         shap_values = explainer.shap_values(X_test[:100])
#         st.write("Showing SHAP summary plot for 100 students")
#         shap.summary_plot(shap_values[1], X_test[:100], show=False)
#         st.pyplot(bbox_inches="tight")
#     except Exception as e:
#         st.error(f"SHAP failed: {e}")

# with tab4:
#     st.subheader("🎯 Predict Student CGPA Category")

#     with st.form("prediction_form"):
#         ia = st.slider("IA Marks", 0, 50, 25)
#         att = st.slider("Attendance (%)", 0.0, 100.0, 75.0)
#         student_year = st.selectbox("Year", [1, 2, 3, 4])
#         interest_code = st.number_input("Interest Code (0–4)", 0, 4)

#         submitted = st.form_submit_button("Predict")

#         if submitted:
#             input_data = pd.DataFrame([[ia, att, student_year, interest_code] + [0]*10])
#             prediction = model.predict(input_data)[0]
#             with st.expander("🔎 Prediction Result"):
#                 st.write("✅ High CGPA Likely" if prediction == 1 else "⚠️ Risk of Low CGPA")


# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data, prepare_features, train_model

st.set_page_config(page_title="🎓 Student Dashboard", layout="wide")

# Load and cache data
@st.cache_data
def get_data():
    df = load_data()
    X, y = prepare_features(df)
    model, X_test = train_model(X, y)
    return df, X, y, model, X_test

df, X, y, model, X_test = get_data()

# Sidebar Filters
st.sidebar.header("🎛️ Filters")
branches = st.sidebar.multiselect("Select Branch(es)", df['Branch'].unique(), default=df['Branch'].unique())
years = st.sidebar.multiselect("Select Year(s)", sorted(df['Year'].unique()), default=sorted(df['Year'].unique()))

# Filter original df and also X, y
filtered_df = df[df['Branch'].isin(branches) & df['Year'].isin(years)]
filtered_X = X.loc[filtered_df.index]
filtered_y = y.loc[filtered_df.index]

# KPIs
st.title("📊 Engineering Student Performance Dashboard")
col1, col2, col3 = st.columns(3)
col1.metric("Average CGPA", f"{filtered_df['CGPA'].mean():.2f}")
col2.metric("Total Students", len(filtered_df))
col3.metric("High CGPA %", f"{(filtered_df['High_CGPA'].mean() * 100):.1f}%")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Charts", "📉 Correlation", "🧠 SHAP", "🎯 Predict"])

# Tab 1: Charts
with tab1:
    st.subheader("Performance by Branch and Attendance")

    col1, col2 = st.columns(2)

    with col1:
        branch_cgpa = filtered_df.groupby("Branch")["CGPA"].mean().reset_index()
        st.plotly_chart(
            px.bar(branch_cgpa, x="Branch", y="CGPA", color="Branch"),
            use_container_width=True
        )

    with col2:
        st.plotly_chart(
            px.scatter(filtered_df, x="Attendance (%)", y="CGPA", color="Year", hover_data=["Branch"]),
            use_container_width=True
        )

# Tab 2: Heatmap
with tab2:
    st.subheader("Correlation Heatmap")
    corr = filtered_df[['IA Marks', 'CGPA', 'Attendance (%)', 'Year']].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Tab 3: SHAP
with tab3:
    st.subheader("SHAP Feature Importance (Model Explanation)")
    try:
        shap.initjs()
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(filtered_X[:100])

        st.write("Showing SHAP summary plot for top 100 filtered students")

        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[1], filtered_X[:100], show=False)
        else:
            shap.summary_plot(shap_values, filtered_X[:100], show=False)

        st.pyplot(bbox_inches="tight")
    except Exception as e:
        st.error(f"SHAP error: {e}")

# Tab 4: Predict
with tab4:
    st.subheader("🎯 Predict Student CGPA Category")
    st.write("Fill in student details to predict if CGPA ≥ 8.0")

    with st.form("prediction_form"):
        ia = st.slider("IA Marks", 0, 50, 25)
        att = st.slider("Attendance (%)", 0.0, 100.0, 75.0)
        student_year = st.selectbox("Year", [1, 2, 3, 4])
        interest_code = st.number_input("Interest Code (0–4)", 0, 4)

        # Simulate Skills vector as all zeros
        dummy_skills = [0] * (X.shape[1] - 4)
        input_data = pd.DataFrame([[ia, att, student_year, interest_code] + dummy_skills], columns=X.columns)

        submitted = st.form_submit_button("Predict")

        if submitted:
            prediction = model.predict(input_data)[0]
            with st.expander("🔎 Prediction Result"):
                st.success("✅ High CGPA Likely (≥ 8.0)" if prediction == 1 else "⚠️ Low CGPA Risk (< 8.0)")
