import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Healthcare ML Dashboard", layout="wide")

st.title("🏥 Healthcare ML Dashboard")

# --------------------
# Tabs
# --------------------
tab1, tab2 = st.tabs(["🔎 Explore", "🤖 Model"])

# --------------------
# EXPLORE TAB
# --------------------
with tab1:
    st.header("📊 Explore Dataset")
    st.markdown(
        "This section allows you to **upload and explore healthcare data**. "
        "You can preview the dataset, check summary statistics, and visualize correlations between features."
    )

    uploaded_file = st.file_uploader("Upload Healthcare Dataset (CSV)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview")
        st.write(df.head())
        st.write("Shape:", df.shape)

        st.write("### Summary Statistics")
        st.write(df.describe())

        st.write("### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)


# --------------------
# MODEL TAB
# --------------------
with tab2:
    st.header("🤖 Model Evaluation & Prediction")
    st.markdown(
        "This section trains multiple **regression models** on your dataset, evaluates their performance, "
        "and allows you to make predictions based on new patient input."
    )

    uploaded_file = st.file_uploader("Upload Dataset for Modeling (CSV)", type=["csv"], key="model_upload")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # Assume last column is target
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Define models
        reg_models = {
            "Baseline": DummyRegressor(strategy="mean"),
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        }

        reg_results = []
        trained_models = {}

        for name, model in reg_models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            mae = mean_absolute_error(y_test, preds)
            rmse = mean_squared_error(y_test, preds) ** 0.5
            r2 = r2_score(y_test, preds)
            cv = cross_val_score(model, X, y, cv=5, scoring="r2").mean()

            reg_results.append([name, mae, rmse, r2, cv])
            trained_models[name] = model

        reg_results_df = pd.DataFrame(
            reg_results, columns=["Model", "MAE", "RMSE", "R²", "CV R²"]
        )

        st.write("### Regression Results")
        st.dataframe(reg_results_df, use_container_width=True)

        # Choose model for prediction
        model_choice = st.selectbox("Select a model for prediction:", list(trained_models.keys()))
        chosen_model = trained_models[model_choice]

        st.subheader("🧪 Make a Prediction")
        st.markdown("Enter values for the features below to generate a prediction:")

        input_values = {}
        for col in X.columns:
            if np.issubdtype(X[col].dtype, np.number):
                input_values[col] = st.number_input(f"{col}", value=float(X[col].mean()))
            else:
                options = X[col].unique().tolist()
                input_values[col] = st.selectbox(f"{col}", options)

        # Convert input to dataframe
        input_df = pd.DataFrame([input_values])
        st.write("Input Data:", input_df)

        if st.button("Predict"):
            try:
                prediction = chosen_model.predict(input_df)
                st.success(f"Predicted Value: {prediction[0]:.2f}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
