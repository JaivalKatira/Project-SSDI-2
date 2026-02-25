import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    r2_score,
    mean_squared_error,
    confusion_matrix
)

st.set_page_config(page_title="ML Dashboard", layout="centered")
st.title("📊 Machine Learning Streamlit App")

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -------------------------
    # Target Selection
    # -------------------------
    target_column = st.selectbox("Select Target Column", df.columns)

    # -------------------------
    # Feature Selection
    # -------------------------
    feature_columns = st.multiselect(
        "Select Feature Columns for Training",
        options=[col for col in df.columns if col != target_column],
        default=[col for col in df.columns if col != target_column]
    )

    # -------------------------
    # Problem Type
    # -------------------------
    problem_type = st.radio(
        "Select Problem Type",
        ["Classification", "Regression"]
    )

    # -------------------------
    # Train-Test Split
    # -------------------------
    test_size_percent = st.slider(
        "Select Test Size (%)",
        min_value=10,
        max_value=50,
        value=20
    )

    test_size = test_size_percent / 100

    # -------------------------
    # Random State
    # -------------------------
    random_state = st.number_input(
        "Enter Random State",
        min_value=0,
        value=42
    )

    if st.button("Train Model"):

        if len(feature_columns) == 0:
            st.error("❌ Please select at least one feature column.")
            st.stop()

        X = df[feature_columns]
        y = df[target_column]

        # -----------------------------
        # Classification Safety Check
        # -----------------------------
        if problem_type == "Classification":

            unique_values = y.nunique()

            if np.issubdtype(y.dtype, np.number) and unique_values > 20:
                st.error(
                    "❌ Selected target appears continuous (like ID). "
                    "Choose a categorical variable for classification."
                )
                st.stop()

        # -----------------------------
        # Handle Missing Values
        # -----------------------------
        numeric_cols = X.select_dtypes(include=np.number).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())

        categorical_cols = X.select_dtypes(exclude=np.number).columns
        for col in categorical_cols:
            X[col] = X[col].fillna(X[col].mode()[0])

        # Encode categorical features
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

        # Encode target if needed
        if problem_type == "Classification" and y.dtype == "object":
            y = LabelEncoder().fit_transform(y)

        # -----------------------------
        # Train Test Split
        # -----------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )

        # -----------------------------
        # Model Selection
        # -----------------------------
        if problem_type == "Classification":
            model = GaussianNB()
        else:
            model = LinearRegression()

        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        st.subheader("📈 Results")

        # -----------------------------
        # Classification Results
        # -----------------------------
        if problem_type == "Classification":

            train_acc = accuracy_score(y_train, train_pred)
            test_acc = accuracy_score(y_test, test_pred)

            st.write("### Classification Results")
            st.write("Training Accuracy:", round(train_acc, 4))
            st.write("Test Accuracy:", round(test_acc, 4))

            # Confusion Matrix
            cm = confusion_matrix(y_test, test_pred)
            st.write("### Confusion Matrix")
            st.write(cm)

        # -----------------------------
        # Regression Results
        # -----------------------------
        else:

            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)

            train_mse = mean_squared_error(y_train, train_pred)
            test_mse = mean_squared_error(y_test, test_pred)

            st.write("### Regression Results")
            st.write("Training R²:", round(train_r2, 4))
            st.write("Test R²:", round(test_r2, 4))
            st.write("Training MSE:", round(train_mse, 4))
            st.write("Test MSE:", round(test_mse, 4))

else:
    st.info("Upload a dataset to begin.")