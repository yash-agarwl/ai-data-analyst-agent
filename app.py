import streamlit as st
import pandas as pd
from data_processor import load_data, clean_data
from agent import analyze_dataset

st.title("AI Data Analyst Agent")

file = st.file_uploader("Upload dataset")

if file:

    df = load_data(file)

    st.subheader("Dataset Health Report")

    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])
    st.write("Duplicate Rows:", df.duplicated().sum())

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    numeric_cols = df.select_dtypes(include=['int64','float64']).columns

    st.subheader("Numeric Column Summary")
    st.write(df[numeric_cols].describe())

    st.subheader("Correlation Matrix")
    st.dataframe(df[numeric_cols].corr())

    st.subheader("Target Column Detection")

    possible_targets = []

    for col in df.columns:
        if df[col].nunique() <= 10:
            possible_targets.append(col)

    st.write("Possible Target Columns:", possible_targets)

    target = st.selectbox("Select Target Column", possible_targets)

    from sklearn.model_selection import train_test_split

    X = df.drop(target, axis=1)
    y = df[target]

    # keep only numeric columns for simplicity
    X = X.select_dtypes(include=['int64','float64'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier()

    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    st.subheader("Model Performance")

    st.write("Accuracy:", accuracy)