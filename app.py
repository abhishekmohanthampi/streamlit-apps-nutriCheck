import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("obesity_data.csv")

le = LabelEncoder()
df["Label"] = le.fit_transform(df["Label"])

df["bmi"] = df["BMI"]

X = df[["bmi", "Age"]]
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.title("AI Health Advisor with Real Data 🧠🥗")
st.write(f"Model accuracy on test data: {acc:.2f}")

height = st.number_input("Height (cm)", 100, 220)
weight = st.number_input("Weight (kg)", 30, 200)
age = st.number_input("Age (years)", 1, 100)

if st.button("Get Health Advice"):
    bmi = weight / (height/100)**2

    pred = model.predict([[bmi, age]])[0]
    label = le.inverse_transform([pred])[0]

    st.write(f"### 🩺 Based on BMI = {bmi:.2f}")
    st.write(f"**Category:** {label}")

    if label == "Underweight":
        st.write("🍽 Eat more calories and protein: milk, nuts, rice, bananas")
    elif label == "Normal Weight":
        st.write("🥗 Maintain balanced diet: fruits, vegetables, dal, eggs")
    elif label == "Overweight":
        st.write("🥦 Eat lighter: green vegetables, avoid sugar & junk")
    else:
        st.write("⚠️ Focus on strict diet & exercise: high fiber, walking daily")

    st.info("This prediction uses a real dataset and machine learning.")
