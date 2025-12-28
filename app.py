import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

data = {
    "height": [150, 155, 160, 165, 170, 175, 180, 160, 170, 165],
    "weight": [40, 45, 50, 55, 60, 65, 70, 80, 85, 90],
    "age":    [18, 20, 22, 24, 26, 28, 30, 32, 35, 40],
    "label":  [0, 0, 1, 1, 1, 1, 1, 2, 3, 3]
}

df = pd.DataFrame(data)
df["bmi"] = df["weight"] / (df["height"]/100)**2
X = df[["bmi", "age"]]
y = df["label"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

st.title("ML-Based Health Advisor 🧠🥗")
st.write("This app uses Random Forest and BMI for better accuracy")

height = st.number_input("Height (cm)", 100, 220)
weight = st.number_input("Weight (kg)", 30, 200)
age = st.number_input("Age", 1, 100)

if st.button("Check Health"):
    bmi = weight / (height/100)**2
    prediction = model.predict([[bmi, age]])[0]

    if prediction == 0:
        st.error("UNDERWEIGHT")
        st.write("🍽 Eat more calories:")
        st.write("- Milk, rice, banana")
        st.write("- Nuts & paneer")
    elif prediction == 1:
        st.success("HEALTHY")
        st.write("🥗 Maintain balanced diet:")
        st.write("- Fruits, vegetables")
        st.write("- Dal, eggs")
    elif prediction == 2:
        st.warning("OVERWEIGHT")
        st.write("🥦 Eat light & healthy:")
        st.write("- Green vegetables")
        st.write("- Avoid sugar & junk")
    else:
        st.error("OBESE")
        st.write("⚠️ Strict diet needed:")
        st.write("- High fiber food")
        st.write("- Daily walking & exercise")

    st.info("This is an ML-based prediction, not medical advice.")
