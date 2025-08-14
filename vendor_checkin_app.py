import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Hospital Vendor Check-in Predictor", layout="wide")

# 1. Simulate vendor check-in data
np.random.seed(42)
dates = pd.date_range(start="2024-01-01", end="2024-04-30", freq="D")
vendors = [f"Vendor_{i}" for i in range(1, 21)]
data = []

for date in dates:
    for vendor in vendors:
        freq = np.random.choice([0, 1], p=[0.7, 0.3])  # 30% chance of check-in
        department = np.random.choice(["Pharmacy", "Radiology", "Maintenance", "IT", "Surgery"])
        purpose = np.random.choice(["Delivery", "Repair", "Meeting", "Inspection"])
        data.append([vendor, date, department, purpose, freq])

df = pd.DataFrame(data, columns=["Vendor", "Date", "Department", "Purpose", "CheckedIn"])

# 2. Feature engineering
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

X = pd.get_dummies(df[['DayOfWeek', 'IsWeekend', 'Department', 'Purpose']], drop_first=True)
y = df['CheckedIn']

# 3. Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# 4. Streamlit UI
st.title("Vendor Check-in Prediction Dashboard")

# Visualization
st.subheader("Daily Vendor Check-ins")
daily_counts = df.groupby("Date")["CheckedIn"].sum().reset_index()
fig, ax = plt.subplots(figsize=(12, 4))
sns.lineplot(data=daily_counts, x="Date", y="CheckedIn", ax=ax)
ax.set_ylabel("Number of Vendors Checked In")
st.pyplot(fig)

# Model accuracy
st.markdown(f"**Model Accuracy:** {acc:.2f}")

# 5. Prediction form
st.subheader("Predict Vendor Check-in")
with st.form("prediction_form"):
    day_of_week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    is_weekend = 1 if day_of_week in ["Saturday", "Sunday"] else 0
    department = st.selectbox("Department", df["Department"].unique())
    purpose = st.selectbox("Purpose", df["Purpose"].unique())
    submit = st.form_submit_button("Predict")

if submit:
    input_data = pd.DataFrame([{
        "DayOfWeek": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day_of_week),
        "IsWeekend": is_weekend,
        **{f"Department_{department}": 1},
        **{f"Purpose_{purpose}": 1}
    }])

    # Ensure all dummy columns match training set
    for col in X.columns:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[X.columns]  # Reorder

    prediction = model.predict(input_data)[0]
    st.success(f"Prediction: Vendor will {'Check In' if prediction == 1 else 'Not Check In'}")

