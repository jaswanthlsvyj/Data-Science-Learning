import streamlit as st
import joblib
import numpy as np

# Load the model and label encoders
model = joblib.load('titanic_model.pkl')
label_encoders = {}
for column in ['sex', 'embarked', 'class', 'who']:
    label_encoders[column] = joblib.load(f'label_encoder_{column}.pkl')

# Define the Streamlit app
def main():
    st.title("Titanic Survival Prediction")

    # Input fields for user data
    pclass = st.selectbox("Pclass", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.number_input("Age", min_value=1, max_value=80, value=30)
    sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=8, value=0)
    parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=6, value=0)
    fare = st.number_input("Fare", min_value=0.0, value=15.0)
    embarked = st.selectbox("Embarked", ["C", "Q", "S"])

    # Preprocess input data
    sex = label_encoders['sex'].transform([sex])[0]
    embarked = label_encoders['embarked'].transform([embarked])[0]

    # Create input array
    input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])

    # Predict
    if st.button("Predict"):
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.success("The passenger would have survived.")
        else:
            st.error("The passenger would not have survived.")

if __name__ == '__main__':
    main()
