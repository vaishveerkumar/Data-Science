# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st

# Load data function
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/vaishveerkumar/Data-Science/main/CAPSTONE_PROJECT/heart.csv"
    data = pd.read_csv(url)
    data = data.replace('?', np.nan)  
    data.dropna(inplace=True) 
    data['target'] = (data['target'] > 0).astype(int)
    return data

# Train the model with selected features
def train_model(X_train, y_train):
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    return knn

# Flashing banner effect CSS 
css = """
.flash-banner {
    animation: flash 2s infinite alternate;
    padding: 10px;
}

@keyframes flash {
    from { opacity: 1; }
    to { opacity: 0; }
}

.flash-text {
    font-size: 18px;
    font-weight: bold;
    color: white;
    text-align: center;
}
"""

# Streamlit app
def main():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://meigsindypress.com/wp-content/uploads/2022/02/qtq80-9zIRqw-1024x576.jpeg");
            background-size: cover;
            background-position: center center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title('HEART DISEASE PREDICTION APP')
    st.markdown(
        """
        <style>
            .big-font {
                font-size:20px !important;
                font-weight: bold;
                color: #EE6C4D;  # Color of the text
            }
            .stButton > button {
                color: white;
                background-color: #264653;
                border-radius: 5px 5px;
                padding: 10px 24px;
                margin: 10px 1px;
                cursor: pointer;
                font-size: 18px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
    """
    <div class="centered-text">
        <br></br> <!-- Blank line or new line -->
        <p class="big-font">Predict Your Risk Of Heart Disease Using Machine Learning</p>
    </div>
    """,
    unsafe_allow_html=True
    )

    # Load data
    data = load_data()

    # Split data into X and y
    X = data.drop("target", axis=1)
    y = data["target"]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling the data
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model
    model = train_model(X_train_scaled, y_train)

    st.sidebar.markdown('<p class="big-font">Please Make Appropriate Selections Below </p>', unsafe_allow_html=True)
    
    with st.sidebar.form(key='user_input_form'):
        age = st.slider('Age', min_value=1, max_value=100, value=50, step=1)
        sex = st.selectbox('Sex', ['Male', 'Female'])
        cp = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
        trestbps = st.slider('Resting Blood Pressure (mm Hg)', min_value=90, max_value=200, value=120, step=1)
        chol = st.slider('Serum Cholesterol (mg/dl)', min_value=100, max_value=600, value=200, step=1)
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['True', 'False'])
        restecg = st.selectbox('Resting Electrocardiographic Results', ['Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy'])
        thalach = st.slider('Maximum Heart Rate Achieved', min_value=70, max_value=220, value=150, step=1)
        exang = st.selectbox('Exercise Induced Angina', ['Yes', 'No'])
        oldpeak = st.slider('ST Depression Induced by Exercise Relative to Rest', min_value=0.0, max_value=6.2, value=1.0, step=0.1)
        slope = st.selectbox('Slope of the Peak Exercise ST Segment', ['Upsloping', 'Flat', 'Downsloping'])
        ca = st.selectbox('Number of Major Vessels Colored by Flourosopy', ['0', '1', '2', '3'])
        thal = st.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect'])
        submit_button = st.form_submit_button(label='Predict')

    if submit_button:
        # Clear previous result
        st.empty()
        
        # Preprocess user input
        sex = 1 if sex == 'Female' else 0
        cp = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3}[cp]
        fbs = 1 if fbs == 'True' else 0
        restecg = {'Normal': 0, 'ST-T wave abnormality': 1, 'Left ventricular hypertrophy': 2}[restecg]
        exang = 1 if exang == 'Yes' else 0
        slope = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}[slope]
        ca = int(ca)
        thal = {'Normal': 0, 'Fixed Defect': 1, 'Reversible Defect': 2}[thal]

        # Prepare user input data
        user_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        user_input_scaled = scaler.transform(user_input)

        # Make prediction
        prediction = model.predict(user_input_scaled)

        # Make prediction probabilities
        prediction_proba = model.predict_proba(user_input_scaled)[0]

        # Display prediction
        st.subheader('PREDICTION RESULT')

        # Determine result text and color
        if prediction[0] == 0:
            result_text = 'The model predicts that the patient does not have heart disease'
            result_color = 'green'
        else:
            result_text = 'The model predicts that the patient has heart disease!'
            result_color = 'red'

        # Flashing banner effect HTML
        flash_banner_html = f"""
        <div class="flash-banner" style="background-color: {result_color};">
            <p class="flash-text">{result_text}</p>
        </div>
        """

        # Write CSS to Streamlit
        st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
        # Display flashing banner HTML
        st.markdown(flash_banner_html, unsafe_allow_html=True)

        # Add disclaimer
        
        st.sidebar.markdown(
        """
        <div class="disclaimer">
            <p style="font-size: 14px; color: gray; text-align: center;">Note: Predictions for male patients may be less accurate due to limited data availability during model training.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    main()
