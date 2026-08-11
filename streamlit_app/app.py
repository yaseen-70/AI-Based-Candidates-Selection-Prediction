import streamlit as st
import pandas as pd
import joblib
import os


# -----------------------------
# Paths
# -----------------------------

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "..",
        "models"
    )
)


# -----------------------------
# Load Model and Preprocessors
# -----------------------------

@st.cache_resource
def load_assets():

    model = joblib.load(
        os.path.join(
            MODEL_DIR,
            "best_model.joblib"
        )
    )

    encoders = joblib.load(
        os.path.join(
            MODEL_DIR,
            "encoder.joblib"
        )
    )

    scaler = joblib.load(
        os.path.join(
            MODEL_DIR,
            "scaler.joblib"
        )
    )

    target_encoder = joblib.load(
        os.path.join(
            MODEL_DIR,
            "target_encoder.joblib"
        )
    )

    return model, encoders, scaler, target_encoder



model, encoders, scaler, target_encoder = load_assets()



# -----------------------------
# Preprocessing Functions
# -----------------------------

def preprocess_input(
    data,
    encoders,
    scaler
):

    data = data.copy()


    categorical_features = [
        "gender",
        "education_level",
        "recruitment_strategy"
    ]


    for col in categorical_features:

        data[col] = encoders[col].transform(
            data[col]
        )


    # data_scaled = scaler.transform(
    #     data
    # )


    return data



def decode_prediction(
    prediction,
    target_encoder
):

    return target_encoder.inverse_transform(
        prediction
    )



# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="AI Candidate Selection Prediction",
    layout="centered"
)


st.title(
    "AI Based Candidate Selection Prediction"
)


st.write(
    "Enter candidate details to predict hiring decision."
)



# -----------------------------
# User Inputs
# -----------------------------

age = st.number_input(
    "Age",
    min_value=18,
    max_value=60,
    value=25
)


gender = st.selectbox(
    "Gender",
    [
        "Male",
        "Female"
    ]
)


education_level = st.selectbox(
    "Education Level",
    [
        "Bachelor's (Type 1)",
        "Bachelor's (Type 2)",
        "Master's",
        "PhD"
    ]
)


experience_years = st.number_input(
    "Experience Years",
    min_value=0,
    max_value=15,
    value=0
)


previous_companies = st.number_input(
    "Previous Companies",
    min_value=0,
    max_value=5,
    value=0
)


distance_from_company = st.number_input(
    "Distance From Company",
    min_value=0,
    max_value=500,
    value=0
)


interview_score = st.slider(
    "Interview Score",
    0,
    100,
    0
)


skill_score = st.slider(
    "Skill Score",
    0,
    100,
    0
)


personality_score = st.slider(
    "Personality Score",
    0,
    100,
    0
)


recruitment_strategy = st.selectbox(
    "Recruitment Strategy",
    [
        "Aggressive",
        "Moderate",
        "Constructive"
    ]
)



# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict"):


    input_data = pd.DataFrame({

        "age": [age],

        "gender": [gender],

        "education_level": [education_level],

        "experience_years": [
            experience_years
        ],

        "pevious_companies": [
            previous_companies
        ],

        "distance_from_company": [
            distance_from_company
        ],

        "interview_score": [
            interview_score
        ],

        "skill_score": [
            skill_score
        ],

        "personality_score": [
            personality_score
        ],

        "recruitment_strategy": [
            recruitment_strategy
        ]

    })


    processed_data = preprocess_input(
        input_data,
        encoders,
        scaler
    )

    st.write("Processed Input : ")
    st.write(processed_data)


    prediction = model.predict(
        processed_data
    )

    # st.write("RAw Prediction : ", prediction)
    # st.write("Target Classes : ", target_encoder.classes_)



    result = decode_prediction(
        prediction,
        target_encoder
    )


    if result[0] == "Selected":

        st.success(
            "Candidate Selected"
        )

    else:

        st.error(
            "Candidate Not Selected"
        )