import joblib
import pandas as pd


CATEGORICAL_FEATURES = [
    "gender",
    "education_level",
    "recruitment_strategy"
]


def load_artifacts(
    encoder_path="../models/encoder.joblib",
    scaler_path="../models/scaler.joblib",
    target_encoder_path="../models/target_encoder.joblib"
):
    """
    Load saved preprocessing artifacts
    """

    encoders = joblib.load(
        encoder_path
    )

    scaler = joblib.load(
        scaler_path
    )

    target_encoder = joblib.load(
        target_encoder_path
    )

    return encoders, scaler, target_encoder



def preprocess_input(
    data,
    encoders,
    scaler
):
    """
    Apply encoding and scaling
    on new candidate data
    """

    data = data.copy()


    # Encode categorical columns

    for col in CATEGORICAL_FEATURES:

        data[col] = encoders[col].transform(
            data[col]
        )


    # Scale all features

    data_scaled = scaler.transform(
        data
    )


    return data_scaled



def decode_prediction(
    prediction,
    target_encoder
):
    """
    Convert model output back
    to original hiring decision label
    """

    result = target_encoder.inverse_transform(
        prediction
    )

    return result

































# import pandas as pd
# import joblib


# def load_artifacts(
#     encoder_path="../models/encoder.joblib",
#     scaler_path="../models/scaler.joblib"
# ):
#     """
#     Load saved encoders and scaler
#     """

#     encoders = joblib.load(encoder_path)
#     scaler = joblib.load(scaler_path)

#     return encoders, scaler



# def preprocess_input(
#     data,
#     encoders,
#     scaler
# ):
#     """
#     Apply preprocessing on new candidate data
#     """

#     data = data.copy()


#     # Categorical columns
#     categorical_features = [
#         "gender",
#         "education_level",
#         "recruitment_strategy"
#     ]


#     # Apply encoding
#     for col in categorical_features:
#         data[col] = encoders[col].transform(
#             data[col]
#         )


#     # Scale features
#     data_scaled = scaler.transform(
#         data
#     )


#     return data_scaled



