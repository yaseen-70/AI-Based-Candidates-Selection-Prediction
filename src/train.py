import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import f1_score



# ---------------------------------
# Paths
# ---------------------------------

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


PROJECT_DIR = os.path.abspath(
    os.path.join(
        BASE_DIR,
        ".."
    )
)


DATA_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "raw",
    "candidate_selection_dataset.csv"
)


MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "best_model.joblib"
)


SCALER_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "scaler.joblib"
)


ENCODER_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "encoder.joblib"
)


TARGET_ENCODER_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "target_encoder.joblib"
)



# ---------------------------------
# Load Dataset
# ---------------------------------

df = pd.read_csv(
    DATA_PATH
)

print(df["hiring_decision"].value_counts())


print(
    "Dataset loaded successfully"
)



# ---------------------------------
# Features and Target
# ---------------------------------

X = df.drop(
    "hiring_decision",
    axis=1
)


y = df["hiring_decision"]



# ---------------------------------
# Encode Categorical Features
# ---------------------------------

categorical_features = [
    "gender",
    "education_level",
    "recruitment_strategy"
]


encoders = {}


for col in categorical_features:

    encoder = LabelEncoder()

    X[col] = encoder.fit_transform(
        X[col]
    )

    encoders[col] = encoder



# ---------------------------------
# Encode Target
# ---------------------------------

target_encoder = LabelEncoder()


y = target_encoder.fit_transform(
    y
)



# ---------------------------------
# Train Test Split
# ---------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)



# ---------------------------------
# Scaling
# ---------------------------------

scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(
    X_train
)


X_test_scaled = scaler.transform(
    X_test
)



# ---------------------------------
# Define Models
# ---------------------------------

models = {


    "Logistic Regression":

        LogisticRegression(
            max_iter=1000
        ),



    "Decision Tree":

        DecisionTreeClassifier(
            random_state=42
        ),



    "Random Forest":

        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),



    "XGBoost":

        XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )

}



# ---------------------------------
# Train and Compare Models
# ---------------------------------

results = {}

trained_models = {}



for name, model in models.items():


    if name == "Logistic Regression":


        model.fit(
            X_train_scaled,
            y_train
        )


        prediction = model.predict(
            X_test_scaled
        )


    else:


        model.fit(
            X_train,
            y_train
        )


        prediction = model.predict(
            X_test
        )



    score = f1_score(
        y_test,
        prediction
    )


    results[name] = score


    trained_models[name] = model



    print(
        name,
        "F1 Score:",
        round(score, 4)
    )



# ---------------------------------
# Select Best Model
# ---------------------------------

best_model_name = max(
    results,
    key=results.get
)


best_model = trained_models[
    best_model_name
]


print(
    "\nBest Model:",
    best_model_name
)



# ---------------------------------
# Save Artifacts
# ---------------------------------

joblib.dump(
    best_model,
    MODEL_PATH
)


joblib.dump(
    scaler,
    SCALER_PATH
)


joblib.dump(
    encoders,
    ENCODER_PATH
)


joblib.dump(
    target_encoder,
    TARGET_ENCODER_PATH
)



print(
    "\nAll artifacts saved successfully"
)