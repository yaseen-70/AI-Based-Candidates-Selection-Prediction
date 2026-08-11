# AI Based Candidate Selection Prediction

## Project Overview

An Artificial Intelligence based candidate selection prediction system that predicts whether a candidate should be selected or not based on education, experience, interview performance, technical skills, personality score, and recruitment strategy.

This project uses Machine Learning classification algorithms and provides a Streamlit web application for real-time candidate selection prediction.

---

## Project Features

- Data preprocessing and encoding
- Exploratory Data Analysis (EDA)
- Candidate selection prediction
- Multiple machine learning model comparison
- Feature importance analysis
- Model evaluation
- Confusion matrix and ROC-AUC analysis
- Saved machine learning model
- Interactive Streamlit web application

---

## Project Structure

```

AI-Candidate-Selection-Prediction/

│
├── data/
│   └── raw/
│       └── candidate_selection_dataset.csv
│
├── models/
│   ├── best_model.joblib
│   ├── scaler.joblib
│   ├── encoder.joblib
│   └── target_encoder.joblib
│
├── notebook/
│   └── main_analysis.ipynb
│
├── reports/
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   ├── confusion_matrix.png
│   └── roc_auc_curve.png
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluation.py
│
├── streamlit_app/
│   └── app.py
│
├── README.md
├── requirements.txt
└── .gitignore

```

---

## Dataset Features

| Feature | Description |
|---|---|
| age | Candidate age |
| gender | Candidate gender |
| education_level | Candidate education qualification |
| experience_years | Years of experience |
| pevious_companies | Number of previous companies |
| distance_from_company | Distance from company |
| interview_score | Interview performance score |
| skill_score | Technical skill score |
| personality_score | Personality assessment score |
| recruitment_strategy | Recruitment approach |
| hiring_decision | Final hiring result |

---

## Machine Learning Models Used

The following classification algorithms were trained and compared:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- XGBoost Classifier

---

## Model Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score

The best performing model is saved as:

```

models/best_model.joblib

````

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
````

Install required libraries:

```bash
pip install -r requirements.txt
```

---

## Training the Model

Run the training script from the project root:

```bash
python src/train.py
```

After training, the following files will be generated:

```
models/
├── best_model.joblib
├── scaler.joblib
├── encoder.joblib
└── target_encoder.joblib
```

---

## Running Streamlit Application

Start the application using:

```bash
streamlit run streamlit_app/app.py
```

The application allows users to enter candidate details and predicts:

```
Selected
```

or

```
Not Selected
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Joblib
* Streamlit
* Jupyter Notebook

---

## Future Improvements

* Add more recruitment datasets
* Deploy the application online
* Add model explainability using SHAP
* Implement candidate ranking system
* Improve prediction confidence display

```
```
