import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)



def evaluate_model(
    y_test,
    y_pred
):
    """
    Return model evaluation metrics
    """

    results = {

        "accuracy":
            accuracy_score(
                y_test,
                y_pred
            ),

        "precision":
            precision_score(
                y_test,
                y_pred
            ),

        "recall":
            recall_score(
                y_test,
                y_pred
            ),

        "f1_score":
            f1_score(
                y_test,
                y_pred
            )
    }


    return results



def get_classification_report(
    y_test,
    y_pred
):
    """
    Return detailed classification report
    """

    return classification_report(
        y_test,
        y_pred
    )



def save_confusion_matrix(
    y_test,
    y_pred,
    save_path="../reports/confusion_matrix.png"
):
    """
    Save confusion matrix plot
    """

    cm = confusion_matrix(
        y_test,
        y_pred
    )


    plt.figure(
        figsize=(6,5)
    )


    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )


    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.title(
        "Confusion Matrix"
    )


    plt.savefig(
        save_path,
        bbox_inches="tight",
        dpi=300
    )


    plt.close()



def calculate_auc(
    y_test,
    probability
):
    """
    Calculate ROC-AUC score
    """

    return roc_auc_score(
        y_test,
        probability
    )



def save_roc_curve(
    y_test,
    probability,
    save_path="../reports/roc_auc_curve.png"
):
    """
    Save ROC curve
    """

    auc_score = roc_auc_score(
        y_test,
        probability
    )


    fpr, tpr, _ = roc_curve(
        y_test,
        probability
    )


    plt.figure(
        figsize=(8,6)
    )


    plt.plot(
        fpr,
        tpr,
        label=f"AUC = {auc_score:.2f}"
    )


    plt.plot(
        [0,1],
        [0,1],
        linestyle="--",
        color="red"
    )


    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curve"
    )


    plt.legend()


    plt.savefig(
        save_path,
        bbox_inches="tight",
        dpi=300
    )


    plt.close()




































# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     classification_report,
#     confusion_matrix,
#     roc_auc_score
# )


# def evaluate_model(y_test, y_pred):
#     """
#     Calculate model performance metrics
#     """

#     results = {
#         "accuracy": accuracy_score(y_test, y_pred),
#         "precision": precision_score(y_test, y_pred),
#         "recall": recall_score(y_test, y_pred),
#         "f1_score": f1_score(y_test, y_pred)
#     }

#     return results



# def print_classification_report(y_test, y_pred):
#     """
#     Print detailed classification report
#     """

#     print(
#         classification_report(
#             y_test,
#             y_pred
#         )
#     )



# def get_confusion_matrix(y_test, y_pred):
#     """
#     Return confusion matrix
#     """

#     return confusion_matrix(
#         y_test,
#         y_pred
#     )



# def calculate_auc(y_test, probability):
#     """
#     Calculate ROC-AUC score
#     """

#     return roc_auc_score(
#         y_test,
#         probability
#     )


