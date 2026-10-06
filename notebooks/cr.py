from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test):
    
    y_pred = model.predict(X_test)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    

from sklearn.model_selection import StratifiedKFold, cross_validate
def cross_validate_model(model, X, y):
    
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )
    
    scoring = {
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }
    
    results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
    )
    
    print("Mean Precision:", results["test_precision"].mean())
    print("Mean Recall:", results["test_recall"].mean())
    print("Mean F1:", results["test_f1"].mean())
    print("Mean ROC-AUC:", results["test_roc_auc"].mean())
    
    
    
    