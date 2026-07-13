import os
os.environ["OMP_NUM_THREADS"] = "1"
import json
import logging
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from scipy.sparse import hstack

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports" / "2_ML_Classification"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    logger.info("Loading training and testing data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv", engine='python', on_bad_lines='skip')
    test_df = pd.read_csv(DATA_DIR / "test.csv", engine='python', on_bad_lines='skip')
    
    with open(ARTIFACTS_DIR / "feature_metadata.json", "r") as f:
        feature_metadata = json.load(f)
        
    tfidf_vectorizer = joblib.load(ARTIFACTS_DIR / "tfidf_vectorizer.pkl")
    label_encoder = joblib.load(ARTIFACTS_DIR / "label_encoder.pkl")
    
    return train_df, test_df, feature_metadata, tfidf_vectorizer, label_encoder

def prepare_features(train_df, test_df, feature_metadata, tfidf_vectorizer):
    logger.info("Preparing features...")
    text_col = feature_metadata["text_column_for_tfidf"]
    num_cols = feature_metadata["numeric_feature_columns"]
    
    # TF IDF Features
    # Fill empty values
    X_train_text = train_df[text_col].fillna("")
    X_test_text = test_df[text_col].fillna("")
    
    X_train_tfidf = tfidf_vectorizer.transform(X_train_text)
    X_test_tfidf = tfidf_vectorizer.transform(X_test_text)
    
    # Numeric Features
    X_train_num = train_df[num_cols].fillna(0)
    X_test_num = test_df[num_cols].fillna(0)
    
    scaler = StandardScaler()
    X_train_num_scaled = scaler.fit_transform(X_train_num)
    X_test_num_scaled = scaler.transform(X_test_num)
    
    # Save scaler
    joblib.dump(scaler, ARTIFACTS_DIR / "scaler.pkl")
    
    # Combine Features
    X_train = hstack([X_train_num_scaled, X_train_tfidf]).tocsr()
    X_test = hstack([X_test_num_scaled, X_test_tfidf]).tocsr()
    
    y_train = train_df[feature_metadata["label_column_encoded"]]
    y_test = test_df[feature_metadata["label_column_encoded"]]
    
    return X_train, X_test, y_train, y_test

def train_and_evaluate(X_train, X_test, y_train, y_test, class_names):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_estimators=50, max_depth=3, tree_method='hist')
    }
    
    results = []
    best_f1 = 0
    best_model_name = ""
    best_model = None
    best_y_pred = None
    best_y_proba = None
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        
        logger.info(f"Evaluating {name}...")
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
        else:
            y_proba = None
            
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision (Macro)": prec,
            "Recall (Macro)": rec,
            "F1 Score (Macro)": f1
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model = model
            best_y_pred = y_pred
            best_y_proba = y_proba
            
    results_df = pd.DataFrame(results)
    logger.info(f"Best Model: {best_model_name} with F1: {best_f1:.4f}")
    
    return results_df, best_model_name, best_model, best_y_pred, best_y_proba

def generate_reports_and_visualizations(best_model_name, best_model, best_y_pred, best_y_proba, y_test, class_names, results_df):
    logger.info("Generating reports and visualizations...")
    
    # Save model comparison
    results_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    
    # Save classification report
    report_dict = classification_report(y_test, best_y_pred, labels=range(len(class_names)), target_names=class_names, output_dict=True, zero_division=0)
    report_str = classification_report(y_test, best_y_pred, labels=range(len(class_names)), target_names=class_names, zero_division=0)
    
    with open(REPORTS_DIR / "classification_report.md", "w") as f:
        f.write(f"# Classification Report (Best Model: {best_model_name})\n\n")
        f.write("```text\n")
        f.write(report_str)
        f.write("\n```\n")
        
    # Save confusion matrix
    cm = confusion_matrix(y_test, best_y_pred, labels=range(len(class_names)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=300)
    plt.close()
    
    # Save ROC curves
    if best_y_proba is not None:
        y_test_bin = label_binarize(y_test, classes=range(len(class_names)))
        n_classes = y_test_bin.shape[1]
        
        plt.figure(figsize=(10, 8))
        for i in range(min(n_classes, 10)): # Plot top 10 classes to avoid clutter
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], best_y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{class_names[i][:15]} (AUC = {roc_auc:.2f})')
            
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - {best_model_name} (Top 10 Classes)')
        plt.legend(loc="lower right", fontsize='small', ncol=2)
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "roc_curves.png", dpi=300)
        plt.close()

def main():
    train_df, test_df, feature_metadata, tfidf_vectorizer, label_encoder = load_data()
    class_names = feature_metadata["class_names"]
    
    X_train, X_test, y_train, y_test = prepare_features(train_df, test_df, feature_metadata, tfidf_vectorizer)
    
    results_df, best_model_name, best_model, best_y_pred, best_y_proba = train_and_evaluate(
        X_train, X_test, y_train, y_test, class_names
    )
    
    generate_reports_and_visualizations(
        best_model_name, best_model, best_y_pred, best_y_proba, y_test, class_names, results_df
    )
    
    # Save best model
    logger.info("Saving best model...")
    joblib.dump(best_model, ARTIFACTS_DIR / "best_model.pkl")
    
    logger.info("Module 3 Training Completed Successfully.")

if __name__ == "__main__":
    main()
